"""Reward: match outcome plus potential-based Crown and tower-HP shaping, per decision step."""

from __future__ import annotations

import numpy as np
import pytest

from cr_rl.env.gym_env import ClashRoyaleEnv
from cr_rl.game.board import MAX_TOWER_HP_PER_PLAYER, OVERTIME_TIME, REGULATION_TIME, TICK_DT, Board
from cr_rl.game.rewards import DEFAULT_GAMMA, RewardWeights, potential, step_reward

OUTCOME_ONLY = RewardWeights(win=1.0, crown=0.0, hp=0.0)
TOWER_HP = 3052.0  # princess tower


class Passive:
    """Opponent that never plays."""

    def reset(self) -> None:
        pass

    def choose_action(self, board, player) -> int:
        return 0


def _env(weights: RewardWeights = RewardWeights(), gamma: float = DEFAULT_GAMMA) -> ClashRoyaleEnv:
    env = ClashRoyaleEnv(seed=0, opponent=Passive(), reward_weights=weights, gamma=gamma)
    env.reset(seed=0)
    return env


def _tower(board: Board, owner: int, name: str, side: str | None = None):
    return next(
        t
        for t in board.towers
        if t.owner == owner and t.name == name and (side is None or (t.x < 9.0) == (side == "L"))
    )


def _destroy(board: Board, owner: int, name: str, side: str | None = None) -> None:
    tower = _tower(board, owner, name, side)
    tower.hp = 0.0
    tower.alive = False


def _play_match(env: ClashRoyaleEnv, seed: int) -> list[float]:
    """Play a whole match with random legal actions and return every step's reward."""
    rng = np.random.default_rng(seed)
    rewards, done = [], False
    while not done:
        action = int(rng.choice(np.flatnonzero(env.action_masks())))
        _, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
    return rewards


class TestOutcome:
    def _last_reward(self, env: ClashRoyaleEnv) -> float:
        _, reward, terminated, truncated, _ = env.step(0)
        assert terminated or truncated
        return reward

    def test_win_by_crowns_at_the_end_of_regulation(self):
        env = _env(OUTCOME_ONLY)
        _destroy(env.board, 1, "Tower", "L")
        env.board.time = REGULATION_TIME - TICK_DT
        assert self._last_reward(env) == 1.0

    def test_loss_by_crowns_at_the_end_of_regulation(self):
        env = _env(OUTCOME_ONLY)
        _destroy(env.board, 0, "Tower", "L")
        env.board.time = REGULATION_TIME - TICK_DT
        assert self._last_reward(env) == -1.0

    def test_first_crown_in_overtime_decides_the_match(self):
        env = _env(OUTCOME_ONLY)
        env.board.time = REGULATION_TIME + 5.0
        _destroy(env.board, 1, "Tower", "R")
        assert self._last_reward(env) == 1.0

    def test_tiebreak_goes_to_the_weaker_weakest_tower(self):
        env = _env(OUTCOME_ONLY)
        env.board.time = REGULATION_TIME + OVERTIME_TIME - TICK_DT
        _tower(env.board, 0, "Tower", "L").hp = 500.0
        assert self._last_reward(env) == -1.0

    def test_equal_weakest_towers_at_the_end_of_overtime_are_a_draw(self):
        env = _env(OUTCOME_ONLY)
        env.board.time = REGULATION_TIME + OVERTIME_TIME - TICK_DT
        assert self._last_reward(env) == 0.0

    def test_king_win_before_the_clock(self):
        env = _env(OUTCOME_ONLY)
        _destroy(env.board, 1, "King_Tower")
        assert self._last_reward(env) == 1.0

    def test_no_outcome_before_the_last_step(self):
        env = _env(OUTCOME_ONLY)
        _destroy(env.board, 1, "Tower", "L")  # ahead on crowns, but the match goes on
        assert env.step(0)[1] == 0.0


class TestShaping:
    def test_enemy_tower_damage_pays_hp_shaping_at_the_discount(self):
        env = _env(RewardWeights(win=1.0, crown=0.2, hp=0.5))
        _tower(env.board, 1, "Tower", "L").hp -= 1000.0
        reward = env.step(0)[1]
        assert reward == pytest.approx(0.5 * DEFAULT_GAMMA * 1000.0 / MAX_TOWER_HP_PER_PLAYER)

    def test_destroying_a_princess_tower_pays_crown_and_hp_shaping(self):
        env = _env(RewardWeights(win=1.0, crown=0.2, hp=0.5))
        _destroy(env.board, 1, "Tower", "L")
        reward = env.step(0)[1]
        assert reward == pytest.approx(DEFAULT_GAMMA * (0.2 + 0.5 * TOWER_HP / MAX_TOWER_HP_PER_PLAYER))

    def test_losing_a_tower_costs_the_same(self):
        env = _env(RewardWeights(win=1.0, crown=0.2, hp=0.5))
        _destroy(env.board, 0, "Tower", "R")
        reward = env.step(0)[1]
        assert reward == pytest.approx(-DEFAULT_GAMMA * (0.2 + 0.5 * TOWER_HP / MAX_TOWER_HP_PER_PLAYER))

    def test_undiscounted_shaping_sums_to_zero_over_a_match(self):
        env = ClashRoyaleEnv(seed=2, opponent="logic", reward_weights=RewardWeights(win=0.0), gamma=1.0)
        env.reset(seed=2)
        rewards = _play_match(env, seed=2)
        assert any(r != 0.0 for r in rewards)  # the shaping did fire
        assert sum(rewards) == pytest.approx(0.0, abs=1e-9)

    def test_undiscounted_return_is_the_outcome_with_shaping_on(self):
        env = ClashRoyaleEnv(seed=3, opponent="logic", gamma=1.0)
        env.reset(seed=3)
        rewards = _play_match(env, seed=3)
        assert sum(rewards) == pytest.approx({0: 1.0, 1: -1.0, None: 0.0}[env.board.winner], abs=1e-9)

    def test_zero_shaping_weights_give_the_outcome_only_reward(self):
        env = ClashRoyaleEnv(seed=2, opponent="logic", reward_weights=OUTCOME_ONLY)
        env.reset(seed=2)
        rewards = _play_match(env, seed=2)
        assert all(r == 0.0 for r in rewards[:-1])
        assert rewards[-1] in (-1.0, 0.0, 1.0)


class TestSymmetry:
    def test_both_players_get_negated_rewards_for_the_same_transition(self):
        weights, gamma = RewardWeights(), 0.99
        board = Board(seed=0)
        before = [potential(board, p, weights) for p in (0, 1)]
        _tower(board, 1, "Tower", "L").hp -= 700.0
        _destroy(board, 0, "Tower", "R")
        rewards = [step_reward(board, p, weights, gamma, before[p]) for p in (0, 1)]
        assert rewards[0] != 0.0
        assert rewards[0] == pytest.approx(-rewards[1])

    def test_terminal_transition_is_symmetric_too(self):
        weights, gamma = RewardWeights(), 0.99
        board = Board(seed=0)
        before = [potential(board, p, weights) for p in (0, 1)]
        _destroy(board, 1, "King_Tower")
        board.step(0, action_p1=0)
        rewards = [step_reward(board, p, weights, gamma, before[p]) for p in (0, 1)]
        assert rewards[0] == pytest.approx(1.0)
        assert rewards[0] == pytest.approx(-rewards[1])
