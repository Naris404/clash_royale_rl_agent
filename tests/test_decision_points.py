"""Decision points: the env asks the agent only when a card play is legal and no noop hold runs."""

from __future__ import annotations

import pytest

from cr_rl.env.gym_env import ClashRoyaleEnv
from cr_rl.agents.logic import LogicAgent
from cr_rl.game.board import TICK_DT, Board


def _ticks_between(env: ClashRoyaleEnv, action: int) -> int:
    before = env.board.time
    env.step(action)
    return round((env.board.time - before) / TICK_DT)


class TestNoopHold:
    def test_noop_holds_three_ticks_by_default(self):
        env = ClashRoyaleEnv(seed=0, opponent="random")
        env.reset(seed=0)
        assert _ticks_between(env, 0) == 3

    def test_noop_only_state_is_never_returned(self):
        env = ClashRoyaleEnv(seed=3, opponent="logic")
        env.reset(seed=3)
        done = False
        while not done:
            mask = env.action_masks()
            assert mask[1:].any()
            _, _, terminated, truncated, _ = env.step(0)  # always wait: elixir hits the cap
            done = terminated or truncated

    def test_reset_clears_a_hold_left_over_from_the_previous_match(self):
        used, fresh = ClashRoyaleEnv(seed=0), ClashRoyaleEnv(seed=0)
        used.reset(seed=0)
        done = False
        while not done:
            _, _, terminated, truncated, _ = used.step(0)
            done = terminated or truncated
        used.reset(seed=0)
        fresh.reset(seed=0)
        play = int(fresh.action_masks()[1:].argmax()) + 1
        used.step(play)
        fresh.step(play)
        assert used.board.time == fresh.board.time

    def test_hold_is_configurable_and_one_gives_per_tick_queries(self):
        env = ClashRoyaleEnv(seed=0, opponent="random", hold_ticks=1)
        env.reset(seed=0)
        assert _ticks_between(env, 0) == 1
        env = ClashRoyaleEnv(seed=0, opponent="random", hold_ticks=5)
        env.reset(seed=0)
        assert _ticks_between(env, 0) == 5


class TestReward:
    def test_skipped_tick_rewards_sum_to_the_per_tick_total(self):
        def play_match(hold_ticks: int) -> float:
            env = ClashRoyaleEnv(seed=1, hold_ticks=hold_ticks)
            env.reset(seed=1)
            total, done = 0.0, False
            while not done:
                _, reward, terminated, truncated, _ = env.step(0)
                total += reward
                done = terminated or truncated
            return total

        # The agent never plays, so only the rule-based bot deals damage; the reference
        # ticks the Board directly, one reward per tick.
        board, bot = Board(seed=1), LogicAgent()
        board.reset(seed=1)
        per_tick = 0.0
        while not board.done:
            per_tick += board.step(0, action_p1=bot.choose_action(board, player=1)).reward
        assert per_tick != 0.0
        assert play_match(3) == pytest.approx(per_tick)
        assert play_match(1) == pytest.approx(per_tick)
