"""Training opponents: named bots, ε-LogicAgent, frozen policies and per-episode seeds."""

from __future__ import annotations

import numpy as np
import pytest

from cr_rl.env.gym_env import ClashRoyaleEnv


def _opponent_plays(opponent, seed: int, steps: int = 150) -> list[str]:
    """Cards player 1 played while player 0 only waits."""
    env = ClashRoyaleEnv(seed=seed, opponent=opponent)
    env.reset(seed=seed)
    for _ in range(steps):
        _, _, terminated, truncated, _ = env.step(0)
        if terminated or truncated:
            break
    return list(env.board.play_history[1])


class TestEpsilonLogicAgent:
    def test_is_deterministic_under_a_fixed_seed(self):
        assert _opponent_plays("logic-eps", 7) == _opponent_plays("logic-eps", 7)

    def test_differs_from_logic_agent(self):
        for seed in range(3):
            assert _opponent_plays("logic-eps", seed) != _opponent_plays("logic", seed)

    def test_differs_across_seeds(self):
        plays = {tuple(_opponent_plays("logic-eps", seed)) for seed in range(4)}
        assert len(plays) > 1

    def test_plays_are_legal(self, caplog):
        _opponent_plays("logic-eps", 1)
        assert not caplog.records  # no illegal play fell back to the Board's default spot


class TestFrozenPolicy:
    @pytest.fixture(scope="class")
    def policy(self):
        from sb3_contrib import MaskablePPO

        return MaskablePPO("MlpPolicy", ClashRoyaleEnv(seed=0), n_steps=64, batch_size=64, device="cpu")

    def test_plays_legal_actions_as_player_1(self, policy, caplog):
        plays = _opponent_plays(policy, 0, steps=200)
        assert plays
        assert not caplog.records

    def test_is_asked_only_at_decision_points(self, policy):
        env = ClashRoyaleEnv(seed=0, opponent=policy)
        env.reset(seed=0)
        asked = []
        predict = policy.predict
        policy.predict = lambda *args, **kwargs: (asked.append(env.board.time), predict(*args, **kwargs))[1]
        try:
            for _ in range(40):
                env.step(0)
        finally:
            del policy.predict
        ticks = env.board.time * 10
        assert 0 < len(asked) < ticks / 2  # noop holds keep it from being asked every tick


def test_unknown_opponent_name_is_rejected():
    with pytest.raises(ValueError):
        ClashRoyaleEnv(seed=0, opponent="nope")


class TestEpisodeSeeds:
    @staticmethod
    def _opening_hands(env: ClashRoyaleEnv, episodes: int = 4) -> list[tuple[str, ...]]:
        hands = []
        for _ in range(episodes):
            env.reset()
            hands.append(tuple(env.board.get_hand(0)) + (env.board.get_next_card(0),))
        return hands

    def test_consecutive_episodes_use_different_seeds(self):
        env = ClashRoyaleEnv(seed=42)
        seeds = [env.reset()[1]["seed"] for _ in range(5)]
        assert len(set(seeds)) == 5
        assert len(set(self._opening_hands(env, 6))) > 1

    def test_seeds_are_reproducible_from_the_run_seed(self):
        env_a, env_b = ClashRoyaleEnv(seed=42), ClashRoyaleEnv(seed=42)
        assert [env_a.reset()[1]["seed"] for _ in range(3)] == [env_b.reset()[1]["seed"] for _ in range(3)]

    def test_envs_with_different_indices_or_run_seeds_diverge(self):
        seeds = {
            ClashRoyaleEnv(seed=run_seed, env_index=index).reset()[1]["seed"]
            for run_seed in (1, 2)
            for index in range(3)
        }
        assert len(seeds) == 6

    def test_an_explicit_reset_seed_is_used_as_given(self):
        assert ClashRoyaleEnv(seed=42).reset(seed=9)[1]["seed"] == 9

    def test_random_opponent_follows_the_episode_seed(self):
        """A later episode replays exactly like a fresh env reset with that episode's seed."""
        env = ClashRoyaleEnv(seed=5, opponent="random")
        env.reset()
        later_seed = env.reset()[1]["seed"]
        for _ in range(150):
            env.step(0)
        assert list(env.board.play_history[1]) == _opponent_plays("random", later_seed)
