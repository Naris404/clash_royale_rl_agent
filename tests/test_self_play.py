"""Past-self pool: seeding from selection checkpoints, admission of candidates, determinism."""

from __future__ import annotations

from pathlib import Path

import pytest
from sb3_contrib import MaskablePPO

from cr_rl.env.gym_env import ClashRoyaleEnv
from cr_rl.training.matches import MatchResult
from cr_rl.training.self_play import (
    ADMISSION_MATCHES,
    PastSelfPool,
    admission_seeds,
    load_past_self,
)
from cr_rl.training.selection import CheckpointResult


def stub_play(wins: int):
    """A stand-in for the admission matches: the first `wins` of the seeds are wins."""

    def play(candidate, members, seeds):
        return [MatchResult(seed, 0 if i < wins else 1, 0.0, 1) for i, seed in enumerate(seeds)]

    return play


def pool_of(*names: str, run_seed: int = 0) -> PastSelfPool:
    return PastSelfPool([Path(name) for name in names], run_seed=run_seed)


class TestAdmission:
    def test_a_candidate_with_45_percent_replaces_the_oldest_past_self(self):
        pool = pool_of("a.zip", "b.zip", "c.zip")
        decision = pool.consider(Path("new.zip"), 100_000, play=stub_play(wins=23))  # 23 of 50
        assert decision.joined and decision.replaced == "a.zip"
        assert pool.members == [Path("b.zip"), Path("c.zip"), Path("new.zip")]

    def test_a_candidate_below_45_percent_does_not_join(self):
        pool = pool_of("a.zip", "b.zip", "c.zip")
        decision = pool.consider(Path("new.zip"), 100_000, play=stub_play(wins=22))  # 44%
        assert not decision.joined and decision.replaced is None
        assert pool.members == [Path("a.zip"), Path("b.zip"), Path("c.zip")]

    def test_the_pool_keeps_replacing_the_oldest(self):
        pool = pool_of("a.zip", "b.zip", "c.zip")
        pool.consider(Path("d.zip"), 100_000, play=stub_play(wins=50))
        pool.consider(Path("e.zip"), 200_000, play=stub_play(wins=50))
        assert pool.members == [Path("c.zip"), Path("d.zip"), Path("e.zip")]

    def test_the_candidate_plays_50_matches(self):
        played = []

        def play(candidate, members, seeds):
            played.append(len(seeds))
            return []

        pool_of("a.zip").consider(Path("new.zip"), 100_000, play=play)
        assert played == [ADMISSION_MATCHES] == [50]

    def test_decisions_are_recorded(self):
        pool = pool_of("a.zip", "b.zip", "c.zip")
        pool.consider(Path("new.zip"), 100_000, play=stub_play(wins=30))
        (decision,) = pool.decisions
        assert (decision.step, decision.wins, decision.matches) == (100_000, 30, 50)


class TestSeeding:
    def test_the_pool_starts_from_the_last_three_selection_checkpoints(self, tmp_path):
        records = [
            CheckpointResult(step, 50, 100, f"ppo_cr_{step}_steps.zip") for step in range(1, 6)
        ]
        pool = PastSelfPool.from_selection(records, tmp_path, run_seed=0)
        assert pool.members == [tmp_path / f"ppo_cr_{step}_steps.zip" for step in (3, 4, 5)]


class TestReproducibility:
    def test_admission_seeds_depend_only_on_run_seed_and_step(self):
        assert admission_seeds(7, 100_000) == admission_seeds(7, 100_000)
        assert admission_seeds(7, 100_000) != admission_seeds(8, 100_000)
        assert admission_seeds(7, 100_000) != admission_seeds(7, 200_000)

    def test_admission_seeds_are_distinct_and_there_are_fifty(self):
        seeds = admission_seeds(7, 100_000)
        assert len(seeds) == len(set(seeds)) == 50

    def test_two_pools_with_the_same_seed_make_the_same_decisions(self):
        def run():
            pool = pool_of("a.zip", "b.zip", "c.zip", run_seed=3)
            seen = []

            def play(candidate, members, seeds):
                seen.append(list(seeds))
                return [MatchResult(seed, seed % 2, 0.0, 1) for seed in seeds]

            for step in (100_000, 200_000):
                pool.consider(Path(f"{step}.zip"), step, play=play)
            return pool.decisions, seen

        assert run() == run()


class TestPastSelfPlay:
    @pytest.fixture(scope="class")
    def checkpoint(self, tmp_path_factory) -> Path:
        path = tmp_path_factory.mktemp("past") / "past.zip"
        MaskablePPO("MlpPolicy", ClashRoyaleEnv(seed=0), n_steps=64, batch_size=64).save(str(path))
        return path

    def test_a_past_self_plays_legal_actions_as_player_1(self, checkpoint):
        asked = []

        class Recorder:
            """The loaded policy, recording each action it picks with the mask it was given."""

            def __init__(self, model):
                self.model = model

            def predict(self, obs, deterministic, action_masks):
                action, state = self.model.predict(obs, deterministic=deterministic, action_masks=action_masks)
                asked.append((deterministic, bool(action_masks[action])))
                return action, state

        env = ClashRoyaleEnv(seed=0, opponent=Recorder(load_past_self(checkpoint)))
        env.reset()
        assert env.opponent_name == "past-self"
        for _ in range(100):
            _, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break
        assert asked and all(deterministic and legal for deterministic, legal in asked)

    def test_real_admission_matches_are_reproducible_from_seed_and_files(self, checkpoint):
        def run():
            pool = PastSelfPool([checkpoint] * 3, run_seed=5, matches=3)
            return pool.consider(checkpoint, 100_000)

        decision = run()
        assert decision.matches == 3 and decision == run()
