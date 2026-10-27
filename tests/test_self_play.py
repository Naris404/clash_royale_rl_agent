"""Past-self pool: seeding from selection checkpoints, admission of candidates, determinism."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from sb3_contrib import MaskablePPO

from cr_rl.env.gym_env import ClashRoyaleEnv
from cr_rl.training.matches import MatchResult
from cr_rl.training.phases import PAST_SELF
from cr_rl.training.self_play import (
    ADMISSION_MATCHES,
    PastSelfPool,
    admission_seeds,
    load_past_self,
    member_seeds,
    pfsp_weights,
)
from cr_rl.training.selection import CheckpointResult


def stub_play(wins: int):
    """A stand-in for the admission matches: the first `wins` of the seeds are wins."""

    def play(candidate, members, seeds):
        return [MatchResult(seed, 0 if i < wins else 1, 0.0, 1) for i, seed in enumerate(seeds)]

    return play


def pool_of(*names: str, run_seed: int = 0, **kwargs) -> PastSelfPool:
    return PastSelfPool([Path(name) for name in names], run_seed=run_seed, **kwargs)


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


class TestPoolSize:
    def test_a_pool_below_its_size_grows_without_dropping_anyone(self):
        pool = pool_of("a.zip", size=3)
        decision = pool.consider(Path("b.zip"), 100_000, play=stub_play(wins=50))
        assert decision.joined and decision.replaced is None
        assert pool.members == [Path("a.zip"), Path("b.zip")]

    def test_a_pool_of_six_drops_the_oldest_only_when_full(self):
        pool = pool_of(*[f"{c}.zip" for c in "abcdef"], size=6)
        pool.consider(Path("g.zip"), 100_000, play=stub_play(wins=50))
        assert [m.name for m in pool.members] == [f"{c}.zip" for c in "bcdefg"]

    def test_the_size_defaults_to_three(self):
        pool = pool_of("a.zip", "b.zip", "c.zip")
        pool.consider(Path("d.zip"), 100_000, play=stub_play(wins=50))
        assert len(pool.members) == 3


class TestPermanentMembers:
    def test_permanent_members_are_in_the_pool_and_play_the_admission_check(self):
        pool = pool_of("a.zip", permanent=[Path("exploiter.zip")])
        seen = []

        def play(candidate, members, seeds):
            seen.append(list(members))
            return []

        pool.consider(Path("new.zip"), 100_000, play=play)
        assert seen == [[Path("a.zip"), Path("exploiter.zip")]]

    def test_permanent_members_do_not_count_towards_the_size(self):
        pool = pool_of("a.zip", "b.zip", "c.zip", permanent=[Path("x.zip"), Path("y.zip")])
        decision = pool.consider(Path("d.zip"), 100_000, play=stub_play(wins=50))
        assert decision.replaced == "a.zip"
        assert pool.members == [Path(n) for n in ("b.zip", "c.zip", "d.zip", "x.zip", "y.zip")]

    def test_permanent_members_survive_every_admission_check(self):
        pool = pool_of("a.zip", "b.zip", "c.zip", permanent=[Path("x.zip")])
        for step in range(1, 8):
            pool.consider(Path(f"new{step}.zip"), step * 100_000, play=stub_play(wins=50))
        assert Path("x.zip") in pool.members
        assert len(pool.members) == 4

    def test_the_decision_lists_the_permanent_members_too(self):
        pool = pool_of("a.zip", permanent=[Path("x.zip")])
        decision = pool.consider(Path("new.zip"), 100_000, play=stub_play(wins=0))
        assert decision.members == ("a.zip", "x.zip")


class TestMemberWinRates:
    def test_the_decision_records_the_candidates_win_rate_against_each_member(self):
        pool = pool_of("a.zip", "b.zip", permanent=[Path("x.zip")], matches=6)

        def play(candidate, members, seeds):
            """Wins every match against the first member, half against the second, none against the third."""
            results = []
            for i, seed in enumerate(seeds):
                member = i % 3  # the rotation `play_against_pool` plays the seeds in
                winner = 0 if member == 0 or (member == 1 and i % 2 == 0) else 1
                results.append(MatchResult(seed, winner, 0.0, 1))
            return results

        decision = pool.consider(Path("new.zip"), 100_000, play=play)
        assert decision.member_win_rates == {"a.zip": 1.0, "b.zip": pytest.approx(0.5), "x.zip": 0.0}

    def test_a_member_that_played_no_match_has_no_win_rate(self):
        pool = pool_of("a.zip", "b.zip")
        decision = pool.consider(Path("new.zip"), 100_000, play=lambda c, m, seeds: [])
        assert decision.member_win_rates == {}

    def test_the_log_file_has_the_win_rates(self, tmp_path):
        pool = pool_of("a.zip")
        pool.consider(Path("new.zip"), 100_000, play=stub_play(wins=25))
        pool.save_log(tmp_path / "pool.json")
        (record,) = json.loads((tmp_path / "pool.json").read_text(encoding="utf-8"))
        assert record["member_win_rates"] == {"a.zip": 0.5}


def measured_pool(rates: dict[str, float], *, permanent: tuple[str, ...] = (), run_seed: int = 0) -> PastSelfPool:
    """A pool whose admission check measured the given win rate against each member (10 matches each)."""
    past = [name for name in rates if name not in permanent]
    pool = pool_of(*past, permanent=[Path(name) for name in permanent], run_seed=run_seed, matches=10 * len(rates), size=99)

    def play(candidate, members, seeds):
        results = []
        for member, share in zip(members, member_seeds(seeds, len(members))):
            wins = round(rates[member.name] * len(share))
            results += [MatchResult(seed, 0 if i < wins else 1, 0.0, 1) for i, seed in enumerate(share)]
        return results

    pool.consider(Path("candidate.zip"), 100_000, play=play)
    pool.past_selves = [member for member in pool.past_selves if member.name != "candidate.zip"]
    pool.size = len(pool.past_selves)
    return pool


def draw_counts(pool: PastSelfPool, n: int = 1000, step: int = 200_000) -> Counter:
    return Counter(path.name for path in pool.assign(n, step))


class TestPfsp:
    def test_weights_are_one_minus_the_win_rate_squared_and_one_when_unmeasured(self):
        assert pfsp_weights([0.0, 0.5, 1.0, None]) == [1.0, 0.25, 0.0, 1.0]

    def test_harder_members_are_drawn_more_often(self):
        counts = draw_counts(measured_pool({"easy.zip": 0.9, "hard.zip": 0.1}))
        assert counts["hard.zip"] > 20 * counts["easy.zip"]

    def test_an_unmeasured_member_weighs_one(self):
        pool = measured_pool({"a.zip": 0.5})
        pool.past_selves.append(Path("fresh.zip"))  # joined after the check: no win rate yet
        counts = draw_counts(pool)
        assert counts["fresh.zip"] > 3 * counts["a.zip"]  # weights 1 vs 0.25

    def test_a_permanent_member_is_always_eligible(self):
        pool = measured_pool({"a.zip": 0.0, "exploiter.zip": 0.5}, permanent=("exploiter.zip",))
        assert draw_counts(pool)["exploiter.zip"] > 0

    def test_a_pool_the_agent_always_beats_is_drawn_uniformly(self):
        counts = draw_counts(measured_pool({"a.zip": 1.0, "b.zip": 1.0}))
        assert set(counts) == {"a.zip", "b.zip"}

    def test_the_assignment_depends_only_on_run_seed_and_step(self):
        rates = {"a.zip": 0.4, "b.zip": 0.4, "c.zip": 0.4}
        pool = measured_pool(rates, run_seed=7)
        assert pool.assign(12, 300_000) == measured_pool(rates, run_seed=7).assign(12, 300_000)
        assert pool.assign(12, 300_000) != pool.assign(12, 400_000)
        assert pool.assign(12, 300_000) != measured_pool(rates, run_seed=8).assign(12, 300_000)

    def test_a_replaced_past_self_is_no_longer_drawn(self):
        pool = measured_pool({"a.zip": 0.0, "b.zip": 0.0, "c.zip": 0.0})
        pool.consider(Path("d.zip"), 200_000, play=stub_play(wins=50))
        assert "a.zip" not in draw_counts(pool)

    def test_a_pool_without_win_rates_is_drawn_uniformly(self):
        counts = draw_counts(pool_of("a.zip", "b.zip", "c.zip"))
        assert all(250 < count < 420 for count in counts.values()) and len(counts) == 3


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
        assert env.opponent_name == PAST_SELF
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
