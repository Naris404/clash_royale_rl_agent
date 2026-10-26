"""Arrows experiment: how a model uses Arrows on a tower, and what blocking that play costs."""

from __future__ import annotations

import json

import pytest
import torch
from sb3_contrib import MaskablePPO

from cr_rl.coach.inference import DEFAULT_MODEL_PATH
from cr_rl.env.gym_env import ClashRoyaleEnv
from cr_rl.experiments.arrows import TOWER_ZONES, run_arrows_experiment, save_arrows_results
from cr_rl.game.board import DEPLOY_ZONES, NUM_ZONES
from cr_rl.training.matches import play_matches, wilson_interval

SEEDS = range(2_000_000, 2_000_002)
TOWER_ZONE_INDEXES = [i for i, zone in enumerate(DEPLOY_ZONES[0]) if zone.name in TOWER_ZONES]


def spell_on_tower_lover(seed: int = 0) -> MaskablePPO:
    """An untrained policy pushed to prefer any spell aimed at a tower, so Arrows on a tower shows up."""
    model = MaskablePPO("MlpPolicy", ClashRoyaleEnv(seed=0), n_steps=64, batch_size=64, seed=seed, device="cpu")
    with torch.no_grad():
        for slot in range(4):
            for zone in TOWER_ZONE_INDEXES:
                model.policy.action_net.bias[1 + slot * NUM_ZONES + zone] += 10.0
    return model


def frozen_opponent() -> MaskablePPO:
    return MaskablePPO("MlpPolicy", ClashRoyaleEnv(seed=0), n_steps=64, batch_size=64, seed=7, device="cpu")


@pytest.fixture(scope="module")
def policies():
    return spell_on_tower_lover(), frozen_opponent()


@pytest.fixture(scope="module")
def results(policies):
    return run_arrows_experiment(*policies, seeds=SEEDS, model_name="m.zip", frozen_name="f.zip")


class TestTowerPlays:
    def test_tower_zones_are_the_three_spell_tower_zones(self):
        assert TOWER_ZONES == ("spell-tower-L", "spell-tower-R", "spell-king")

    def test_every_decision_records_the_state_and_the_outcome(self, results):
        decisions = results.profile.decisions
        assert decisions, "the biased policy must play Arrows on a tower"
        for decision in decisions:
            assert decision.seed in SEEDS
            assert decision.zone in TOWER_ZONES
            assert decision.own_elixir >= 3.0  # Arrows costs 3
            assert 0.0 <= decision.enemy_elixir <= 10.0
            assert 0.0 < decision.p_chosen <= 1.0
            assert 0.0 <= decision.p_noop < 1.0
            assert decision.outcome in ("win", "loss", "draw")
            assert decision.time >= 0.0

    def test_profile_counts_plays_per_zone_and_the_tower_share(self, results):
        profile = results.profile
        tower_plays = sum(count for zone, count in profile.plays_per_zone.items() if zone in TOWER_ZONES)
        assert tower_plays == len(profile.decisions) > 0
        assert profile.tower_share == tower_plays / sum(profile.plays_per_zone.values())
        assert profile.matches == len(SEEDS)
        assert profile.matches_with_tower_play == len({d.seed for d in profile.decisions})


class TestBlocking:
    def test_blocked_matches_never_play_arrows_on_a_tower(self, results):
        blocked = results.blocked_profile
        assert blocked.decisions == ()
        assert all(zone not in TOWER_ZONES for zone in blocked.plays_per_zone)

    def test_win_rates_sit_next_to_each_other_with_wilson_intervals(self, results):
        assert set(results.win_rates) == {"logic", "f.zip"}
        for pair in results.win_rates.values():
            for report in (pair.allowed, pair.blocked):
                assert report.episodes == len(SEEDS)
                assert (report.ci_low, report.ci_high) == wilson_interval(report.wins, report.episodes)

    def test_the_allowed_logic_win_rate_matches_a_plain_evaluation(self, policies, results):
        wins = sum(result.winner == 0 for result in play_matches(policies[0], "logic", SEEDS))
        assert results.win_rates["logic"].allowed.wins == wins

    def test_is_deterministic(self, policies, results):
        assert run_arrows_experiment(*policies, seeds=SEEDS, model_name="m.zip", frozen_name="f.zip") == results


def test_results_file_holds_the_profile_and_both_win_rates(tmp_path, results):
    path = tmp_path / "results.json"
    save_arrows_results(results, path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["experiment"] == "arrows"
    assert saved["metadata"]["model"] == "m.zip"
    assert saved["metadata"]["episodes"] == len(SEEDS)
    assert saved["profile"]["decisions"][0]["zone"] in TOWER_ZONES
    assert set(saved["win_rates"]) == {"logic", "f.zip"}
    assert {"allowed", "blocked"} == set(saved["win_rates"]["logic"])


@pytest.mark.slow
class TestRealCheckpoint:
    """The Shipped model (models/ppo_cr_best.zip, gitignored): skipped on a clone without it."""

    @pytest.fixture(scope="class")
    def shipped(self):
        if not DEFAULT_MODEL_PATH.is_file():
            pytest.skip(f"no Shipped model at {DEFAULT_MODEL_PATH}")
        return MaskablePPO.load(str(DEFAULT_MODEL_PATH))

    def test_two_runs_on_the_same_model_are_identical(self, shipped):
        assert run_arrows_experiment(shipped, shipped, seeds=SEEDS) == run_arrows_experiment(shipped, shipped, seeds=SEEDS)

    def test_with_blocking_on_no_arrows_on_a_tower_is_ever_played(self, shipped):
        results = run_arrows_experiment(shipped, shipped, seeds=SEEDS)
        assert results.profile.decisions, "the Shipped model plays Arrows on a tower when nothing blocks it"
        assert results.blocked_profile.decisions == ()
        assert all(zone not in TOWER_ZONES for zone in results.blocked_profile.plays_per_zone)
