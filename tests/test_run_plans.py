"""The final-configuration and Ablation run plans, end to end with tiny step and episode counts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cr_rl.experiments import arms
from cr_rl.experiments.ablations import run_ablations
from cr_rl.experiments.arms import ARMS
from cr_rl.experiments.final_configuration import run_final_configuration
from cr_rl.stats.reporting import load_experiment_results
from cr_rl.training.report import ENV_VERSION

REPORT_SEEDS = range(2_000_000, 2_000_002)
SMOKE = dict(
    n_envs=2,
    n_steps=512,
    selection_seeds=range(1_000_000, 1_000_001),
    pool_interval=100_000,
    pool_matches=2,
    report_seeds=REPORT_SEEDS,
)
SMOKE_STEPS = 1_024


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class TestFinalConfigurationPlan:
    @pytest.fixture(scope="class")
    def run(self, tmp_path_factory):
        out_dir = tmp_path_factory.mktemp("final")
        run_final_configuration(timesteps=SMOKE_STEPS, seeds=(0,), out_dir=out_dir, **SMOKE)
        return out_dir, read(out_dir / "results.json")

    def test_trains_one_run_per_seed_of_the_full_configuration(self, run):
        out_dir, results = run
        assert [path.name for path in sorted(out_dir.glob("full-seed*"))] == ["full-seed0"]
        assert read(out_dir / "full-seed0" / "manifest.json")["layout"] == "full"
        assert results["training_seeds"] == [0] and results["timesteps"] == SMOKE_STEPS

    def test_evaluates_each_run_on_report_seeds_and_states_the_ship_verdict(self, run):
        _, results = run
        assert [row["run"] for row in results["results"]] == ["full-seed0"] * 2
        assert {row["opponent"] for row in results["results"]} == {"logic", "random"}
        assert all(row["episodes"] == 2 for row in results["results"])
        assert results["shipped"] == "full-seed0"
        assert isinstance(results["ship_gate_passed"], bool)

    def test_reports_curriculum_metrics_for_every_run(self, run):
        _, results = run
        assert set(results["curriculum"]) == {"full-seed0"}
        metrics = results["curriculum"]["full-seed0"]
        assert set(metrics) == {"jumpstart", "steps_to_half", "report_win_rate"}

    def test_checks_the_budget_rule_on_the_first_run(self, run):
        _, results = run
        assert set(results["budget_rule"]) == {"start_win_rate", "end_win_rate", "rise", "extend"}

    def test_run_metadata_records_the_training_configuration(self, run):
        _, results = run
        metadata = results["runs"][0]["evaluation"]["metadata"]
        assert metadata["env_version"] == ENV_VERSION
        assert metadata["config"]["arm"] == "full"

    def test_results_are_read_by_the_statistics_tooling(self, run):
        out_dir, _ = run
        rows = load_experiment_results(out_dir.parent)
        assert len([row for row in rows if row["experiment"] == "final_configuration"]) == 2


class TestAblationPlan:
    @pytest.fixture(scope="class")
    def run(self, tmp_path_factory):
        out_dir = tmp_path_factory.mktemp("ablations")
        run_ablations(timesteps=SMOKE_STEPS, seeds=(0,), out_dir=out_dir, **SMOKE)
        return out_dir, read(out_dir / "results.json")

    def test_trains_every_arm_with_its_own_configuration(self, run):
        out_dir, _ = run
        for name, arm in ARMS.items():
            manifest = read(out_dir / f"{name}-seed0" / "manifest.json")
            assert (manifest["arm"], manifest["layout"]) == (name, arm.layout)
            assert manifest["reward_weights"]["crown"] == arm.reward_weights.crown

    def test_aggregates_each_arm_over_its_seeds(self, run):
        _, results = run
        assert list(results["arms"]) == list(ARMS)
        full = results["arms"]["full"]
        assert set(full["spread"]) == {"logic", "random"}
        assert full["shipped"] == "full-seed0"

    def test_reports_curriculum_metrics_for_every_run(self, run):
        _, results = run
        assert set(results["curriculum"]) == {f"{name}-seed0" for name in ARMS}

    def test_plays_full_against_no_self_play_head_to_head(self, run):
        _, results = run
        head_to_head = results["head_to_head"]
        assert head_to_head["metadata"]["model"] == "full-seed0"
        assert head_to_head["overall"]["opponent"] == "no-self-play-seed0"
        assert head_to_head["overall"]["episodes"] == 2 * len(REPORT_SEEDS)

    def test_results_are_read_by_the_statistics_tooling(self, run):
        out_dir, _ = run
        rows = [row for row in load_experiment_results(out_dir.parent) if row["experiment"] == "ablations"]
        assert len(rows) == 4 * 2 + 1  # 4 runs x 2 opponents + the head-to-head
        assert {row["arm"] for row in rows} == set(ARMS)


def test_a_finished_run_is_not_trained_again(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(arms, "train", lambda **kwargs: calls.append(kwargs))
    run_dir = tmp_path / "full-seed0"
    arms.train_arm("full", seed=0, timesteps=100, run_dir=run_dir)
    (run_dir / "ppo_cr_final.zip").write_bytes(b"")
    arms.train_arm("full", seed=0, timesteps=100, run_dir=run_dir)
    assert len(calls) == 1
