"""The final-configuration and Ablation run plans, end to end with tiny step and episode counts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cr_rl.experiments import arms
from cr_rl.experiments.ablations import run_ablations
from cr_rl.experiments.arms import ARMS
from cr_rl.experiments.final_configuration import run_final_configuration, ship_run
from cr_rl.stats.reporting import load_experiment_results
from cr_rl.training.report import ENV_VERSION, selected_checkpoint

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
ABLATION_SEEDS = (0, 1)  # two runs per arm, so the head-to-head matrix has more than one pair


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.slow
class TestFinalConfigurationPlan:
    @pytest.fixture(scope="class")
    def run(self, tmp_path_factory):
        out_dir = tmp_path_factory.mktemp("final")
        ship_to = out_dir / "web" / "ppo_cr_best.zip"
        run_final_configuration(timesteps=SMOKE_STEPS, seeds=(0,), out_dir=out_dir, ship_to=ship_to, **SMOKE)
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

    def test_ships_the_shipped_run_only_if_it_passes_the_ship_gate(self, run):
        out_dir, results = run
        assert (out_dir / "web" / "ppo_cr_best.zip").is_file() == results["ship_gate_passed"]

    def test_shipping_a_run_copies_its_selected_checkpoint(self, run, tmp_path):
        out_dir, results = run
        run_dir = out_dir / results["shipped"]
        target = tmp_path / "models" / "ppo_cr_best.zip"
        ship_run(run_dir, target)
        assert target.read_bytes() == selected_checkpoint(run_dir)[1].read_bytes()

    def test_plays_each_selected_checkpoint_against_its_final_model(self, run):
        _, results = run
        pairs = results["selected_vs_final"]
        assert [(pair["metadata"]["model"], pair["overall"]["opponent"]) for pair in pairs] == [
            ("full-seed0 selected", "full-seed0 final")
        ]
        assert pairs[0]["overall"]["episodes"] == 2 * len(REPORT_SEEDS)

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


@pytest.mark.slow
class TestAblationPlan:
    @pytest.fixture(scope="class")
    def run(self, tmp_path_factory):
        out_dir = tmp_path_factory.mktemp("ablations")
        run_ablations(timesteps=SMOKE_STEPS, seeds=ABLATION_SEEDS, out_dir=out_dir, **SMOKE)
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
        assert [run["name"] for run in full["runs"]] == ["full-seed0", "full-seed1"]
        assert full["shipped"] in {"full-seed0", "full-seed1"}

    def test_reports_curriculum_metrics_for_every_run(self, run):
        _, results = run
        assert set(results["curriculum"]) == {f"{name}-seed{seed}" for name in ARMS for seed in ABLATION_SEEDS}

    def test_plays_full_against_no_self_play_head_to_head(self, run):
        _, results = run
        head_to_head = results["head_to_head"]
        assert head_to_head["metadata"]["model"] == results["arms"]["full"]["shipped"]
        assert head_to_head["overall"]["opponent"] == results["arms"]["no-self-play"]["shipped"]
        assert head_to_head["overall"]["episodes"] == 2 * len(REPORT_SEEDS)

    def test_plays_every_full_run_against_every_no_self_play_run(self, run):
        _, results = run
        pairs = [(pair["metadata"]["model"], pair["overall"]["opponent"]) for pair in results["head_to_head_matrix"]]
        assert pairs == [("full-seed0", "no-self-play-seed0"), ("full-seed0", "no-self-play-seed1"),
                         ("full-seed1", "no-self-play-seed0"), ("full-seed1", "no-self-play-seed1")]
        assert all(pair["overall"]["episodes"] == 2 * len(REPORT_SEEDS) for pair in results["head_to_head_matrix"])

    def test_results_are_read_by_the_statistics_tooling(self, run):
        out_dir, _ = run
        rows = [row for row in load_experiment_results(out_dir.parent) if row["experiment"] == "ablations"]
        assert len(rows) == 4 * 2 * 2 + 1  # 4 arms x 2 seeds x 2 opponents + the shipped head-to-head
        assert {row["arm"] for row in rows} == set(ARMS)


def test_a_finished_run_is_not_trained_again(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(arms, "train", lambda **kwargs: calls.append(kwargs))
    run_dir = tmp_path / "full-seed0"
    arms.train_arm("full", seed=0, timesteps=100, run_dir=run_dir)
    (run_dir / "ppo_cr_final.zip").write_bytes(b"")
    arms.train_arm("full", seed=0, timesteps=100, run_dir=run_dir)
    assert len(calls) == 1
