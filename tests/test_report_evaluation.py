"""Report-seed evaluation and the Ship gate (ADR-0002 eval protocol)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sb3_contrib import MaskablePPO

from cr_rl.env.gym_env import ClashRoyaleEnv, derive_episode_seed
from cr_rl.game.board import NUM_ACTIONS, OBS_DIM
from cr_rl.stats.reporting import load_experiment_results
from cr_rl.training.matches import MatchResult, wilson_interval
from cr_rl.training.report import (
    ENV_VERSION,
    REPORT_SEEDS,
    OpponentReport,
    ReportEvaluation,
    ReportMetadata,
    RunReport,
    evaluate_configuration,
    evaluate_report,
    report_matches,
    save_configuration_report,
    ship_gate_passes,
    summarize_configuration,
)
from cr_rl.training.seeds import TRAINING_SEED_LIMIT
from cr_rl.training.selection import SELECTION_SEEDS, CheckpointResult, save_selection
from cr_rl.training.self_play import admission_seeds

TINY_SEEDS = range(2_000_000, 2_000_002)


@pytest.fixture(scope="module")
def policy() -> MaskablePPO:
    return MaskablePPO("MlpPolicy", ClashRoyaleEnv(seed=0), n_steps=64, batch_size=64, seed=0, device="cpu")


def _opponent_report(wins: int, opponent: str = "logic", episodes: int = 200) -> OpponentReport:
    results = [MatchResult(i, 0 if i < wins else 1, 0.0, 10) for i in range(episodes)]
    return report_matches(opponent, results)


def _run(name: str, selection_wins: int, logic_wins: int) -> RunReport:
    metadata = ReportMetadata("stub", REPORT_SEEDS.start, len(REPORT_SEEDS), True, ENV_VERSION, OBS_DIM, NUM_ACTIONS)
    evaluation = ReportEvaluation(metadata, (_opponent_report(logic_wins), _opponent_report(180, "random")))
    return RunReport(name, CheckpointResult(100, selection_wins, 100, f"{name}.zip"), evaluation)


class TestReportSeeds:
    def test_are_200_fixed_seeds_from_two_million(self):
        assert list(REPORT_SEEDS) == [2_000_000 + i for i in range(200)]

    def test_are_disjoint_from_selection_seeds(self):
        assert not set(REPORT_SEEDS) & set(SELECTION_SEEDS)

    def test_the_training_seed_limit_lies_below_both_ranges(self):
        assert TRAINING_SEED_LIMIT <= min(SELECTION_SEEDS.start, REPORT_SEEDS.start)

    def test_the_largest_possible_training_episode_seed_stays_below_the_limit(self, monkeypatch):
        class WorstCase:  # a SeedSequence whose generated state is the largest 32-bit word
            def __init__(self, entropy):
                pass

            def generate_state(self, n_words):
                return np.full(n_words, 2**32 - 1, dtype=np.uint32)

        monkeypatch.setattr(np.random, "SeedSequence", WorstCase)
        assert derive_episode_seed(0, 0, 0) < TRAINING_SEED_LIMIT

    def test_the_largest_possible_admission_seed_stays_below_the_limit(self, monkeypatch):
        class WorstCase:  # a generator that always draws the largest values the population allows
            def choice(self, population, size, replace):
                return np.arange(population - size, population)

        monkeypatch.setattr(np.random, "default_rng", lambda seed_sequence: WorstCase())
        assert max(admission_seeds(0, 100_000)) < TRAINING_SEED_LIMIT


class TestOpponentReport:
    def test_counts_win_rate_interval_return_and_length(self):
        results = [
            MatchResult(0, 0, 1.0, 100),
            MatchResult(1, 0, 0.5, 200),
            MatchResult(2, 1, -1.0, 300),
            MatchResult(3, None, 0.0, 400),
        ]
        report = report_matches("logic", results)
        assert (report.opponent, report.episodes) == ("logic", 4)
        assert (report.wins, report.losses, report.draws) == (2, 1, 1)
        assert report.win_rate == 0.5
        assert (report.ci_low, report.ci_high) == wilson_interval(2, 4)
        assert report.mean_return == pytest.approx(0.125)
        assert report.mean_length == 250


class TestShipGate:
    def test_passes_when_the_wilson_lower_bound_is_above_half(self):
        assert ship_gate_passes(_opponent_report(wins=130))  # 65%: lower bound about 58%

    def test_fails_when_the_point_estimate_is_above_half_but_the_lower_bound_is_not(self):
        report = _opponent_report(wins=110)  # 55%: lower bound about 48%
        assert report.win_rate > 0.5 and report.ci_low < 0.5
        assert not ship_gate_passes(report)


class TestEvaluateReport:
    @pytest.fixture(scope="class")
    def evaluation(self, policy):
        return evaluate_report(policy, model="tiny.zip", seeds=TINY_SEEDS, config={"seed": 7})

    def test_reports_both_opponents_on_the_same_seeds(self, evaluation):
        assert [report.opponent for report in evaluation.opponents] == ["logic", "random"]
        assert all(report.episodes == 2 for report in evaluation.opponents)
        for report in evaluation.opponents:
            assert report.wins + report.losses + report.draws == 2
            assert 0.0 <= report.ci_low <= report.win_rate <= report.ci_high <= 1.0

    def test_metadata_makes_the_numbers_reproducible(self, evaluation):
        metadata = evaluation.metadata
        assert metadata.model == "tiny.zip"
        assert (metadata.seed_start, metadata.episodes) == (2_000_000, 2)
        assert metadata.deterministic is True
        assert metadata.env_version == ENV_VERSION == "env-v1.1"
        assert (metadata.obs_dim, metadata.num_actions) == (OBS_DIM, NUM_ACTIONS)
        assert metadata.config == {"seed": 7}

    def test_is_deterministic(self, policy, evaluation):
        assert evaluate_report(policy, model="tiny.zip", seeds=TINY_SEEDS, config={"seed": 7}) == evaluation

    def test_the_verdict_is_taken_against_logic_agent(self, evaluation):
        assert evaluation.passes_ship_gate == ship_gate_passes(evaluation.against("logic"))


class TestConfigurationReport:
    def test_the_shipped_run_has_the_best_selection_win_rate_not_the_best_report_win_rate(self):
        runs = [_run("a", selection_wins=60, logic_wins=150), _run("b", selection_wins=75, logic_wins=100)]
        report = summarize_configuration(runs)
        assert report.shipped == "b"
        assert not report.ship_gate_passed  # b's 50% on the Report seeds does not clear the gate

    def test_the_verdict_is_computed_on_the_shipped_runs_report_results(self):
        runs = [_run("a", selection_wins=60, logic_wins=100), _run("b", selection_wins=75, logic_wins=140)]
        assert summarize_configuration(runs).ship_gate_passed

    def test_reports_per_seed_win_rates_and_their_spread(self):
        runs = [_run(name, 50, wins) for name, wins in (("a", 100), ("b", 120), ("c", 140))]
        report = summarize_configuration(runs)
        logic = report.spread["logic"]
        assert [run.evaluation.against("logic").win_rate for run in report.runs] == [0.5, 0.6, 0.7]
        assert (logic.min, logic.max) == (0.5, 0.7)
        assert logic.mean == pytest.approx(0.6)
        assert logic.std == pytest.approx(0.1)  # sample standard deviation

    def test_a_single_run_has_no_spread(self):
        assert summarize_configuration([_run("a", 50, 100)]).spread["logic"].std == 0.0


class TestRunDirectories:
    @pytest.fixture(scope="class")
    def run_dirs(self, policy, tmp_path_factory):
        dirs = []
        for name, wins in (("seed0", 30), ("seed1", 70)):
            run_dir = tmp_path_factory.mktemp(name)
            (run_dir / "checkpoints").mkdir()
            policy.save(str(run_dir / "checkpoints" / "ppo_cr_100_steps.zip"))
            save_selection([CheckpointResult(100, wins, 100, "ppo_cr_100_steps.zip")], run_dir / "selection.json")
            dirs.append(run_dir)
        return dirs

    def test_evaluates_each_runs_selected_checkpoint_and_picks_the_ship_run(self, run_dirs):
        report = evaluate_configuration(run_dirs, seeds=TINY_SEEDS)
        assert [run.name for run in report.runs] == [run_dir.name for run_dir in run_dirs]
        assert report.shipped == run_dirs[1].name
        assert all(run.evaluation.metadata.episodes == 2 for run in report.runs)

    def test_results_are_saved_where_the_statistics_tooling_reads_them(self, run_dirs, tmp_path):
        report = evaluate_configuration(run_dirs, seeds=TINY_SEEDS)
        save_configuration_report(report, tmp_path / "report" / "results.json")
        saved = json.loads((tmp_path / "report" / "results.json").read_text(encoding="utf-8"))
        assert saved["shipped"] == report.shipped
        rows = load_experiment_results(tmp_path)
        assert len(rows) == 4  # 2 runs x 2 opponents
        assert {row["opponent"] for row in rows} == {"logic", "random"}
