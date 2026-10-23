"""Report-seed evaluation: the reportable numbers and the Ship gate verdict (ADR-0002).

A model plays the Report seeds against LogicAgent and RandomAgent; a configuration of several
training runs ships the run with the best Selection win rate and is judged on that run's Report
results only. Report seeds are never used in training or selection.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from sb3_contrib import MaskablePPO

from cr_rl.game.board import NUM_ACTIONS, OBS_DIM
from cr_rl.jsonio import write_json
from cr_rl.paths import EXPERIMENT_RUNS_DIR
from cr_rl.training.matches import MatchResult, play_matches, summarize_matches, wilson_interval
from cr_rl.training.seeds import REPORT_SEEDS
from cr_rl.training.selection import (
    SELECTION_FILENAME,
    CheckpointResult,
    load_selection,
    select_checkpoint,
)

REPORT_OPPONENTS = ("logic", "random")
ENV_VERSION = "env-v1.1"
SHIP_GATE_OPPONENT = "logic"
SHIP_GATE_THRESHOLD = 0.5  # the Wilson lower bound vs LogicAgent must be above this
REPORT_FILENAME = "results.json"
MANIFEST_FILENAME = "manifest.json"  # what trained a run; becomes the config of its evaluation
DEFAULT_REPORT_PATH = EXPERIMENT_RUNS_DIR / "report_evaluation" / REPORT_FILENAME


@dataclass(frozen=True)
class OpponentReport:
    """A model's results against one opponent on a seed list, from player 0's side."""

    opponent: str
    episodes: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    ci_low: float  # 95% Wilson interval of the win rate
    ci_high: float
    mean_return: float
    mean_length: float  # decision steps


@dataclass(frozen=True)
class ReportMetadata:
    """Everything needed to regenerate the numbers: the model, seeds, layout and environment."""

    model: str
    seed_start: int
    episodes: int  # the seeds are seed_start .. seed_start + episodes - 1
    deterministic: bool
    env_version: str
    obs_dim: int
    num_actions: int
    config: dict[str, Any] | None = None  # training configuration of the model, if known


@dataclass(frozen=True)
class ReportEvaluation:
    """One model on the Report seeds against each opponent."""

    metadata: ReportMetadata
    opponents: tuple[OpponentReport, ...]

    def against(self, opponent: str) -> OpponentReport:
        return next(report for report in self.opponents if report.opponent == opponent)

    @property
    def passes_ship_gate(self) -> bool:
        return ship_gate_passes(self.against(SHIP_GATE_OPPONENT))


@dataclass(frozen=True)
class RunReport:
    """One training run: its selected checkpoint and that checkpoint's Report evaluation."""

    name: str
    selection: CheckpointResult
    evaluation: ReportEvaluation


@dataclass(frozen=True)
class Spread:
    """Mean and spread of one opponent's win rate across the runs of a configuration."""

    mean: float
    std: float  # sample standard deviation, 0 for a single run
    min: float
    max: float


@dataclass(frozen=True)
class ConfigurationReport:
    """All runs of a configuration, the shipped run and the Ship gate verdict for it."""

    runs: tuple[RunReport, ...]
    shipped: str  # name of the run with the best Selection win rate
    ship_gate_passed: bool
    spread: dict[str, Spread]  # opponent -> spread of the runs' win rates


def report_matches(opponent: str, results: Sequence[MatchResult]) -> OpponentReport:
    stats = summarize_matches(results)
    wins = int(stats["wins"])
    ci_low, ci_high = wilson_interval(wins, int(stats["episodes"]))
    return OpponentReport(
        opponent=opponent,
        episodes=int(stats["episodes"]),
        wins=wins,
        losses=int(stats["losses"]),
        draws=int(stats["draws"]),
        win_rate=stats["win_rate"],
        ci_low=ci_low,
        ci_high=ci_high,
        mean_return=stats["mean_reward"],
        mean_length=stats["mean_length"],
    )


def report_metadata(model: str, seeds: range, config: dict[str, Any] | None = None) -> ReportMetadata:
    """Metadata of a deterministic evaluation of `model` on `seeds` in the current environment."""
    return ReportMetadata(
        model=model,
        seed_start=seeds.start,
        episodes=len(seeds),
        deterministic=True,
        env_version=ENV_VERSION,
        obs_dim=OBS_DIM,
        num_actions=NUM_ACTIONS,
        config=config,
    )


def ship_gate_passes(report: OpponentReport) -> bool:
    """The Ship gate: the lower bound of the 95% Wilson interval is above 50%."""
    return report.ci_low > SHIP_GATE_THRESHOLD


def evaluate_report(
    player: Any,
    *,
    model: str = "",
    config: dict[str, Any] | None = None,
    seeds: range = REPORT_SEEDS,
    opponents: Sequence[str] = REPORT_OPPONENTS,
) -> ReportEvaluation:
    """Play `player` deterministically on the same seeds against each opponent."""
    metadata = report_metadata(model, seeds, config)
    reports = tuple(report_matches(name, play_matches(player, name, seeds)) for name in opponents)
    return ReportEvaluation(metadata, reports)


def summarize_configuration(runs: Sequence[RunReport]) -> ConfigurationReport:
    """Ship the run with the best Selection win rate (the first on a tie), never by Report results."""
    shipped = max(runs, key=lambda run: run.selection.win_rate)
    spread = {}
    for opponent in (report.opponent for report in runs[0].evaluation.opponents):
        rates = [run.evaluation.against(opponent).win_rate for run in runs]
        spread[opponent] = Spread(
            mean=statistics.fmean(rates),
            std=statistics.stdev(rates) if len(rates) > 1 else 0.0,
            min=min(rates),
            max=max(rates),
        )
    return ConfigurationReport(tuple(runs), shipped.name, shipped.evaluation.passes_ship_gate, spread)


def selected_checkpoint(run_dir: Path) -> tuple[CheckpointResult, Path]:
    """The checkpoint a run selected on its Selection seeds, and its file."""
    selected = select_checkpoint(load_selection(run_dir / SELECTION_FILENAME))
    return selected, run_dir / "checkpoints" / selected.path


def evaluate_run(run_dir: Path, *, seeds: range = REPORT_SEEDS) -> RunReport:
    """Evaluate the checkpoint a run selected on its Selection seeds."""
    selected, checkpoint = selected_checkpoint(run_dir)
    manifest = run_dir / MANIFEST_FILENAME
    config = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else None
    evaluation = evaluate_report(
        MaskablePPO.load(str(checkpoint)), model=str(checkpoint), config=config, seeds=seeds
    )
    return RunReport(run_dir.name, selected, evaluation)


def evaluate_configuration(run_dirs: Sequence[Path], *, seeds: range = REPORT_SEEDS) -> ConfigurationReport:
    return summarize_configuration([evaluate_run(run_dir, seeds=seeds) for run_dir in run_dirs])


def configuration_rows(report: ConfigurationReport) -> list[dict[str, Any]]:
    """One row per run and opponent, the shape `load_experiment_results` reads."""
    return [
        {**asdict(opponent), "config": f"{run.name} vs {opponent.opponent}", "run": run.name}
        for run in report.runs
        for opponent in run.evaluation.opponents
    ]


def save_configuration_report(
    report: ConfigurationReport,
    path: Path,
    *,
    experiment: str = "report_evaluation",
    extra: dict[str, Any] | None = None,
) -> None:
    """Write the report, with `extra` entries beside it; `results` holds `configuration_rows`."""
    payload = {"experiment": experiment, **asdict(report), **(extra or {}), "results": configuration_rows(report)}
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the selected checkpoint of each run on the Report seeds")
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories (one per training seed)")
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()

    report = evaluate_configuration(args.runs)
    save_configuration_report(report, args.out)
    for run in report.runs:
        for opponent in run.evaluation.opponents:
            print(
                f"{run.name} vs {opponent.opponent}: win rate {opponent.win_rate:.1%} "
                f"(95% CI {opponent.ci_low:.1%} to {opponent.ci_high:.1%}), "
                f"W/L/D {opponent.wins}/{opponent.losses}/{opponent.draws}"
            )
    print(f"Shipped run: {report.shipped} | Ship gate: {'PASS' if report.ship_gate_passed else 'FAIL'}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
