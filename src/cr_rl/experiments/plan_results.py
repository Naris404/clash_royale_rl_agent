"""Read-only summary of the run plans' results, from the files each plan already wrote.

Reads a plan's `results.json` and each run's `manifest.json` and `selection.json`; no model is
loaded and no match is played.

Usage:
  cr-rl-results                                      # every plan under experiment_runs/
  cr-rl-results experiment_runs/ablations --json
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from cr_rl.paths import EXPERIMENT_RUNS_DIR
from cr_rl.training.phases import FULL, TrainingPhase, phase_steps
from cr_rl.training.report import (
    MANIFEST_FILENAME,
    REPORT_FILENAME,
    SHIP_GATE_OPPONENT,
    OpponentReport,
    Spread,
    Target,
)
from cr_rl.training.selection import SELECTION_FILENAME, SELECTION_INTERVAL, CheckpointResult, load_selection

SELF_PLAY_WARNING = "selected before self-play: the {model} model has no self-play training"
NO_SELECTION_WARNING = f"no {SELECTION_FILENAME}: phase, selection curve and self-play check unavailable"


@dataclass(frozen=True)
class SelectionRound:
    step: int
    wins: int
    phase: str  # the Training phase that trained the model evaluated at this step


@dataclass(frozen=True)
class RunSummary:
    """One run: its selected checkpoint, that checkpoint's Report results and its selection curve."""

    name: str
    selection: CheckpointResult
    selected_phase: str | None  # None without the run's selection.json
    report: tuple[OpponentReport, ...]
    shipped: bool
    selected_before_self_play: bool  # the layout has a self-play phase and the selection kept its start
    curve: tuple[SelectionRound, ...]


@dataclass(frozen=True)
class ArmSummary:
    """An arm's runs, the spread of their Report win rate vs LogicAgent and their mean curriculum metrics."""

    arm: str
    runs: tuple[RunSummary, ...]
    logic_spread: Spread
    jumpstart: float | None  # None if a run has no value
    steps_to_half: float | None


@dataclass(frozen=True)
class Matchup:
    """A head-to-head from `model`'s side; `result.opponent` names the other model."""

    model: str
    result: OpponentReport


@dataclass(frozen=True)
class HeadToHeadGrid:
    """The head-to-head matrix: the win rate of each row's model against each column's model."""

    rows: tuple[str, ...]
    columns: tuple[str, ...]
    win_rates: tuple[tuple[float, ...], ...]
    row_means: tuple[float, ...]
    column_means: tuple[float, ...]


@dataclass(frozen=True)
class ExploiterSummary:
    """How an Exploiter's training ended: `stop_reason` is "win-rate" or "step-limit"."""

    run: str
    stop_reason: str
    win_rate: float | None
    steps: int


@dataclass(frozen=True)
class PlanSummary:
    name: str
    timesteps: int
    arms: tuple[ArmSummary, ...]
    shipped_pair: Matchup | None  # the ablations' head-to-head between the shipped runs
    matrix: HeadToHeadGrid | None
    selected_vs_final: tuple[Matchup, ...]  # the final configuration's selected vs final model
    target: Target | None = None  # the v3 plan's ship decision
    exploiters: tuple[ExploiterSummary, ...] = ()


def phase_starts(layout: str, timesteps: int, steps: Sequence[int]) -> list[tuple[TrainingPhase, int]]:
    """Each Training phase of a run with the step it starts after, given the run's selection steps.

    A phase that plays Past selves starts with a selection round (`train()` evaluates the model to
    seed the pool), taken at the rollout end at or after the phase's planned start. The next regular
    round comes a full selection interval later, so the start round is the last one before the planned
    start plus one interval, the final model's round aside. Other phases start at their planned step.
    """
    starts, planned = [], 0
    for phase, length in phase_steps(layout, timesteps):
        start = planned
        if phase.uses_pool:
            start = max((step for step in steps[:-1] if step < planned + SELECTION_INTERVAL), default=planned)
        starts.append((phase, start))
        planned += length
    return starts


def phase_at(step: int, starts: Sequence[tuple[TrainingPhase, int]]) -> str:
    """The phase that trained the model at `step`; a step at a phase's start belongs to the phase before."""
    return [phase.name for phase, start in starts if start < step or start == 0][-1]


def summarize_run(record: dict[str, Any], run_dir: Path, *, shipped: bool) -> RunSummary:
    selection = CheckpointResult(**record["selection"])
    report = tuple(OpponentReport(**opponent) for opponent in record["evaluation"]["opponents"])
    selection_path = run_dir / SELECTION_FILENAME
    selected_phase, before_self_play, curve = None, False, ()
    if selection_path.is_file():
        manifest = json.loads((run_dir / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        rounds = load_selection(selection_path)
        starts = phase_starts(manifest["layout"], manifest["timesteps"], [result.step for result in rounds])
        selected_phase = phase_at(selection.step, starts)
        self_play_start = next((start for phase, start in starts if phase.uses_pool), None)
        before_self_play = self_play_start is not None and selection.step <= self_play_start
        curve = tuple(SelectionRound(result.step, result.wins, phase_at(result.step, starts)) for result in rounds)
    return RunSummary(record["name"], selection, selected_phase, report, shipped, before_self_play, curve)


def mean_or_none(values: Sequence[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.fmean(present) if len(present) == len(values) else None


def summarize_arm(arm: str, report: dict[str, Any], curriculum: dict[str, Any] | None, plan_dir: Path) -> ArmSummary:
    runs = tuple(
        summarize_run(record, plan_dir / record["name"], shipped=record["name"] == report["shipped"])
        for record in report["runs"]
    )
    jumpstart = steps_to_half = None  # the v3 plan records no curriculum metrics
    if curriculum is not None:
        metrics = [curriculum[run.name] for run in runs]
        jumpstart = mean_or_none([run_metrics["jumpstart"] for run_metrics in metrics])
        steps_to_half = mean_or_none([run_metrics["steps_to_half"] for run_metrics in metrics])
    return ArmSummary(
        arm,
        runs,
        logic_spread=Spread(**report["spread"][SHIP_GATE_OPPONENT]),
        jumpstart=jumpstart,
        steps_to_half=steps_to_half,
    )


def matchup(record: dict[str, Any]) -> Matchup:
    return Matchup(record["metadata"]["model"], OpponentReport(**record["overall"]))


def head_to_head_grid(matchups: Sequence[Matchup]) -> HeadToHeadGrid:
    rows = tuple(dict.fromkeys(item.model for item in matchups))
    columns = tuple(dict.fromkeys(item.result.opponent for item in matchups))
    rates = {(item.model, item.result.opponent): item.result.win_rate for item in matchups}
    win_rates = tuple(tuple(rates[row, column] for column in columns) for row in rows)
    return HeadToHeadGrid(
        rows,
        columns,
        win_rates,
        row_means=tuple(statistics.fmean(row) for row in win_rates),
        column_means=tuple(statistics.fmean(column) for column in zip(*win_rates)),
    )


def summarize_plan(plan_dir: Path) -> PlanSummary:
    """The summary of one plan directory; the final configuration is a plan with one arm."""
    results = json.loads((plan_dir / REPORT_FILENAME).read_text(encoding="utf-8"))
    arms = results.get("arms") or {FULL: results}
    matrix = [matchup(record) for record in results.get("head_to_head_matrix", [])]
    return PlanSummary(
        name=plan_dir.name,
        timesteps=results["timesteps"],
        arms=tuple(summarize_arm(arm, report, results.get("curriculum"), plan_dir) for arm, report in arms.items()),
        shipped_pair=matchup(results["head_to_head"]) if "head_to_head" in results else None,
        matrix=head_to_head_grid(matrix) if matrix else None,
        selected_vs_final=tuple(matchup(record) for record in results.get("selected_vs_final", [])),
        target=Target(**results["target"]) if "target" in results else None,
        exploiters=tuple(
            ExploiterSummary(record["run"], record["stop_reason"], record["win_rate"], record["steps"])
            for record in results.get("exploiters", [])
        ),
    )


def percent(value: float) -> str:
    return f"{value:.1%}"


def thousands(step: float) -> str:
    """A step count in thousands: 601524 -> 601.5k, 600048 -> 600k."""
    return f"{step / 1000:.1f}".removesuffix(".0") + "k"


def with_interval(report: OpponentReport) -> str:
    return f"{percent(report.win_rate)} [{percent(report.ci_low)}, {percent(report.ci_high)}]"


def describe(item: Matchup) -> str:
    result = item.result
    record = f"W/L/D {result.wins}/{result.losses}/{result.draws}"
    return f"{item.model} vs {result.opponent}: {with_interval(result)}, {record}"


def table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """Left-aligned plain-text columns."""
    widths = [max(len(cell) for cell in column) for column in zip(headers, *rows)]
    lines = (headers, *rows)
    return ["  ".join(cell.ljust(width) for cell, width in zip(line, widths)).rstrip() for line in lines]


def run_row(arm: str, run: RunSummary) -> list[str]:
    selection = run.selection
    row = [arm, run.name, thousands(selection.step), run.selected_phase or "-", f"{selection.wins}/{selection.episodes}"]
    for report in run.report:
        row += [with_interval(report), f"{report.mean_length:.1f}"]
    return row + ["yes" if run.shipped else ""]


def arm_row(arm: ArmSummary) -> list[str]:
    spread = arm.logic_spread
    return [
        arm.arm,
        f"{percent(spread.mean)} +/- {percent(spread.std)}",
        "-" if arm.jumpstart is None else percent(arm.jumpstart),
        "-" if arm.steps_to_half is None else thousands(arm.steps_to_half),
    ]


def grid_lines(grid: HeadToHeadGrid) -> list[str]:
    rows = [
        [row, *map(percent, rates), percent(mean)]
        for row, rates, mean in zip(grid.rows, grid.win_rates, grid.row_means)
    ]
    rows.append(["mean", *map(percent, grid.column_means), ""])
    return table(["", *grid.columns, "mean"], rows)


def curve_line(run: RunSummary, width: int) -> str:
    """`step:wins` per round, a `|` and the phase name where a phase begins, `*` on the selected round."""
    parts, phase = [], None
    for point in run.curve:
        if point.phase != phase:
            parts.append(point.phase if phase is None else f"| {point.phase}")
            phase = point.phase
        parts.append(f"{thousands(point.step)}:{point.wins}" + ("*" if point.step == run.selection.step else ""))
    return f"{run.name.ljust(width)}  " + (" ".join(parts) or "-")


def target_line(target: Target) -> str:
    verdict = "MET" if target.met else "NOT met"
    goal = f"{percent(target.logic_threshold)} vs logic and a head-to-head win over the v2 Shipped model"
    result = f"{percent(target.logic_win_rate)} vs logic, {percent(target.head_to_head_win_rate)} head-to-head"
    return f"v3 target ({goal}): {verdict}. {target.run}: {result}"


def run_warnings(run: RunSummary) -> list[str]:
    if not run.curve:
        return [f"{run.name}: {NO_SELECTION_WARNING}"]
    if run.selected_before_self_play:
        return [f"{run.name}: " + SELF_PLAY_WARNING.format(model="shipped" if run.shipped else "selected")]
    return []


def format_plan(plan: PlanSummary) -> str:
    """The plan's summary as plain-text tables."""
    runs = [run for arm in plan.arms for run in arm.runs]
    size = f"{len(plan.arms)} arm(s) x {len(runs) // len(plan.arms)} runs, {thousands(plan.timesteps)} steps"
    lines = [f"=== {plan.name}: {size} ===", ""]

    headers = ["arm", "run", "selected", "phase", "selection wins"]
    for report in runs[0].report:
        headers += [f"vs {report.opponent} [95% CI]", "length"]
    rows = [run_row(arm.arm, run) for arm in plan.arms for run in arm.runs]
    lines += ["Runs (Report seeds)", *table(headers + ["shipped"], rows), ""]

    headers = ["arm", f"vs {SHIP_GATE_OPPONENT} mean +/- sd", "jumpstart", "steps to 50%"]
    lines += ["Arms (mean over runs)", *table(headers, [arm_row(arm) for arm in plan.arms]), ""]

    if plan.shipped_pair:
        lines += ["Head-to-head (shipped runs)", describe(plan.shipped_pair), ""]
    if plan.matrix:
        lines += ["Head-to-head matrix (win rate of the row against the column)", *grid_lines(plan.matrix), ""]
    if plan.selected_vs_final:
        lines += ["Selected vs final model", *map(describe, plan.selected_vs_final), ""]

    if plan.target:
        lines += [target_line(plan.target), ""]
    if plan.exploiters:
        rows = [
            [item.run, item.stop_reason, "-" if item.win_rate is None else percent(item.win_rate), thousands(item.steps)]
            for item in plan.exploiters
        ]
        headers = ["run", "stopped", "win rate", "steps"]
        lines += ["Exploiters (rolling win rate against the target when training stopped)", *table(headers, rows), ""]

    width = max(len(run.name) for run in runs)
    lines += ["Selection curve (step:wins, * = selected)", *(curve_line(run, width) for run in runs)]
    warnings = [warning for run in runs for warning in run_warnings(run)]
    if warnings:
        lines += ["", "Warnings", *warnings]
    return "\n".join(lines)


def is_plan_dir(path: Path) -> bool:
    """Other experiments (the Arrows experiment) write a results file too; only a run plan's records timesteps."""
    results = path / REPORT_FILENAME
    return results.is_file() and "timesteps" in json.loads(results.read_text(encoding="utf-8"))


def plan_dirs(path: Path) -> list[Path]:
    """`path` itself if it is a plan directory, otherwise its subdirectories that hold a plan's results."""
    if is_plan_dir(path):
        return [path]
    return sorted(child for child in path.iterdir() if is_plan_dir(child))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summary of the run plans' results, read from the files they wrote")
    parser.add_argument(
        "path", nargs="?", type=Path, default=EXPERIMENT_RUNS_DIR, help="A plan directory or a directory of plans"
    )
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON")
    args = parser.parse_args(argv)

    plans = [summarize_plan(plan_dir) for plan_dir in plan_dirs(args.path)]
    if args.json:
        print(json.dumps([asdict(plan) for plan in plans], indent=2))
    else:
        print("\n\n".join(format_plan(plan) for plan in plans))


if __name__ == "__main__":
    main()
