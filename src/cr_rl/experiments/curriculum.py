"""Curriculum metrics and the budget rule, read from a run's recorded selection results (ADR-0002)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from cr_rl.training.phases import phase_steps
from cr_rl.training.selection import CheckpointResult

HALF_WIN_RATE = 0.5  # target of the time-to-50% metric
BUDGET_WINDOW = 0.2  # the budget rule looks at the last 20% of the run's steps
BUDGET_RISE = 0.10  # a rise above this means the run is still learning
EXTENDED_TIMESTEPS = 3_000_000  # budget of every final run when the rule fires


@dataclass(frozen=True)
class CurriculumMetrics:
    """How fast a run learned: `None` where the run never got there."""

    jumpstart: float | None  # selection win rate at the first round after the warm-up
    steps_to_half: int | None  # first round with a 50% selection win rate (warm-up steps count)
    report_win_rate: float  # final win rate vs LogicAgent on the Report seeds


@dataclass(frozen=True)
class BudgetCheck:
    """The budget rule: did the selection win rate rise by more than 10 points in the last 20%?"""

    start_win_rate: float | None  # last round at or before the window's start
    end_win_rate: float
    rise: float
    extend: bool  # True: all final runs should move to 3M steps


def warm_up_steps(timesteps: int) -> int:
    """Length of the full layout's warm-up; the no-curriculum arm is read at the same step."""
    return phase_steps("full", timesteps)[0][1]


def win_rate_after(results: Sequence[CheckpointResult], step: int) -> float | None:
    """Selection win rate of the first round at or after `step`."""
    return next((result.win_rate for result in results if result.step >= step), None)


def steps_to_win_rate(results: Sequence[CheckpointResult], target: float) -> int | None:
    """Step of the first round whose selection win rate reaches `target`."""
    return next((result.step for result in results if result.win_rate >= target), None)


def curriculum_metrics(
    results: Sequence[CheckpointResult], *, warm_up_steps: int, report_win_rate: float
) -> CurriculumMetrics:
    return CurriculumMetrics(
        jumpstart=win_rate_after(results, warm_up_steps),
        steps_to_half=steps_to_win_rate(results, HALF_WIN_RATE),
        report_win_rate=report_win_rate,
    )


def budget_rule(results: Sequence[CheckpointResult], *, timesteps: int) -> BudgetCheck:
    """Compare the final round with the last round at or before 80% of the run's steps."""
    window_start = timesteps * (1 - BUDGET_WINDOW)
    before = [result for result in results if result.step <= window_start]
    end = results[-1].win_rate
    if not before:
        return BudgetCheck(None, end, 0.0, False)
    start = before[-1].win_rate
    rise = end - start
    return BudgetCheck(start, end, rise, round(rise, 9) > BUDGET_RISE)  # rounding: 0.8 - 0.7 is not exactly 0.1
