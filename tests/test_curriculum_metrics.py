"""Curriculum metrics and the budget rule, computed from a run's recorded selection results."""

from __future__ import annotations

import pytest

from cr_rl.experiments.curriculum import (
    budget_rule,
    curriculum_metrics,
    steps_to_win_rate,
    win_rate_after,
)
from cr_rl.training.selection import CheckpointResult


def rounds(*points: tuple[int, int]) -> list[CheckpointResult]:
    """Selection rounds from (step, wins out of 100) pairs."""
    return [CheckpointResult(step, wins, 100, f"ppo_cr_{step}_steps.zip") for step, wins in points]


class TestJumpstart:
    def test_is_the_win_rate_of_the_first_round_after_the_warm_up(self):
        results = rounds((150, 10), (200, 30), (250, 50))
        assert win_rate_after(results, 200) == 0.30  # a round exactly at the warm-up's end counts

    def test_skips_rounds_during_the_warm_up(self):
        assert win_rate_after(rounds((150, 10), (260, 40)), 200) == 0.40

    def test_is_none_when_no_round_follows_the_warm_up(self):
        assert win_rate_after(rounds((50, 10), (100, 20)), 200) is None


class TestTimeToWinRate:
    def test_is_the_first_round_that_reaches_the_target(self):
        results = rounds((50, 20), (100, 49), (150, 50), (200, 70))
        assert steps_to_win_rate(results, 0.5) == 150

    def test_counts_warm_up_steps_because_the_step_is_the_runs_own(self):
        assert steps_to_win_rate(rounds((300, 55)), 0.5) == 300

    def test_is_none_when_the_target_is_never_reached(self):
        assert steps_to_win_rate(rounds((50, 20), (100, 49)), 0.5) is None


class TestCurriculumMetrics:
    def test_combines_the_selection_metrics_with_the_report_win_rate(self):
        results = rounds((50, 20), (200, 30), (250, 55))
        metrics = curriculum_metrics(results, warm_up_steps=200, report_win_rate=0.61)
        assert metrics.jumpstart == 0.30
        assert metrics.steps_to_half == 250
        assert metrics.report_win_rate == 0.61


class TestBudgetRule:
    # A 1000-step run: the last 20% starts at step 800.
    def test_extends_the_runs_when_the_rate_rose_by_more_than_10_points(self):
        check = budget_rule(rounds((400, 30), (800, 40), (1000, 51)), timesteps=1000)
        assert check.start_win_rate == 0.40 and check.end_win_rate == 0.51
        assert check.rise == pytest.approx(0.11)
        assert check.extend

    def test_keeps_the_budget_when_the_rise_is_exactly_10_points(self):
        assert not budget_rule(rounds((800, 40), (1000, 50)), timesteps=1000).extend
        assert not budget_rule(rounds((800, 70), (1000, 80)), timesteps=1000).extend  # 0.8 - 0.7 > 0.1 in floats

    def test_keeps_the_budget_when_the_rate_fell(self):
        assert not budget_rule(rounds((800, 60), (1000, 40)), timesteps=1000).extend

    def test_starts_from_the_last_round_at_or_before_the_window(self):
        check = budget_rule(rounds((700, 10), (790, 20), (900, 90), (1000, 35)), timesteps=1000)
        assert check.start_win_rate == 0.20  # the round at 790, not the later peak
        assert check.rise == pytest.approx(0.15)

    def test_has_no_rise_without_a_round_before_the_window(self):
        check = budget_rule(rounds((900, 20), (1000, 90)), timesteps=1000)
        assert check.rise == 0.0 and not check.extend
