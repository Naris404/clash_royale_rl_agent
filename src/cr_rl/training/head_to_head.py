"""Head-to-head: two selected models play each other on the Report seeds (ADR-0002).

This is how the thesis shows whether self-play helped, which a win rate vs LogicAgent alone
cannot show. Each seed is played twice, once with each model as player 0.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from cr_rl.training.matches import MatchResult, play_matches
from cr_rl.training.report import REPORT_SEEDS, OpponentReport, ReportMetadata, report_matches, report_metadata


@dataclass(frozen=True)
class HeadToHead:
    """Model A's results against model B; `opponent` of every report names model B."""

    metadata: ReportMetadata  # `model` is model A; `episodes` counts the seeds, each played twice
    overall: OpponentReport  # both sides together: 2 * len(seeds) matches
    as_player_0: OpponentReport
    as_player_1: OpponentReport


def evaluate_head_to_head(
    model_a: Any,
    model_b: Any,
    *,
    name_a: str = "a",
    name_b: str = "b",
    config: dict[str, Any] | None = None,
    seeds: range = REPORT_SEEDS,
) -> HeadToHead:
    """Play A against B deterministically on every seed, with A and then B as player 0."""
    as_player_0 = play_matches(model_a, model_b, seeds)
    as_player_1 = [_from_other_side(result) for result in play_matches(model_b, model_a, seeds)]
    return HeadToHead(
        report_metadata(name_a, seeds, config),
        report_matches(name_b, as_player_0 + as_player_1),
        report_matches(name_b, as_player_0),
        report_matches(name_b, as_player_1),
    )


def _from_other_side(result: MatchResult) -> MatchResult:
    """A match played from B's side as seen from A's: winner swapped, zero-sum return negated."""
    winner = None if result.winner is None else 1 - result.winner
    return replace(result, winner=winner, episode_return=-result.episode_return)
