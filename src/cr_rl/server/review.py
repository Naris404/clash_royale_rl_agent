"""Match review: picks the player's strongest and weakest moves."""

from __future__ import annotations

MAX_PICKS = 3


def pick_strongest_weakest(move_log: list[dict]) -> tuple[list[int], list[int]]:
    """Returns (strongest, weakest) as move-log indices, never sharing a move.

    Moves are ranked by the change in evaluation when every move has one (a trained model),
    otherwise by grade score (the heuristic fallback).
    """
    use_evaluation = all(m["delta_value"] is not None for m in move_log)
    rank_key = "delta_value" if use_evaluation else "score"
    worst_first = sorted(range(len(move_log)), key=lambda i: move_log[i][rank_key])
    picks = min(MAX_PICKS, len(move_log) // 2)
    return worst_first[::-1][:picks], worst_first[:picks]
