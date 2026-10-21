"""Decision schedule: when a player is asked for an action.

A Decision point is a tick where at least one card play is legal and no noop hold
is running. Choosing noop starts a hold of `hold_ticks` ticks before the player is
asked again. The training env and the Session RL opponent share this component so
the live policy plays at the timing it was trained with.
"""

from __future__ import annotations

from cr_rl.game.board import Board

NOOP_HOLD_TICKS = 3  # 0.3 s at 10 Hz


class DecisionSchedule:
    """Tracks the noop hold of one player; call `advance()` once per simulated tick."""

    def __init__(self, hold_ticks: int = NOOP_HOLD_TICKS):
        if hold_ticks < 1:
            raise ValueError("hold_ticks must be at least 1")
        self.hold_ticks = hold_ticks
        self._hold_left = 0

    def reset(self) -> None:
        self._hold_left = 0

    def is_decision_point(self, board: Board, player: int) -> bool:
        return self._hold_left == 0 and bool(board.valid_action_mask(player)[1:].any())

    def record(self, action: int) -> None:
        """Register the chosen action; noop starts the hold."""
        if action == 0:
            self._hold_left = self.hold_ticks

    def advance(self) -> None:
        self._hold_left = max(0, self._hold_left - 1)
