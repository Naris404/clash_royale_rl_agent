"""Reward: the match outcome plus potential-based shaping (ADR-0002), per decision step.

    r = λ_win · outcome
      + λ_crown · (γ·Φ_crown(s′) − Φ_crown(s))      Φ_crown = own Crowns − enemy Crowns
      + λ_hp    · (γ·Φ_hp(s′)    − Φ_hp(s))         Φ_hp = (own tower HP − enemy tower HP) / max

Φ(terminal) = 0, so the shaping cannot change the optimal policy. Every term is
antisymmetric between the players, so both get negated rewards for the same transition.
"""

from __future__ import annotations

from dataclasses import dataclass

from cr_rl.game.board import MAX_TOWER_HP_PER_PLAYER, Board

DEFAULT_GAMMA = 0.995  # PPO's discount, applied per decision step


@dataclass(frozen=True)
class RewardWeights:
    win: float = 1.0
    crown: float = 0.2
    hp: float = 0.5


def outcome(board: Board, player: int) -> float:
    """+1 for a win, -1 for a loss, 0 for a draw; 0 while the match runs."""
    if not board.done or board.winner is None:
        return 0.0
    return 1.0 if board.winner == player else -1.0


def potential(board: Board, player: int, weights: RewardWeights) -> float:
    """Weighted shaping potential Φ of the position for `player`; 0 once the match is over."""
    if board.done:
        return 0.0
    enemy = 1 - player
    crowns = board.crowns[player] - board.crowns[enemy]
    tower_hp = sum(t.hp if t.owner == player else -t.hp for t in board.towers)
    return weights.crown * crowns + weights.hp * tower_hp / MAX_TOWER_HP_PER_PLAYER


def step_reward(
    board: Board,
    player: int,
    weights: RewardWeights,
    gamma: float,
    previous_potential: float,
) -> float:
    """Reward of one decision step ending in `board`; `previous_potential` is Φ at its start."""
    return (
        weights.win * outcome(board, player)
        + gamma * potential(board, player, weights)
        - previous_potential
    )
