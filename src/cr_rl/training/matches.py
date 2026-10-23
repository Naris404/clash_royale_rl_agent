"""Match runner: the one "play matches on given seeds" loop behind every evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from cr_rl.agents.rl import RLAgent
from cr_rl.env.gym_env import ClashRoyaleEnv

WILSON_Z_95 = 1.959963984540054


@dataclass(frozen=True)
class MatchResult:
    """One finished match from player 0's side; `length` counts decision steps."""

    seed: int
    winner: int | None
    episode_return: float
    length: int


def play_matches(
    player: Any,
    opponent: str | Any,
    seeds: Sequence[int],
    *,
    deterministic: bool = True,
) -> list[MatchResult]:
    """Play one match per seed, `player` as player 0 against a Training opponent.

    `player` is a policy with `predict` (e.g. MaskablePPO) or an agent with
    `choose_action(board, player)`. `opponent` is whatever `ClashRoyaleEnv` accepts: a bot
    name ("logic", "logic-eps", "random") or a frozen policy, which plays as player 1.
    """
    if hasattr(player, "predict"):
        player = RLAgent(model=player, deterministic=deterministic)
    env = ClashRoyaleEnv(opponent=opponent)
    results = []
    for seed in seeds:
        env.reset(seed=seed)
        episode_return = 0.0
        length = 0
        done = False
        while not done:
            _obs, reward, terminated, truncated, info = env.step(player.choose_action(env.board, 0))
            episode_return += float(reward)
            length += 1
            done = terminated or truncated
        results.append(MatchResult(seed, info["winner"], episode_return, length))
    env.close()
    return results


def summarize_matches(results: Sequence[MatchResult]) -> dict[str, float]:
    """Outcome counts, win rate, mean return and mean length of player 0's matches."""
    n = len(results)
    wins = sum(result.winner == 0 for result in results)
    losses = sum(result.winner == 1 for result in results)
    return {
        "episodes": n,
        "wins": wins,
        "losses": losses,
        "draws": n - wins - losses,
        "win_rate": wins / n,
        "mean_reward": float(np.mean([result.episode_return for result in results])),
        "mean_length": float(np.mean([result.length for result in results])),
    }


def wilson_interval(wins: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval for a win rate of `wins` out of `n` matches."""
    if n == 0:
        return 0.0, 1.0
    p = wins / n
    z2 = WILSON_Z_95**2
    denominator = 1 + z2 / n
    centre = (p + z2 / (2 * n)) / denominator
    half_width = WILSON_Z_95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    return max(0.0, centre - half_width), min(1.0, centre + half_width)
