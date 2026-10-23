"""Past-self pool: frozen earlier versions of the agent that it trains against in the self-play phase.

The pool starts as the last 3 selection checkpoints. Every `POOL_UPDATE_INTERVAL` steps the
current model is a candidate: it plays `ADMISSION_MATCHES` matches against the pool and, if it
wins at least `ADMISSION_WIN_RATE` of them, replaces the oldest Past self. Every decision follows
from the run seed and the checkpoint files.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from sb3_contrib import MaskablePPO

from cr_rl.training.matches import MatchResult, play_matches
from cr_rl.training.selection import CheckpointResult

POOL_SIZE = 3
POOL_UPDATE_INTERVAL = 100_000  # env steps between admission checks
ADMISSION_MATCHES = 50
ADMISSION_WIN_RATE = 0.45
POOL_LOG_FILENAME = "pool.json"
_ADMISSION_STREAM = 1  # keeps admission seeds apart from the episode seeds of the same run seed


@dataclass(frozen=True)
class PoolDecision:
    """One admission check: the candidate's record against the pool and what it changed."""

    step: int
    wins: int
    matches: int
    joined: bool
    replaced: str | None  # file name of the Past self that left the pool
    members: tuple[str, ...]  # file names of the pool afterwards, oldest first


def load_past_self(path: Path) -> MaskablePPO:
    return MaskablePPO.load(str(path), device="cpu")


def admission_seeds(run_seed: int, step: int, matches: int = ADMISSION_MATCHES) -> list[int]:
    """The seeds of one admission check, derived from the run seed and the training step."""
    rng = np.random.default_rng(np.random.SeedSequence([run_seed, _ADMISSION_STREAM, step]))
    return [int(seed) for seed in rng.choice(2**31 - 1, size=matches, replace=False)]


def play_against_pool(
    candidate: Path, members: Sequence[Path], seeds: Sequence[int]
) -> list[MatchResult]:
    """The candidate (player 0) plays the seeds, spread over the pool members in rotation."""
    model = load_past_self(candidate)
    results = []
    for i, member in enumerate(members):
        results += play_matches(model, load_past_self(member), seeds[i :: len(members)])
    return results


class PastSelfPool:
    """The Past selves (checkpoint files, oldest first) and the record of admission checks."""

    def __init__(self, members: Sequence[Path], *, run_seed: int, matches: int = ADMISSION_MATCHES):
        self.members = list(members)
        self.run_seed = run_seed
        self.matches = matches
        self.decisions: list[PoolDecision] = []

    @classmethod
    def from_selection(
        cls, results: Sequence[CheckpointResult], checkpoint_dir: Path, **kwargs
    ) -> PastSelfPool:
        """Seed the pool with the last `POOL_SIZE` selection checkpoints."""
        return cls([checkpoint_dir / result.path for result in results[-POOL_SIZE:]], **kwargs)

    def consider(
        self,
        candidate: Path,
        step: int,
        play: Callable[[Path, Sequence[Path], Sequence[int]], list[MatchResult]] = play_against_pool,
    ) -> PoolDecision:
        """Admission check: the candidate replaces the oldest Past self if it wins often enough."""
        seeds = admission_seeds(self.run_seed, step, self.matches)
        results = play(candidate, self.members, seeds)
        wins = sum(result.winner == 0 for result in results)
        joined = wins >= ADMISSION_WIN_RATE * self.matches
        replaced = None
        if joined:
            replaced = self.members.pop(0).name
            self.members.append(candidate)
        decision = PoolDecision(
            step, wins, self.matches, joined, replaced, tuple(member.name for member in self.members)
        )
        self.decisions.append(decision)
        return decision

    def save_log(self, path: Path) -> None:
        path.write_text(json.dumps([asdict(d) for d in self.decisions], indent=2), encoding="utf-8")
