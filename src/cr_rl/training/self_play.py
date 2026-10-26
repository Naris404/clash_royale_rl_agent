"""Past-self pool: frozen earlier versions of the agent that it trains against in the self-play phase.

The pool starts as the last `size` selection checkpoints (3 in a v2 run; 6 in v3 fine-tuning, where
it starts from the model being fine-tuned). Every `POOL_UPDATE_INTERVAL` steps the current model is a
candidate: it plays `ADMISSION_MATCHES` matches against the pool, spread over all members, and if it
wins at least `ADMISSION_WIN_RATE` of them it joins, replacing the oldest Past self once the pool is
full. The check records the candidate's win rate against each member. Permanent members (Exploiters)
are never replaced and don't count towards `size`. Layouts with PFSP draw their pool envs from these
win rates. Every decision follows from the run seed and the checkpoint files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from sb3_contrib import MaskablePPO

from cr_rl.jsonio import write_json
from cr_rl.training.matches import MatchResult, play_matches
from cr_rl.training.seeds import TRAINING_SEED_LIMIT
from cr_rl.training.selection import CheckpointResult

POOL_SIZE = 3
FINE_TUNE_POOL_SIZE = 6
POOL_UPDATE_INTERVAL = 100_000  # env steps between admission checks
ADMISSION_MATCHES = 50
ADMISSION_WIN_RATE = 0.45
POOL_LOG_FILENAME = "pool.json"
_ADMISSION_STREAM = 1  # keeps admission seeds apart from the episode seeds of the same run seed
_PFSP_STREAM = 2  # and the PFSP draws apart from the admission seeds


@dataclass(frozen=True)
class PoolDecision:
    """One admission check: the candidate's record against the pool and what it changed."""

    step: int
    wins: int
    matches: int
    joined: bool
    replaced: str | None  # file name of the Past self that left the pool
    members: tuple[str, ...]  # file names of the pool afterwards: Past selves oldest first, then permanent
    member_win_rates: dict[str, float] = field(default_factory=dict)  # candidate's win rate per member


def load_past_self(path: Path) -> MaskablePPO:
    return MaskablePPO.load(str(path), device="cpu")


def admission_seeds(run_seed: int, step: int, matches: int = ADMISSION_MATCHES) -> list[int]:
    """The seeds of one admission check, derived from the run seed and the training step."""
    rng = np.random.default_rng(np.random.SeedSequence([run_seed, _ADMISSION_STREAM, step]))
    return [int(seed) for seed in rng.choice(TRAINING_SEED_LIMIT, size=matches, replace=False)]


def member_seeds(seeds: Sequence[int], n_members: int) -> list[Sequence[int]]:
    """The seeds each pool member plays: the check's seeds spread over the members in rotation."""
    return [seeds[i::n_members] for i in range(n_members)]


def pfsp_weights(win_rates: Sequence[float | None]) -> list[float]:
    """Hardest-first PFSP weights (1 - win rate)^2; a member without a measured win rate weighs 1."""
    return [1.0 if rate is None else (1.0 - rate) ** 2 for rate in win_rates]


def play_against_pool(
    candidate: Path, members: Sequence[Path], seeds: Sequence[int]
) -> list[MatchResult]:
    """The candidate (player 0) plays each member on its share of the seeds."""
    model = load_past_self(candidate)
    results = []
    for member, member_share in zip(members, member_seeds(seeds, len(members))):
        results += play_matches(model, load_past_self(member), member_share)
    return results


class PastSelfPool:
    """The Past selves (checkpoint files, oldest first), the permanent members that are never
    replaced and don't count towards `size`, and the record of admission checks."""

    def __init__(
        self,
        past_selves: Sequence[Path],
        *,
        run_seed: int,
        matches: int = ADMISSION_MATCHES,
        size: int = POOL_SIZE,
        permanent: Sequence[Path] = (),
    ):
        self.past_selves = list(past_selves)
        self.permanent = list(permanent)
        self.size = size
        self.run_seed = run_seed
        self.matches = matches
        self.win_rates: dict[Path, float] = {}  # latest measured win rate against each member
        self.decisions: list[PoolDecision] = []

    @property
    def members(self) -> list[Path]:
        """Every opponent of the pool: the Past selves (oldest first), then the permanent members."""
        return self.past_selves + self.permanent

    @classmethod
    def from_selection(
        cls, results: Sequence[CheckpointResult], checkpoint_dir: Path, **kwargs
    ) -> PastSelfPool:
        """Seed the pool with the last `size` selection checkpoints."""
        size = kwargs.get("size", POOL_SIZE)
        return cls([checkpoint_dir / result.path for result in results[-size:]], **kwargs)

    def consider(
        self,
        candidate: Path,
        step: int,
        play: Callable[[Path, Sequence[Path], Sequence[int]], list[MatchResult]] = play_against_pool,
    ) -> PoolDecision:
        """Admission check: the candidate replaces the oldest Past self if it wins often enough."""
        seeds = admission_seeds(self.run_seed, step, self.matches)
        results = play(candidate, self.members, seeds)
        won = {result.seed: result.winner == 0 for result in results}
        wins = sum(won.values())
        member_win_rates = {}
        for member, share in zip(self.members, member_seeds(seeds, len(self.members))):
            outcomes = [won[seed] for seed in share if seed in won]
            if outcomes:
                member_win_rates[member] = sum(outcomes) / len(outcomes)
        self.win_rates.update(member_win_rates)
        joined = wins >= ADMISSION_WIN_RATE * self.matches
        replaced = None
        if joined:
            if len(self.past_selves) >= self.size:
                oldest = self.past_selves.pop(0)
                replaced = oldest.name
                self.win_rates.pop(oldest, None)
            self.past_selves.append(candidate)
        decision = PoolDecision(
            step, wins, self.matches, joined, replaced,
            tuple(member.name for member in self.members),
            {member.name: rate for member, rate in member_win_rates.items()},
        )
        self.decisions.append(decision)
        return decision

    def assign(self, n: int, step: int) -> list[Path]:
        """PFSP: `n` draws (with replacement) from the pool, harder members more often.

        The draws follow from the run seed and `step` alone. If every weight is 0 (the agent
        beats everyone every time), members are drawn uniformly.
        """
        weights = np.array(pfsp_weights([self.win_rates.get(member) for member in self.members]))
        if not weights.any():
            weights = np.ones(len(weights))
        rng = np.random.default_rng(np.random.SeedSequence([self.run_seed, _PFSP_STREAM, step]))
        members = self.members
        return [members[i] for i in rng.choice(len(members), size=n, p=weights / weights.sum())]

    def save_log(self, path: Path) -> None:
        write_json(path, [asdict(d) for d in self.decisions])
