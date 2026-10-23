"""Checkpoint selection: a run's model is the checkpoint with the best win rate on Selection seeds."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from stable_baselines3.common.callbacks import BaseCallback

from cr_rl.training.matches import play_matches, summarize_matches, wilson_interval

# Selection seeds never overlap Report seeds or training episode seeds (which are ≥ 0 and
# derived from the run seed); they are used to choose a checkpoint, never to report results.
SELECTION_SEEDS = range(1_000_000, 1_000_100)
SELECTION_INTERVAL = 50_000  # env steps between selection rounds
SELECTION_FILENAME = "selection.json"


@dataclass(frozen=True)
class CheckpointResult:
    """One evaluated checkpoint: its training step, wins on the Selection seeds and file name."""

    step: int
    wins: int
    episodes: int
    path: str  # file name inside the run's `checkpoints/` directory

    @property
    def win_rate(self) -> float:
        return self.wins / self.episodes


def select_checkpoint(results: Sequence[CheckpointResult]) -> CheckpointResult:
    """Highest win rate; a tie goes to the later checkpoint."""
    return max(results, key=lambda result: (result.win_rate, result.step))


def save_selection(results: Sequence[CheckpointResult], path: Path) -> None:
    path.write_text(json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8")


def load_selection(path: Path) -> list[CheckpointResult]:
    return [CheckpointResult(**record) for record in json.loads(path.read_text(encoding="utf-8"))]


class SelectionCallback(BaseCallback):
    """Plays the current policy vs LogicAgent on the Selection seeds and keeps the checkpoint.

    A round runs every `interval` env steps; `evaluate()` runs one more for the final model.
    Rounds are logged with their Wilson interval and recorded in `selection.json`.
    """

    def __init__(
        self,
        model_dir: Path,
        *,
        interval: int = SELECTION_INTERVAL,
        seeds: Sequence[int] = SELECTION_SEEDS,
    ):
        super().__init__()
        self.model_dir = model_dir
        self.interval = interval
        self.seeds = seeds
        self.results: list[CheckpointResult] = []
        self._last_round = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_round >= self.interval:
            self.evaluate()
        return True

    def evaluate(self) -> None:
        """Evaluate and keep the current model, unless this step was already evaluated."""
        step = self.model.num_timesteps
        self._last_round = step
        if self.results and self.results[-1].step == step:
            return
        stats = summarize_matches(play_matches(self.model, "logic", self.seeds))
        checkpoint = Path("checkpoints") / f"ppo_cr_{step}_steps.zip"
        (self.model_dir / checkpoint.parent).mkdir(exist_ok=True)
        self.model.save(str(self.model_dir / checkpoint))
        result = CheckpointResult(step, int(stats["wins"]), int(stats["episodes"]), checkpoint.name)
        self.results.append(result)
        save_selection(self.results, self.model_dir / SELECTION_FILENAME)
        low, high = wilson_interval(result.wins, result.episodes)
        self.logger.record("selection/win_rate", result.win_rate)
        print(f"Selection @ {step:,} steps: win rate {result.win_rate:.1%} (95% CI {low:.1%} to {high:.1%})")
