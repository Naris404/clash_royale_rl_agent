"""Checkpoint selection: a run's model is the checkpoint with the best win rate on Selection seeds."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from cr_rl.jsonio import write_json
from cr_rl.training.matches import play_matches, summarize_matches, wilson_interval
from cr_rl.training.seeds import SELECTION_SEEDS

logger = logging.getLogger(__name__)

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
    write_json(path, [asdict(result) for result in results])


def load_selection(path: Path) -> list[CheckpointResult]:
    return [CheckpointResult(**record) for record in json.loads(path.read_text(encoding="utf-8"))]


class SelectionCallback(BaseCallback):
    """Plays the current policy vs LogicAgent on the Selection seeds and keeps the checkpoint.

    A round runs every `interval` env steps; `evaluate(model)` runs one more for a model outside
    `learn` (a resumed one, the final one). Rounds are logged with their Wilson interval and
    recorded in `selection.json`.
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
            self.logger.record("selection/win_rate", self.evaluate().win_rate)
        return True

    def evaluate(self, model: MaskablePPO | None = None) -> CheckpointResult:
        """Evaluate and keep a model (default: the one being trained), unless its step was already evaluated."""
        model = model or self.model
        step = model.num_timesteps
        self._last_round = step
        if self.results and self.results[-1].step == step:
            return self.results[-1]
        stats = summarize_matches(play_matches(model, "logic", self.seeds))
        checkpoint = Path("checkpoints") / f"ppo_cr_{step}_steps.zip"
        (self.model_dir / checkpoint.parent).mkdir(exist_ok=True)
        model.save(str(self.model_dir / checkpoint))
        result = CheckpointResult(step, int(stats["wins"]), int(stats["episodes"]), checkpoint.name)
        self.results.append(result)
        save_selection(self.results, self.model_dir / SELECTION_FILENAME)
        low, high = wilson_interval(result.wins, result.episodes)
        logger.info("Selection @ %s steps: win rate %.1f%% (95%% CI %.1f%% to %.1f%%)",
                    f"{step:,}", result.win_rate * 100, low * 100, high * 100)
        return result
