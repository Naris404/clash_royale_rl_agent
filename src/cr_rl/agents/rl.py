"""
Agent RL — PPO (MaskablePPO) wytrenowany przeciwko LogicAgent.

Bez modelu na dysku używa losowych akcji z maską eliksiru (fallback).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

from cr_rl.paths import MODELS_DIR
from typing import Optional

from cr_rl.game.board import Board, NUM_ACTIONS, OBS_DIM

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = MODELS_DIR / "ppo_cr_best.zip"


def load_compatible_model(path: Path):
    """Load a MaskablePPO checkpoint; None when its action/observation layout is outdated."""
    from sb3_contrib import MaskablePPO

    model = MaskablePPO.load(str(path))
    if model.action_space.n != NUM_ACTIONS or model.observation_space.shape != (OBS_DIM,):
        logger.warning(
            "Ignoring %s: trained for %s actions / obs %s, current layout is %d / (%d,)",
            path, model.action_space.n, model.observation_space.shape, NUM_ACTIONS, OBS_DIM,
        )
        return None
    return model


class RandomAgent:
    """Bot z losowymi legalnymi zagraniami (maska eliksiru + strefy)."""

    def __init__(self, *, seed: int | None = None, play_chance: float = 0.45):
        self._rng = random.Random(seed)
        self.play_chance = play_chance

    def choose_action(self, board: Board, player: int = 0) -> int:
        board.set_pending_play(player, None)
        mask = board.valid_action_mask(player)
        affordable = [i for i in range(NUM_ACTIONS) if mask[i]]
        if affordable and self._rng.random() < self.play_chance:
            return self._rng.choice(affordable)
        return 0

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng.seed(seed)


class RLAgent:
    """
    Gracz RL: akcja 0..NUM_ACTIONS-1 (noop lub karta ze slotu ręki + strefa).

    Po treningu wczytuje MaskablePPO z models/ppo_cr_best.zip (lub podanej ścieżki).
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        model_path: str | Path | None = DEFAULT_MODEL_PATH,
        deterministic: bool = True,
        play_chance: float = 0.5,
    ):
        self._rng = random.Random(seed)
        self.deterministic = deterministic
        self.play_chance = play_chance
        self._model = None
        self._model_path: Path | None = None

        if model_path is not None:
            path = Path(model_path)
            if path.is_file():
                self._load_model(path)

    def _load_model(self, path: Path) -> None:
        self._model = load_compatible_model(path)
        self._model_path = path if self._model is not None else None

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def choose_action(self, board: Board, player: int = 0) -> int:
        board.set_pending_play(player, None)

        if self._model is not None:
            obs = board.get_observation(player)  # widok lustrzany dla P1
            mask = board.valid_action_mask(player)
            action, _ = self._model.predict(
                obs,
                deterministic=self.deterministic,
                action_masks=mask,
            )
            return int(action)

        return self._random_valid_action(board, player)

    def _random_valid_action(self, board: Board, player: int) -> int:
        mask = board.valid_action_mask(player)
        affordable = [i for i in range(NUM_ACTIONS) if mask[i]]
        if affordable and self._rng.random() < self.play_chance:
            return self._rng.choice(affordable)
        return 0

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng.seed(seed)
