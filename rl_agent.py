"""
Agent RL — PPO (MaskablePPO) wytrenowany przeciwko LogicAgent.

Bez modelu na dysku używa losowych akcji z maską eliksiru (fallback).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

from board import Board, NUM_ACTIONS

DEFAULT_MODEL_PATH = Path("models/ppo_cr_best.zip")


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
        from sb3_contrib import MaskablePPO

        self._model = MaskablePPO.load(str(path))
        self._model_path = path

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def choose_action(self, board: Board, player: int = 0) -> int:
        board.set_pending_play(player, None)

        if self._model is not None and player == 0:
            obs = board.get_observation(player)
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
