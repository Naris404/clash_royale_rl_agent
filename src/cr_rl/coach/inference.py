"""Extracting the full policy from MaskablePPO: action probabilities + V(s).

SB3 `predict` returns only the chosen action — the coach needs the distribution
(how good the alternatives are) and the state value (position eval bar).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from cr_rl.agents.rl import load_compatible_model
from cr_rl.game.board import NUM_ACTIONS, Board
from cr_rl.paths import MODELS_DIR

DEFAULT_MODEL_PATH = MODELS_DIR / "ppo_cr_best.zip"


@dataclass
class PolicyEstimate:
    """One state as evaluated by the network: best action, distribution, V(s)."""

    action: int
    probs: np.ndarray  # (NUM_ACTIONS,), masked (0 for illegal)
    value: float
    top_k: list[tuple[int, float]] = field(default_factory=list)  # (action, p) descending


class PolicyInspector:
    """Thread-safe MaskablePPO inference with the policy distribution."""

    def __init__(
        self,
        model_path: str | Path | None = DEFAULT_MODEL_PATH,
        *,
        top_k: int = 3,
    ):
        self.top_k = top_k
        self._lock = threading.Lock()
        self._model = None
        self._model_path: Path | None = None
        if model_path is not None:
            path = Path(model_path)
            if path.is_file():
                self._load(path)

    def _load(self, path: Path) -> None:
        self._model = load_compatible_model(path)
        self._model_path = path if self._model is not None else None

    @property
    def available(self) -> bool:
        return self._model is not None

    @property
    def model_path(self) -> Path | None:
        return self._model_path

    def estimate(self, board: Board, player: int = 0) -> Optional[PolicyEstimate]:
        """Policy distribution + V(s) for the state from `player`'s perspective; None without a model."""
        if self._model is None:
            return None

        import torch

        obs = board.get_observation(player)
        mask = board.valid_action_mask(player)

        with self._lock, torch.no_grad():
            device = self._model.policy.device
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)

            distribution = self._model.policy.get_distribution(obs_t, mask_t)
            probs = distribution.distribution.probs.squeeze(0).cpu().numpy()
            value = float(self._model.policy.predict_values(obs_t).item())

        # clean up floating-point artifacts on illegal actions
        probs = np.where(mask, probs, 0.0).astype(np.float64)
        total = probs.sum()
        if total > 0:
            probs /= total

        order = np.argsort(probs)[::-1]
        top_k = [(int(a), float(probs[a])) for a in order[: self.top_k] if probs[a] > 0]
        return PolicyEstimate(
            action=int(np.argmax(probs)),
            probs=probs,
            value=value,
            top_k=top_k,
        )

    def value(self, board: Board, player: int = 0) -> Optional[float]:
        """V(s) only — faster path for the eval bar and key moments."""
        estimate = self.estimate(board, player)
        return estimate.value if estimate else None
