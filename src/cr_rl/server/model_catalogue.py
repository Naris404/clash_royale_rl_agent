"""Model catalogue — the fixed list of models a player may pick as the RL Opponent.

Players pick an entry by id, never a file. Every entry is loaded once when the
server starts; an entry whose file is missing or has an outdated layout is
listed as unavailable. To add a model, add an entry here and put its file in
`models/` (see docs/rl-agent.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from cr_rl.agents.rl import load_compatible_model
from cr_rl.paths import MODELS_DIR


@dataclass(frozen=True)
class CatalogueEntry:
    id: str
    label: dict[str, str]  # per language
    description: dict[str, str]  # per language
    filename: Optional[str]  # under models/; None for the Shipped model, which follows MODEL_PATH


DEFAULT_MODEL_ID = "shipped"

CATALOGUE = (
    CatalogueEntry(
        id=DEFAULT_MODEL_ID,
        label={"en": "Shipped model", "pl": "Model wdrożony"},
        description={
            "en": "The default opponent, and the model the Coach uses.",
            "pl": "Domyślny przeciwnik i model, z którego korzysta Trener.",
        },
        filename=None,
    ),
    CatalogueEntry(
        id="v2-shipped",
        label={"en": "v2 model", "pl": "Model v2"},
        description={
            "en": "The first shipped model: wins 93.5% of matches against the rule-based bot.",
            "pl": "Pierwszy model wdrożony: wygrywa 93,5% meczów z botem regułowym.",
        },
        filename="ppo_cr_v2_shipped.zip",
    ),
    CatalogueEntry(
        id="v2-self-play",
        label={"en": "Self-play model", "pl": "Model z self-play"},
        description={
            "en": (
                "400k more training steps against past versions of itself: wins 78% against the "
                "rule-based bot, but beats the shipped model in 77.5% of head-to-head matches."
            ),
            "pl": (
                "400 tys. kroków treningu więcej przeciwko własnym wcześniejszym wersjom: wygrywa 78% meczów "
                "z botem regułowym, ale pokonuje model wdrożony w 77,5% bezpośrednich pojedynków."
            ),
        },
        filename="ppo_cr_self_play.zip",
    ),
    CatalogueEntry(
        id="v3",
        label={"en": "v3 model", "pl": "Model v3"},
        description={
            "en": (
                "Trained further after self-play, against its past versions and opponents trained to find "
                "its weaknesses: wins 73.5% against the rule-based bot, but beats the shipped model in 78% of "
                "head-to-head matches."
            ),
            "pl": (
                "Trenowany dalej po self-play, przeciwko własnym wcześniejszym wersjom i przeciwnikom "
                "uczonym szukać jego słabości: wygrywa 73,5% meczów z botem regułowym, ale pokonuje model "
                "wdrożony w 78% bezpośrednich pojedynków."
            ),
        },
        filename="ppo_cr_v3.zip",
    ),
)


class ModelNotAvailable(ValueError):
    """The requested id is not in the catalogue, or its model could not be loaded."""


class ModelCatalogue:
    """The entries of `CATALOGUE` with their loaded models (None when unavailable)."""

    def __init__(self, shipped_model: Any, models_dir: Path = MODELS_DIR):
        self._models: dict[str, Any] = {}
        for entry in CATALOGUE:
            if entry.filename is None:
                self._models[entry.id] = shipped_model
            else:
                path = models_dir / entry.filename
                self._models[entry.id] = load_compatible_model(path) if path.is_file() else None

    def describe(self) -> list[dict]:
        """Entries for the client: copy and availability, never a file path."""
        return [
            {
                "id": entry.id,
                "label": entry.label,
                "description": entry.description,
                "available": self._models[entry.id] is not None,
            }
            for entry in CATALOGUE
        ]

    def resolve(self, model_id: Optional[str]) -> Optional[tuple[str, Any]]:
        """The (id, model) to play; no id means the Shipped model, or None when it is unavailable."""
        if model_id is None:
            model = self._models[DEFAULT_MODEL_ID]
            return None if model is None else (DEFAULT_MODEL_ID, model)
        if model_id not in self._models:
            raise ModelNotAvailable(f"Unknown model: {model_id!r}")
        if self._models[model_id] is None:
            raise ModelNotAvailable(f"Model not available: {model_id!r}")
        return model_id, self._models[model_id]
