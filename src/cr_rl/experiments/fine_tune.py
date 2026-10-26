"""Fine-tuning: continue a Final model against the Past-self pool, with a manifest of its source."""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from cr_rl.game.rewards import RewardWeights
from cr_rl.jsonio import write_json
from cr_rl.training.phases import FINE_TUNE
from cr_rl.training.report import ENV_VERSION, MANIFEST_FILENAME
from cr_rl.training.train import train

logger = logging.getLogger(__name__)


def fine_tune(
    source: Path,
    *,
    seed: int,
    timesteps: int,
    run_dir: Path,
    permanent_members: Sequence[Path] = (),
    **train_kwargs: Any,
) -> None:
    """Fine-tune the model file `source` for `timesteps` in the v3 mix, into `run_dir`.

    The pool starts with `source` as its only Past self plus the `permanent_members`
    (Exploiters). A run that already has its final model is kept.
    """
    if (run_dir / "ppo_cr_final.zip").is_file():
        logger.info("Skipping %s: already trained", run_dir.name)
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / MANIFEST_FILENAME,
        {
            "arm": FINE_TUNE,
            "seed": seed,
            "timesteps": timesteps,
            "layout": FINE_TUNE,
            "reward_weights": asdict(RewardWeights()),
            "source_model": str(source),
            "permanent_members": [str(member) for member in permanent_members],
            "env_version": ENV_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    train(
        timesteps=timesteps,
        seed=seed,
        layout=FINE_TUNE,
        resume=source,
        permanent_members=list(permanent_members),
        model_dir=run_dir,
        **train_kwargs,
    )
