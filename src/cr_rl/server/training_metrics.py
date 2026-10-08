"""Read live Stable-Baselines3/TensorBoard metrics for the web dashboard."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from cr_rl.paths import MODELS_DIR

_cache_lock = threading.Lock()
_cache_at = 0.0
_cache_value: dict[str, Any] | None = None


def _run_updated_at(path: Path) -> float:
    return max(
        (event.stat().st_mtime for event in path.glob("events.out.tfevents.*")),
        default=path.stat().st_mtime,
    )


def _downsample(items: list[Any], limit: int) -> list[Any]:
    if len(items) <= limit:
        return items
    indices = np.linspace(0, len(items) - 1, limit, dtype=int)
    return [items[int(index)] for index in indices]


def read_training_metrics(
    models_dir: Path = MODELS_DIR,
    *,
    max_points: int = 300,
) -> dict[str, Any]:
    tb_root = models_dir / "tb"
    run_dirs = sorted(
        (path for path in tb_root.iterdir() if path.is_dir()) if tb_root.is_dir() else [],
        key=_run_updated_at,
        reverse=True,
    )
    if not run_dirs:
        return {
            "training": False,
            "latest_run": None,
            "latest_step": 0,
            "series": [],
            "runs": [],
            "evaluation": None,
            "models": [],
        }

    latest = run_dirs[0]
    latest_update = _run_updated_at(latest)
    accumulator = EventAccumulator(str(latest), size_guidance={"scalars": 0})
    accumulator.Reload()

    series = []
    latest_step = 0
    for tag in sorted(accumulator.Tags().get("scalars", [])):
        events = accumulator.Scalars(tag)
        if not events:
            continue
        latest_step = max(latest_step, int(events[-1].step))
        sampled = _downsample(events, max_points)
        series.append(
            {
                "tag": tag,
                "steps": [int(event.step) for event in sampled],
                "values": [float(event.value) for event in sampled],
                "latest": float(events[-1].value),
                "minimum": float(min(event.value for event in events)),
                "maximum": float(max(event.value for event in events)),
                "points": len(events),
            }
        )

    evaluation = None
    eval_path = models_dir / "eval" / "evaluations.npz"
    if eval_path.is_file():
        with np.load(eval_path) as data:
            rewards = np.asarray(data["results"], dtype=float)
            lengths = np.asarray(data["ep_lengths"], dtype=float)
            evaluation = {
                "steps": np.asarray(data["timesteps"], dtype=int).tolist(),
                "mean_rewards": (
                    rewards.mean(axis=1) if rewards.ndim > 1 else rewards
                ).tolist(),
                "mean_lengths": (
                    lengths.mean(axis=1) if lengths.ndim > 1 else lengths
                ).tolist(),
            }

    model_files = sorted(
        models_dir.glob("**/*.zip"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return {
        # Evaluating 50 episodes can stall TB writes for a few minutes.
        "training": time.time() - latest_update < 300.0,
        "latest_run": latest.name,
        "latest_step": latest_step,
        "updated_at": latest_update,
        "series": series,
        "runs": [
            {"name": path.name, "updated_at": _run_updated_at(path)}
            for path in run_dirs
        ],
        "evaluation": evaluation,
        "models": [
            {
                "name": path.name,
                "path": str(path.relative_to(models_dir.parent)),
                "size_bytes": path.stat().st_size,
                "updated_at": path.stat().st_mtime,
            }
            for path in model_files
        ],
    }


def cached_training_metrics(ttl_seconds: float = 2.0) -> dict[str, Any]:
    global _cache_at, _cache_value
    now = time.monotonic()
    with _cache_lock:
        if _cache_value is not None and now - _cache_at < ttl_seconds:
            return _cache_value
        _cache_value = read_training_metrics()
        _cache_at = now
        return _cache_value
