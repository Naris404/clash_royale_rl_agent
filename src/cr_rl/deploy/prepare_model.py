"""Fetch an optional deployment model before the API process starts."""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

from cr_rl.paths import MODELS_DIR


def main() -> None:
    target = Path(os.getenv("MODEL_PATH", str(MODELS_DIR / "ppo_cr_best.zip")))
    if target.is_file():
        print(f"Using bundled RL model: {target}")
        return

    model_url = os.getenv("MODEL_URL")
    if not model_url:
        print(f"No model at {target}; starting with heuristic coach.", file=sys.stderr)
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".download")
    print(f"Downloading RL model to {target}...")
    try:
        urllib.request.urlretrieve(model_url, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
