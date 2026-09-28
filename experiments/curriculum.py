"""Compatibility shim for :mod:`cr_rl.experiments.curriculum`."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _cr_rl_compat  # noqa: F401
from cr_rl.experiments.curriculum import *  # noqa: F401,F403

if __name__ == "__main__":
    main()
