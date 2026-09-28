"""Compatibility entry point for :mod:`cr_rl.experiments.coach_accuracy`."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _cr_rl_compat  # noqa: F401,E402
from cr_rl.experiments.coach_accuracy import *  # noqa: F401,F403,E402

if __name__ == "__main__":
    main()
