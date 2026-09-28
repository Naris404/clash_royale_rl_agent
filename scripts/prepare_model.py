"""Compatibility CLI shim for :mod:`cr_rl.deploy.prepare_model`."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import _cr_rl_compat  # noqa: F401
from cr_rl.deploy.prepare_model import main

if __name__ == "__main__":
    main()
