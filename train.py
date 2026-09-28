"""Compatibility shim for :mod:`cr_rl.training.train`."""
import _cr_rl_compat  # noqa: F401
from cr_rl.training.train import *  # noqa: F401,F403

if __name__ == "__main__":
    main()
