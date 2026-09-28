"""Compatibility shim for :mod:`cr_rl.stats.reporting`."""
import _cr_rl_compat  # noqa: F401
from cr_rl.stats.reporting import *  # noqa: F401,F403

if __name__ == "__main__":
    main()
