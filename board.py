"""Compatibility shim for :mod:`cr_rl.game.board`."""
import _cr_rl_compat  # noqa: F401
from cr_rl.game.board import *  # noqa: F401,F403  # pyright: ignore[reportMissingImports]
from cr_rl.game.board import run_random_episode  # pyright: ignore[reportMissingImports]

if __name__ == "__main__":
    run_random_episode()
