"""Compatibility CLI shim for :mod:`cr_rl.reports.docx`."""
import _cr_rl_compat  # noqa: F401
from cr_rl.reports.docx import *  # noqa: F401,F403

if __name__ == "__main__":
    build()
