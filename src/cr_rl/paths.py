"""Repository data paths shared by all package modules."""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_PROJECT_ROOT = PACKAGE_ROOT.parents[1]
default_root = SOURCE_PROJECT_ROOT if (SOURCE_PROJECT_ROOT / "pyproject.toml").is_file() else Path.cwd()
PROJECT_ROOT = Path(os.getenv("CR_RL_PROJECT_ROOT", default_root)).resolve()
MODELS_DIR = PROJECT_ROOT / "models"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENT_RUNS_DIR = EXPERIMENTS_DIR / "runs"
WEB_DIR = PROJECT_ROOT / "web"
WEB_DIST_DIR = WEB_DIR / "dist"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
REPORT_OUTPUT = PROJECT_ROOT / "RAPORT_PROJEKTU.docx"
