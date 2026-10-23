"""The Ablation arms: the full configuration and the three variants that each change one setting."""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from cr_rl.experiments.curriculum import CurriculumMetrics, curriculum_metrics, warm_up_steps
from cr_rl.game.rewards import RewardWeights
from cr_rl.jsonio import write_json
from cr_rl.training.phases import FULL, NO_CURRICULUM, NO_SELF_PLAY
from cr_rl.training.report import (
    ENV_VERSION,
    MANIFEST_FILENAME,
    REPORT_SEEDS,
    SHIP_GATE_OPPONENT,
    ConfigurationReport,
    evaluate_configuration,
)
from cr_rl.training.selection import SELECTION_FILENAME, load_selection
from cr_rl.training.train import DEFAULT_N_ENVS, train

logger = logging.getLogger(__name__)

TRAINING_SEEDS = (0, 1, 2)  # one run per seed and arm


@dataclass(frozen=True)
class Arm:
    """A training configuration: the phase layout and the reward weights."""

    layout: str = FULL
    reward_weights: RewardWeights = RewardWeights()


ARMS: dict[str, Arm] = {
    FULL: Arm(),
    "no-shaping": Arm(reward_weights=RewardWeights(crown=0.0, hp=0.0)),
    NO_CURRICULUM: Arm(layout=NO_CURRICULUM),
    NO_SELF_PLAY: Arm(layout=NO_SELF_PLAY),
}


def add_plan_arguments(parser: argparse.ArgumentParser, *, timesteps: int, out_dir: Path) -> None:
    """The command-line options every run plan shares."""
    parser.add_argument("--timesteps", type=int, default=timesteps)
    parser.add_argument("--n-envs", type=int, default=DEFAULT_N_ENVS)
    parser.add_argument("--out", type=Path, default=out_dir)
    parser.add_argument("--report-episodes", type=int, default=len(REPORT_SEEDS), help="Report seeds to play")


def plan_options(args: argparse.Namespace) -> dict[str, Any]:
    """Keyword arguments of a run plan for the options parsed by `add_plan_arguments`."""
    return dict(
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        out_dir=args.out,
        report_seeds=range(REPORT_SEEDS.start, REPORT_SEEDS.start + args.report_episodes),
    )


def train_arm(arm: str, *, seed: int, timesteps: int, run_dir: Path, **train_kwargs: Any) -> None:
    """Train one run of an arm into `run_dir`, with a manifest of what produced it.

    A run that already has its final model is kept, so an interrupted plan can be started again.
    """
    if (run_dir / "ppo_cr_final.zip").is_file():
        logger.info("Skipping %s: already trained", run_dir.name)
        return
    configuration = ARMS[arm]
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "arm": arm,
        "seed": seed,
        "timesteps": timesteps,
        "layout": configuration.layout,
        "reward_weights": asdict(configuration.reward_weights),
        "env_version": ENV_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(run_dir / MANIFEST_FILENAME, manifest)
    train(
        timesteps=timesteps,
        seed=seed,
        layout=configuration.layout,
        reward_weights=configuration.reward_weights,
        model_dir=run_dir,
        **train_kwargs,
    )


def run_name(arm: str, seed: int) -> str:
    return f"{arm}-seed{seed}"


def run_arm(
    arm: str,
    *,
    seeds: Sequence[int],
    timesteps: int,
    out_dir: Path,
    report_seeds: range = REPORT_SEEDS,
    **train_kwargs: Any,
) -> tuple[ConfigurationReport, dict[str, CurriculumMetrics]]:
    """Train an arm once per seed, evaluate every run on the Report seeds, read its curriculum metrics."""
    run_dirs = [out_dir / run_name(arm, seed) for seed in seeds]
    for seed, run_dir in zip(seeds, run_dirs):
        train_arm(arm, seed=seed, timesteps=timesteps, run_dir=run_dir, **train_kwargs)
    report = evaluate_configuration(run_dirs, seeds=report_seeds)
    metrics = {
        run.name: curriculum_metrics(
            load_selection(run_dir / SELECTION_FILENAME),
            warm_up_steps=warm_up_steps(timesteps),
            report_win_rate=run.evaluation.against(SHIP_GATE_OPPONENT).win_rate,
        )
        for run, run_dir in zip(report.runs, run_dirs)
    }
    return report, metrics
