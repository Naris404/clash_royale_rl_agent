"""Ablation plan: 4 arms x 3 seeds x 500k steps, per-arm aggregation, full vs no-self-play head-to-head.

Usage:
  cr-rl-ablations
  cr-rl-ablations --timesteps 20000 --n-envs 2 --report-episodes 10   # quick check
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sb3_contrib import MaskablePPO

from cr_rl.experiments.arms import ARMS, run_arm
from cr_rl.paths import EXPERIMENT_RUNS_DIR
from cr_rl.training.head_to_head import evaluate_head_to_head
from cr_rl.training.report import (
    REPORT_FILENAME,
    REPORT_SEEDS,
    configuration_rows,
    selected_checkpoint,
)
from cr_rl.training.train import DEFAULT_N_ENVS

ABLATION_DIR = EXPERIMENT_RUNS_DIR / "ablations"
ABLATION_TIMESTEPS = 500_000
TRAINING_SEEDS = (0, 1, 2)
HEAD_TO_HEAD = ("full", "no-self-play")  # what the Past-self pool adds beyond the scripted opponents


def run_ablations(
    *,
    timesteps: int = ABLATION_TIMESTEPS,
    seeds: tuple[int, ...] = TRAINING_SEEDS,
    out_dir: Path = ABLATION_DIR,
    report_seeds: range = REPORT_SEEDS,
    **train_kwargs: Any,
) -> None:
    """Train and evaluate every arm; results land in `out_dir/results.json`."""
    reports, curriculum, rows = {}, {}, []
    for arm in ARMS:
        report, metrics = run_arm(
            arm, seeds=seeds, timesteps=timesteps, out_dir=out_dir, report_seeds=report_seeds, **train_kwargs
        )
        reports[arm] = report
        curriculum.update({name: asdict(run_metrics) for name, run_metrics in metrics.items()})
        rows += [{**row, "arm": arm} for row in configuration_rows(report)]
        logic = report.spread["logic"]
        print(f"{arm}: win rate vs logic {logic.mean:.1%} +/- {logic.std:.1%}, shipped {report.shipped}")

    name_a, name_b = (reports[arm].shipped for arm in HEAD_TO_HEAD)
    models = [MaskablePPO.load(str(selected_checkpoint(out_dir / name)[1])) for name in (name_a, name_b)]
    head_to_head = evaluate_head_to_head(*models, name_a=name_a, name_b=name_b, seeds=report_seeds)
    rows.append(
        {**asdict(head_to_head.overall), "config": f"{name_a} vs {name_b} (head-to-head)", "run": name_a, "arm": HEAD_TO_HEAD[0]}
    )

    payload = {
        "experiment": "ablations",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "timesteps": timesteps,
        "training_seeds": list(seeds),
        "arms": {arm: asdict(report) for arm, report in reports.items()},
        "curriculum": curriculum,
        "head_to_head": asdict(head_to_head),
        "results": rows,
    }
    (out_dir / REPORT_FILENAME).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    overall = head_to_head.overall
    print(
        f"Head-to-head {name_a} vs {name_b}: win rate {overall.win_rate:.1%} "
        f"(95% CI {overall.ci_low:.1%} to {overall.ci_high:.1%}), W/L/D {overall.wins}/{overall.losses}/{overall.draws}"
    )
    print(f"Saved: {out_dir / REPORT_FILENAME}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation plan: 4 arms, 3 seeds, Report-seed evaluation")
    parser.add_argument("--timesteps", type=int, default=ABLATION_TIMESTEPS)
    parser.add_argument("--n-envs", type=int, default=DEFAULT_N_ENVS)
    parser.add_argument("--out", type=Path, default=ABLATION_DIR)
    parser.add_argument("--report-episodes", type=int, default=len(REPORT_SEEDS), help="Report seeds to play")
    args = parser.parse_args()
    run_ablations(
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        out_dir=args.out,
        report_seeds=range(REPORT_SEEDS.start, REPORT_SEEDS.start + args.report_episodes),
    )


if __name__ == "__main__":
    main()
