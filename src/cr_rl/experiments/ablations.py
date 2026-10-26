"""Ablation plan: 4 arms x 3 seeds x 500k steps, per-arm aggregation, full vs no-self-play head-to-head.

The head-to-head is played between the two shipped runs and, as a matrix, between every full run and
every no-self-play run, so the result does not rest on one pair of models.

Usage:
  cr-rl-ablations
  cr-rl-ablations --timesteps 20000 --n-envs 2 --report-episodes 10   # quick check
"""

from __future__ import annotations

import argparse
import itertools
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sb3_contrib import MaskablePPO

from cr_rl.experiments.arms import ARMS, TRAINING_SEEDS, add_plan_arguments, plan_options, run_arm
from cr_rl.jsonio import write_json
from cr_rl.paths import EXPERIMENT_RUNS_DIR
from cr_rl.training.head_to_head import HeadToHead, evaluate_head_to_head
from cr_rl.training.phases import FULL, NO_SELF_PLAY
from cr_rl.training.report import (
    REPORT_FILENAME,
    REPORT_SEEDS,
    ConfigurationReport,
    configuration_rows,
    selected_checkpoint,
)

ABLATION_DIR = EXPERIMENT_RUNS_DIR / "ablations"
ABLATION_TIMESTEPS = 500_000
HEAD_TO_HEAD = (FULL, NO_SELF_PLAY)  # what the Past-self pool adds beyond the scripted opponents


def head_to_head_matrix(
    report_a: ConfigurationReport, report_b: ConfigurationReport, *, out_dir: Path, seeds: range
) -> list[HeadToHead]:
    """Every run of configuration A against every run of configuration B, each on its selected checkpoint."""
    names = [run.name for run in report_a.runs + report_b.runs]
    models = {name: MaskablePPO.load(str(selected_checkpoint(out_dir / name)[1])) for name in names}
    return [
        evaluate_head_to_head(models[run_a.name], models[run_b.name], name_a=run_a.name, name_b=run_b.name, seeds=seeds)
        for run_a, run_b in itertools.product(report_a.runs, report_b.runs)
    ]


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

    matrix = head_to_head_matrix(*(reports[arm] for arm in HEAD_TO_HEAD), out_dir=out_dir, seeds=report_seeds)
    name_a, name_b = (reports[arm].shipped for arm in HEAD_TO_HEAD)
    head_to_head = next(pair for pair in matrix if (pair.metadata.model, pair.overall.opponent) == (name_a, name_b))
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
        "head_to_head_matrix": [asdict(pair) for pair in matrix],
        "results": rows,
    }
    write_json(out_dir / REPORT_FILENAME, payload)
    overall = head_to_head.overall
    print(
        f"Head-to-head {name_a} vs {name_b}: win rate {overall.win_rate:.1%} "
        f"(95% CI {overall.ci_low:.1%} to {overall.ci_high:.1%}), W/L/D {overall.wins}/{overall.losses}/{overall.draws}"
    )
    for pair in matrix:
        print(f"  {pair.metadata.model} vs {pair.overall.opponent}: win rate {pair.overall.win_rate:.1%}")
    print(f"Saved: {out_dir / REPORT_FILENAME}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation plan: 4 arms, 3 seeds, Report-seed evaluation")
    add_plan_arguments(parser, timesteps=ABLATION_TIMESTEPS, out_dir=ABLATION_DIR)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_ablations(**plan_options(parser.parse_args()))


if __name__ == "__main__":
    main()
