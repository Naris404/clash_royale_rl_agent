"""Final-configuration plan: 3 seeds x 1M steps, Report-seed evaluation, shipped run, budget rule.

Usage:
  cr-rl-final
  cr-rl-final --timesteps 3000000   # after the budget rule says the runs are still learning
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cr_rl.experiments.arms import TRAINING_SEEDS, add_plan_arguments, plan_options, run_arm, run_name
from cr_rl.experiments.curriculum import EXTENDED_TIMESTEPS, budget_rule
from cr_rl.paths import EXPERIMENT_RUNS_DIR
from cr_rl.training.phases import FULL
from cr_rl.training.report import REPORT_FILENAME, REPORT_SEEDS, save_configuration_report
from cr_rl.training.selection import SELECTION_FILENAME, load_selection

FINAL_DIR = EXPERIMENT_RUNS_DIR / "final_configuration"
FINAL_TIMESTEPS = 1_000_000


def run_final_configuration(
    *,
    timesteps: int = FINAL_TIMESTEPS,
    seeds: tuple[int, ...] = TRAINING_SEEDS,
    out_dir: Path = FINAL_DIR,
    report_seeds: range = REPORT_SEEDS,
    **train_kwargs: Any,
) -> None:
    """Train and evaluate the full configuration; results land in `out_dir/results.json`."""
    report, metrics = run_arm(
        FULL, seeds=seeds, timesteps=timesteps, out_dir=out_dir, report_seeds=report_seeds, **train_kwargs
    )
    first_run = out_dir / run_name(FULL, seeds[0])
    budget = budget_rule(load_selection(first_run / SELECTION_FILENAME), timesteps=timesteps)
    save_configuration_report(
        report,
        out_dir / REPORT_FILENAME,
        experiment="final_configuration",
        extra={
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "timesteps": timesteps,
            "training_seeds": list(seeds),
            "curriculum": {name: asdict(run_metrics) for name, run_metrics in metrics.items()},
            "budget_rule": asdict(budget),
        },
    )
    print(f"Shipped run: {report.shipped} | Ship gate: {'PASS' if report.ship_gate_passed else 'FAIL'}")
    print(
        f"Budget rule ({first_run.name}): selection win rate rose {budget.rise:+.1%} over the last 20% -> "
        + (f"move all final runs to {EXTENDED_TIMESTEPS // 1_000_000}M steps" if budget.extend else "keep the budget")
    )
    print(f"Saved: {out_dir / REPORT_FILENAME}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Final-configuration plan: 3 seeds, Report-seed evaluation")
    add_plan_arguments(parser, timesteps=FINAL_TIMESTEPS, out_dir=FINAL_DIR)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_final_configuration(**plan_options(parser.parse_args()))


if __name__ == "__main__":
    main()
