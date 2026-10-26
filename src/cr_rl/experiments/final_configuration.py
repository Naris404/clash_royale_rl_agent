"""Final-configuration plan: 3 seeds x 1M steps, Report-seed evaluation, shipped run, budget rule.

A shipped run that passes the Ship gate becomes the model the web game loads. Each run's selected
checkpoint also plays its own final model head-to-head, which shows what the steps after the selected
checkpoint (the self-play phase, when selection picks its start) did beyond the win rate vs LogicAgent.

Usage:
  cr-rl-final
  cr-rl-final --timesteps 3000000   # after the budget rule says the runs are still learning
"""

from __future__ import annotations

import argparse
import logging
import shutil
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sb3_contrib import MaskablePPO

from cr_rl.coach.inference import DEFAULT_MODEL_PATH
from cr_rl.experiments.arms import TRAINING_SEEDS, add_plan_arguments, plan_options, run_arm, run_name
from cr_rl.experiments.curriculum import EXTENDED_TIMESTEPS, budget_rule
from cr_rl.paths import EXPERIMENT_RUNS_DIR
from cr_rl.training.head_to_head import HeadToHead, evaluate_head_to_head
from cr_rl.training.phases import FULL
from cr_rl.training.report import REPORT_FILENAME, REPORT_SEEDS, save_configuration_report, selected_checkpoint
from cr_rl.training.selection import SELECTION_FILENAME, load_selection

FINAL_DIR = EXPERIMENT_RUNS_DIR / "final_configuration"
FINAL_TIMESTEPS = 1_000_000


def ship_checkpoint(checkpoint: Path, target: Path) -> None:
    """Copy a checkpoint to `target`, the model the web game and the Coach load."""
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoint, target)


def ship_run(run_dir: Path, target: Path) -> None:
    """Copy the run's selected checkpoint to `target`."""
    ship_checkpoint(selected_checkpoint(run_dir)[1], target)


def selected_vs_final(run_dir: Path, *, seeds: range) -> HeadToHead:
    """The run's selected checkpoint against the model it ended training with."""
    models = [MaskablePPO.load(str(path)) for path in (selected_checkpoint(run_dir)[1], run_dir / "ppo_cr_final.zip")]
    return evaluate_head_to_head(
        *models, name_a=f"{run_dir.name} selected", name_b=f"{run_dir.name} final", seeds=seeds
    )


def run_final_configuration(
    *,
    timesteps: int = FINAL_TIMESTEPS,
    seeds: tuple[int, ...] = TRAINING_SEEDS,
    out_dir: Path = FINAL_DIR,
    report_seeds: range = REPORT_SEEDS,
    ship_to: Path = DEFAULT_MODEL_PATH,
    **train_kwargs: Any,
) -> None:
    """Train and evaluate the full configuration; results land in `out_dir/results.json`.

    The shipped run is copied to `ship_to`, the web game's model, only if it passes the Ship gate.
    """
    report, metrics = run_arm(
        FULL, seeds=seeds, timesteps=timesteps, out_dir=out_dir, report_seeds=report_seeds, **train_kwargs
    )
    first_run = out_dir / run_name(FULL, seeds[0])
    budget = budget_rule(load_selection(first_run / SELECTION_FILENAME), timesteps=timesteps)
    against_final = [selected_vs_final(out_dir / run.name, seeds=report_seeds) for run in report.runs]
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
            "selected_vs_final": [asdict(pair) for pair in against_final],
        },
    )
    print(f"Shipped run: {report.shipped} | Ship gate: {'PASS' if report.ship_gate_passed else 'FAIL'}")
    if report.ship_gate_passed:
        ship_run(out_dir / report.shipped, ship_to)
        print(f"Shipped as the RL Opponent model: {ship_to}")
    else:
        print(f"Not shipped; the RL Opponent model is unchanged: {ship_to}")
    print(
        f"Budget rule ({first_run.name}): selection win rate rose {budget.rise:+.1%} over the last 20% -> "
        + (f"move all final runs to {EXTENDED_TIMESTEPS // 1_000_000}M steps" if budget.extend else "keep the budget")
    )
    for pair in against_final:
        overall = pair.overall
        print(
            f"{pair.metadata.model} vs {overall.opponent}: win rate {overall.win_rate:.1%} "
            f"(95% CI {overall.ci_low:.1%} to {overall.ci_high:.1%})"
        )
    print(f"Saved: {out_dir / REPORT_FILENAME}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Final-configuration plan: 3 seeds, Report-seed evaluation")
    add_plan_arguments(parser, timesteps=FINAL_TIMESTEPS, out_dir=FINAL_DIR)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_final_configuration(**plan_options(parser.parse_args()))


if __name__ == "__main__":
    main()
