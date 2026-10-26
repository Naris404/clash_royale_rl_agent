"""v3 plan: fine-tune each v2 Final model for two generations of Exploiter + self-play (ADR-0007).

Per seed, starting from that seed's v2 Final model, each generation
  1. trains an Exploiter against the current main agent,
  2. adds it to the Past-self pool as a permanent member,
  3. fine-tunes the main agent in the v3 mix.
The generation-2 checkpoints are then scored against a reference pool (LogicAgent, the v2 Shipped
model and the three v2 Final models) and each run's best one is its Selected model. The Selected
models play the Report seeds; the run with the best reference score is judged against the target
(about 90% vs LogicAgent and a head-to-head win over the v2 Shipped model) and, if it meets it,
becomes the Shipped model. The v2 Shipped model is first copied to its own file, where the Model
catalogue keeps it. The Arrows experiment runs on the v3 Selected model.

Usage:
  cr-rl-v3
  cr-rl-v3 --smoke --out some/scratch/dir   # the whole pipeline in minutes, on a copy of the models
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from sb3_contrib import MaskablePPO

from cr_rl.coach.inference import DEFAULT_MODEL_PATH
from cr_rl.experiments.arms import TRAINING_SEEDS, add_plan_arguments, plan_options
from cr_rl.experiments.arrows import run_arrows_experiment, save_arrows_results
from cr_rl.experiments.exploiter import EXPLOITER_FILENAME, EXPLOITER_LOG_FILENAME, MAX_STEPS, train_exploiter
from cr_rl.experiments.final_configuration import FINAL_DIR
from cr_rl.experiments.fine_tune import fine_tune
from cr_rl.jsonio import write_json
from cr_rl.paths import EXPERIMENT_RUNS_DIR, MODELS_DIR
from cr_rl.training.head_to_head import HeadToHead, evaluate_head_to_head
from cr_rl.training.reference_selection import (
    LOGIC_OPPONENT,
    REFERENCE_FILENAME,
    Opponent,
    ReferenceScore,
    save_scores,
    score_checkpoints,
    select_reference,
)
from cr_rl.training.report import (
    MANIFEST_FILENAME,
    REPORT_FILENAME,
    REPORT_SEEDS,
    RunReport,
    evaluate_report,
    save_configuration_report,
    summarize_configuration,
)
from cr_rl.training.seeds import SELECTION_SEEDS
from cr_rl.training.selection import SELECTION_FILENAME, load_selection

logger = logging.getLogger(__name__)

V3_DIR = EXPERIMENT_RUNS_DIR / "v3"
V3_TIMESTEPS = 500_000  # one generation's fine-tune
GENERATIONS = 2
SELECTION_INTERVAL = 100_000  # steps between the checkpoints a generation saves
TARGET_LOGIC_WIN_RATE = 0.9  # "about 90%": the point estimate on the Report seeds
V2_SHIPPED = "v2-shipped"  # the v2 Shipped model's name as a reference opponent and head-to-head opponent
V2_SHIPPED_FILENAME = "ppo_cr_v2_shipped.zip"  # where the Model catalogue keeps it
SHIPPED_FILENAME = DEFAULT_MODEL_PATH.name
ARROWS_FILENAME = "arrows.json"
DEFAULT_WORKERS = 6  # processes scoring checkpoints

# `cr-rl-v3 --smoke`: every phase of the plan with tiny step counts and few matches.
SMOKE_OPTIONS: dict[str, Any] = dict(
    timesteps=2_048,
    seeds=(0,),
    n_envs=4,
    exploiter_steps=1_024,
    selection_interval=1_024,
    selection_seeds=range(SELECTION_SEEDS.start, SELECTION_SEEDS.start + 2),
    report_seeds=range(REPORT_SEEDS.start, REPORT_SEEDS.start + 4),
    pool_interval=1_024,
    pool_matches=4,
    workers=2,
)


@dataclass(frozen=True)
class Target:
    """The ship decision: the best run's Report-seed results against the v3 target."""

    run: str
    logic_win_rate: float
    head_to_head_win_rate: float  # against the v2 Shipped model, both sides together
    logic_threshold: float
    met: bool


def exploiter_dir(out_dir: Path, generation: int, seed: int) -> Path:
    return out_dir / f"exploiter{generation}-seed{seed}"


def generation_dir(out_dir: Path, generation: int, seed: int) -> Path:
    return out_dir / f"gen{generation}-seed{seed}"


def train_seed(
    v2_final: Path,
    *,
    seed: int,
    timesteps: int,
    out_dir: Path,
    n_envs: int,
    exploiter_steps: int,
    selection_interval: int,
    **train_kwargs: Any,
) -> Path:
    """Both generations of one seed; returns the last generation's run directory."""
    source, exploiters = v2_final, []
    for generation in range(1, GENERATIONS + 1):
        generation_seed = seed + 1_000 * (generation - 1)
        run_dir = exploiter_dir(out_dir, generation, seed)
        train_exploiter(source, seed=generation_seed, run_dir=run_dir, n_envs=n_envs, max_steps=exploiter_steps)
        exploiters.append(run_dir / EXPLOITER_FILENAME)
        run_dir = generation_dir(out_dir, generation, seed)
        fine_tune(
            source,
            seed=generation_seed,
            timesteps=timesteps,
            run_dir=run_dir,
            permanent_members=list(exploiters),
            n_envs=n_envs,
            selection_interval=selection_interval,
            **train_kwargs,
        )
        source = run_dir / "ppo_cr_final.zip"
    return run_dir


def keep_v2_shipped(shipped_path: Path, v2_shipped_path: Path) -> None:
    """Copy the current Shipped model to the file the Model catalogue keeps the v2 Shipped model in."""
    if not v2_shipped_path.is_file():
        v2_shipped_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(shipped_path, v2_shipped_path)


def reference_pool(v2_dir: Path, v2_shipped_path: Path) -> dict[str, Opponent]:
    """LogicAgent, the v2 Shipped model and the v2 Final models, by name."""
    pool: dict[str, Opponent] = {LOGIC_OPPONENT: LOGIC_OPPONENT, V2_SHIPPED: v2_shipped_path}
    for seed in TRAINING_SEEDS:
        pool[f"v2-final-seed{seed}"] = v2_dir / f"full-seed{seed}" / "ppo_cr_final.zip"
    return pool


def select_generation_checkpoint(
    run_dir: Path, opponents: dict[str, Opponent], *, seeds: Sequence[int], workers: int
) -> ReferenceScore:
    """The run's Selected model: the best reference-pool score among the checkpoints the run itself trained.

    The first checkpoint of the run is the model it started from, the previous generation's, so it is left out.
    """
    started_at = min(result.step for result in load_selection(run_dir / SELECTION_FILENAME))
    scores = score_checkpoints(run_dir, opponents, seeds, min_step=started_at + 1, workers=workers)
    save_scores(scores, run_dir / REFERENCE_FILENAME)
    selected = select_reference(scores)
    logger.info("%s selected: %s (average %.1f%%)", run_dir.name, selected.path, selected.average * 100)
    return selected


def evaluate_selected(run_dir: Path, selected: ReferenceScore, *, report_seeds: range) -> RunReport:
    """The Selected model on the Report seeds against LogicAgent and RandomAgent."""
    checkpoint = run_dir / "checkpoints" / selected.path
    evaluation = evaluate_report(
        MaskablePPO.load(str(checkpoint), device="cpu"),
        model=str(checkpoint),
        config=json.loads((run_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")),
        seeds=report_seeds,
    )
    selection = next(result for result in load_selection(run_dir / SELECTION_FILENAME) if result.path == selected.path)
    return RunReport(run_dir.name, selection, evaluation)


def check_target(run: RunReport, head_to_head: HeadToHead) -> Target:
    logic = run.evaluation.against(LOGIC_OPPONENT).win_rate
    versus_v2 = head_to_head.overall.win_rate
    return Target(run.name, logic, versus_v2, TARGET_LOGIC_WIN_RATE, logic >= TARGET_LOGIC_WIN_RATE and versus_v2 > 0.5)


def ship_model(checkpoint: Path, target: Path) -> None:
    """Copy the Selected checkpoint to `target`, the model the web game and the Coach load."""
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoint, target)


def run_v3(
    *,
    timesteps: int = V3_TIMESTEPS,
    seeds: Sequence[int] = TRAINING_SEEDS,
    out_dir: Path = V3_DIR,
    v2_dir: Path = FINAL_DIR,
    shipped_path: Path = DEFAULT_MODEL_PATH,
    v2_shipped_path: Path = MODELS_DIR / V2_SHIPPED_FILENAME,
    report_seeds: range = REPORT_SEEDS,
    selection_seeds: Sequence[int] = SELECTION_SEEDS,
    exploiter_steps: int = MAX_STEPS,
    selection_interval: int = SELECTION_INTERVAL,
    n_envs: int,
    workers: int = DEFAULT_WORKERS,
    **train_kwargs: Any,
) -> None:
    """Train, select and evaluate the v3 models; results land in `out_dir/results.json`.

    `shipped_path` is the Shipped model, overwritten only if v3 meets the target; `v2_shipped_path`
    is where the Model catalogue keeps the v2 Shipped model. Training that already finished is kept.
    """
    keep_v2_shipped(shipped_path, v2_shipped_path)
    opponents = reference_pool(v2_dir, v2_shipped_path)
    run_dirs = [
        train_seed(
            v2_dir / f"full-seed{seed}" / "ppo_cr_final.zip",
            seed=seed,
            timesteps=timesteps,
            out_dir=out_dir,
            n_envs=n_envs,
            exploiter_steps=exploiter_steps,
            selection_interval=selection_interval,
            selection_seeds=selection_seeds,
            **train_kwargs,
        )
        for seed in seeds
    ]

    selected = {run_dir: select_generation_checkpoint(run_dir, opponents, seeds=selection_seeds, workers=workers) for run_dir in run_dirs}
    runs = [evaluate_selected(run_dir, score, report_seeds=report_seeds) for run_dir, score in selected.items()]
    best_dir = max(selected, key=lambda run_dir: selected[run_dir].average)
    report = summarize_configuration(runs)
    best = next(run for run in runs if run.name == best_dir.name)
    report = replace(report, shipped=best.name, ship_gate_passed=best.evaluation.passes_ship_gate)

    v2_shipped = MaskablePPO.load(str(v2_shipped_path), device="cpu")
    checkpoints = {run_dir.name: run_dir / "checkpoints" / score.path for run_dir, score in selected.items()}
    matrix = [
        evaluate_head_to_head(
            MaskablePPO.load(str(checkpoints[run.name]), device="cpu"), v2_shipped,
            name_a=run.name, name_b=V2_SHIPPED, seeds=report_seeds,
        )
        for run in runs
    ]
    head_to_head = next(pair for pair in matrix if pair.metadata.model == best.name)
    target = check_target(best, head_to_head)

    arrows = run_arrows_experiment(
        MaskablePPO.load(str(checkpoints[best.name]), device="cpu"), v2_shipped,
        seeds=report_seeds, model_name=f"v3 {best.name}", frozen_name=V2_SHIPPED,
    )
    save_arrows_results(arrows, out_dir / ARROWS_FILENAME)

    save_configuration_report(
        report,
        out_dir / REPORT_FILENAME,
        experiment="v3",
        extra={
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "timesteps": timesteps,
            "generations": GENERATIONS,
            "training_seeds": list(seeds),
            "arms": {"v3": asdict(report)},
            "head_to_head": asdict(head_to_head),
            "head_to_head_matrix": [asdict(pair) for pair in matrix],
            "target": asdict(target),
            "reference_pool": list(opponents),
            "exploiters": [
                {"run": path.parent.name, **json.loads(path.read_text(encoding="utf-8"))}
                for path in sorted(out_dir.glob(f"exploiter*-seed*/{EXPLOITER_LOG_FILENAME}"))
            ],
        },
    )
    print(
        f"Best run {target.run}: {target.logic_win_rate:.1%} vs LogicAgent (target {TARGET_LOGIC_WIN_RATE:.0%}), "
        f"{target.head_to_head_win_rate:.1%} vs the v2 Shipped model -> target {'MET' if target.met else 'NOT met'}"
    )
    if target.met:
        ship_model(checkpoints[best.name], shipped_path)
        print(f"Shipped as the RL Opponent model: {shipped_path} (v2 Shipped model kept at {v2_shipped_path})")
    else:
        print(f"Not shipped; the RL Opponent model is unchanged: {shipped_path}")
    print(f"Saved: {out_dir / REPORT_FILENAME}")


def main() -> None:
    parser = argparse.ArgumentParser(description="v3 plan: two generations of Exploiter + self-play fine-tuning per seed")
    add_plan_arguments(parser, timesteps=V3_TIMESTEPS, out_dir=V3_DIR)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(TRAINING_SEEDS))
    parser.add_argument("--exploiter-steps", type=int, default=MAX_STEPS, help="Step limit of each Exploiter")
    parser.add_argument("--selection-episodes", type=int, default=len(SELECTION_SEEDS), help="Selection seeds to play")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Processes scoring checkpoints")
    parser.add_argument("--v2-dir", type=Path, default=FINAL_DIR, help="The final-configuration plan holding the v2 Final models")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="Where the Shipped model and the v2 Shipped model live")
    parser.add_argument("--smoke", action="store_true", help="Tiny step counts on a copy of the models in --out; overrides the sizes above")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    options = dict(
        plan_options(args),
        seeds=tuple(args.seeds),
        exploiter_steps=args.exploiter_steps,
        selection_seeds=range(SELECTION_SEEDS.start, SELECTION_SEEDS.start + args.selection_episodes),
        workers=args.workers,
        v2_dir=args.v2_dir,
        shipped_path=args.models_dir / SHIPPED_FILENAME,
        v2_shipped_path=args.models_dir / V2_SHIPPED_FILENAME,
    )
    if args.smoke:
        # The Shipped model is copied into --out, so the smoke run can ship and register without touching the real files.
        models_dir = args.out / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        if not (models_dir / SHIPPED_FILENAME).is_file():
            shutil.copyfile(args.models_dir / SHIPPED_FILENAME, models_dir / SHIPPED_FILENAME)
        options |= SMOKE_OPTIONS | dict(shipped_path=models_dir / SHIPPED_FILENAME, v2_shipped_path=models_dir / V2_SHIPPED_FILENAME)
    run_v3(**options)


if __name__ == "__main__":
    main()
