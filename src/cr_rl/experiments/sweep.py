"""Hyperparameter sweep — results table for the thesis experiments chapter.

Each configuration trains into its own directory experiments/runs/<name>/,
then evaluation vs LogicAgent. Output: results.json + a console table.

Usage:
  python -m experiments.sweep --timesteps 1000000 --configs baseline low_lr
  python -m experiments.sweep --timesteps 20000 --n-envs 2 --eval-episodes 20  # smoke
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cr_rl.paths import EXPERIMENT_RUNS_DIR
from cr_rl.training.train import evaluate_model, train

RUNS_DIR = EXPERIMENT_RUNS_DIR

CONFIGS: dict[str, dict[str, Any]] = {
    "baseline": {},
    "low_lr": {"learning_rate": 1e-4},
    "high_lr": {"learning_rate": 1e-3},
    "high_entropy": {"ent_coef": 0.05},
    "low_entropy": {"ent_coef": 0.005},
    "gamma_99": {"gamma": 0.99},
    "big_net": {"net_arch": (512, 512)},
    "small_net": {"net_arch": (128, 128)},
}


def run_sweep(
    *,
    timesteps: int,
    n_envs: int,
    eval_episodes: int,
    configs: list[str],
    seed: int,
) -> list[dict]:
    results: list[dict] = []
    for name in configs:
        overrides = CONFIGS.get(name)
        if overrides is None:
            print(f"Skipped unknown configuration: {name}")
            continue

        run_dir = RUNS_DIR / name
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "experiment": "hyperparameter_sweep",
            "config": name,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "timesteps": timesteps,
            "n_envs": n_envs,
            "eval_episodes": eval_episodes,
            "overrides": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in overrides.items()
            },
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\n=== [{name}] training {timesteps:,} steps → {run_dir} ===")
        train(
            timesteps=timesteps,
            n_envs=n_envs,
            seed=seed,
            model_dir=run_dir,
            **overrides,
        )

        best = run_dir / "best_model.zip"
        final = run_dir / "ppo_cr_final.zip"
        model_path = best if best.is_file() else final
        print(f"=== [{name}] evaluation ({model_path.name}, {eval_episodes} matches) ===")
        stats = evaluate_model(model_path, n_episodes=eval_episodes, seed=seed + 500)

        results.append(
            {
                "config": name,
                "label": f"Sweep: {name}",
                "experiment": "hyperparameter_sweep",
                "seed": seed,
                "overrides": {k: (list(v) if isinstance(v, tuple) else v) for k, v in overrides.items()},
                "timesteps": timesteps,
                **stats,
            }
        )
        print(f"[{name}] win_rate={stats['win_rate']:.1%}  reward={stats['mean_reward']:+.4f}")

    return results


def print_table(results: list[dict]) -> None:
    print("\n=== Sweep results (vs LogicAgent) ===")
    header = f"{'configuration':<16} {'win_rate':>9} {'W':>4} {'L':>4} {'D':>4} {'reward':>9}"
    print(header)
    print("-" * len(header))
    for row in sorted(results, key=lambda r: -r["win_rate"]):
        print(
            f"{row['config']:<16} {row['win_rate']:>8.1%} "
            f"{row['wins']:>4} {row['losses']:>4} {row['draws']:>4} {row['mean_reward']:>+9.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO hyperparameter sweep")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["baseline", "low_lr", "high_entropy"],
        choices=list(CONFIGS),
    )
    args = parser.parse_args()

    results = run_sweep(
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        eval_episodes=args.eval_episodes,
        configs=args.configs,
        seed=args.seed,
    )
    print_table(results)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = RUNS_DIR / "results.json"
    payload = {
        "experiment": "hyperparameter_sweep",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
