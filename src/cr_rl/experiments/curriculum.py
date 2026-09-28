"""Curriculum learning: faza 1 vs RandomAgent, faza 2 vs LogicAgent.

Hipoteza: łatwy przeciwnik na starcie uczy podstaw (zagrywanie kart, eliksir),
trudniejszy docenia taktykę. Porównanie z baseline w rozdziale eksperymentów.

Uruchomienie:
  python -m experiments.curriculum --phase1 300000 --phase2 700000
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback

from cr_rl.paths import EXPERIMENT_RUNS_DIR
from cr_rl.training.train import DEFAULT_MODEL_DIR, evaluate_model, make_training_env, make_vec_training_env

RUN_DIR = EXPERIMENT_RUNS_DIR / "curriculum"


def train_curriculum(
    *,
    phase1_timesteps: int,
    phase2_timesteps: int,
    n_envs: int,
    seed: int,
) -> MaskablePPO:
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    print(f"=== Faza 1: vs RandomAgent ({phase1_timesteps:,} kroków) ===")
    env_random = make_vec_training_env(n_envs, seed, opponent="random")
    model = MaskablePPO(
        "MlpPolicy",
        env_random,
        learning_rate=3e-4,
        n_steps=max(2048 // n_envs, 256),
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        ent_coef=0.02,
        verbose=1,
        seed=seed,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        tensorboard_log=str(RUN_DIR / "tb"),
    )
    model.learn(total_timesteps=phase1_timesteps, progress_bar=True)
    env_random.close()

    print(f"=== Faza 2: vs LogicAgent ({phase2_timesteps:,} kroków) ===")
    env_logic = make_vec_training_env(n_envs, seed + 1000, opponent="logic")
    eval_env = make_training_env(seed + 10_000, opponent="logic")
    model.set_env(env_logic)
    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(RUN_DIR),
        log_path=str(RUN_DIR / "eval"),
        eval_freq=max(25_000 // n_envs, 1),
        n_eval_episodes=30,
        deterministic=True,
    )
    model.learn(
        total_timesteps=phase2_timesteps,
        reset_num_timesteps=False,
        callback=eval_callback,
        progress_bar=True,
    )
    env_logic.close()
    eval_env.close()

    final_path = RUN_DIR / "ppo_cr_curriculum_final.zip"
    model.save(str(final_path))
    print(f"Zapisano: {final_path}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Curriculum: random → logic")
    parser.add_argument("--phase1", type=int, default=300_000)
    parser.add_argument("--phase2", type=int, default=700_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-episodes", type=int, default=100)
    args = parser.parse_args()

    train_curriculum(
        phase1_timesteps=args.phase1,
        phase2_timesteps=args.phase2,
        n_envs=args.n_envs,
        seed=args.seed,
    )

    evaluations = []
    for name in ("best_model.zip", "ppo_cr_curriculum_final.zip"):
        path = RUN_DIR / name
        if path.is_file():
            stats = evaluate_model(path, n_episodes=args.eval_episodes, seed=args.seed + 500)
            evaluations.append((path, stats))
            print(
                f"[curriculum/{name}] win_rate={stats['win_rate']:.1%} "
                f"(W:{stats['wins']} L:{stats['losses']} D:{stats['draws']})"
            )
    if evaluations:
        model_path, stats = max(evaluations, key=lambda item: item[1]["win_rate"])
        payload = {
            "experiment": "curriculum",
            "label": "Curriculum Random → Logic",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed": args.seed,
            "phase1_timesteps": args.phase1,
            "phase2_timesteps": args.phase2,
            "n_envs": args.n_envs,
            "model": str(model_path),
            **stats,
        }
        (RUN_DIR / "results.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
