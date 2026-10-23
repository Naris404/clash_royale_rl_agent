"""
Training (MaskablePPO) against a mix of Training opponents.

Phase 1 runs every env against RandomAgent; phase 2 assigns an opponent per env index
(8 LogicAgent, 2 ε-LogicAgent, 2 RandomAgent). `--no-curriculum` starts in phase 2.

Usage:
  cr-rl-train
  cr-rl-train --timesteps 500000 --n-envs 12
  cr-rl-train --no-curriculum
  cr-rl-evaluate --model models/ppo_cr_best.zip
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Sequence

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from cr_rl.env.gym_env import ClashRoyaleEnv
from cr_rl.game.rewards import DEFAULT_GAMMA
from cr_rl.paths import MODELS_DIR
from cr_rl.training.matches import play_matches, summarize_matches
from cr_rl.training.selection import (
    SELECTION_INTERVAL,
    SELECTION_SEEDS,
    SelectionCallback,
    select_checkpoint,
)


DEFAULT_MODEL_DIR = MODELS_DIR
DEFAULT_BEST_PATH = DEFAULT_MODEL_DIR / "ppo_cr_best.zip"

DEFAULT_N_ENVS = 12
DEFAULT_PHASE1_FRACTION = 0.2  # share of the step budget spent against RandomAgent
LOG_INTERVAL = 5_000  # env steps between per-opponent win-rate logs
PHASE2_OPPONENTS = ("logic",) * 8 + ("logic-eps",) * 2 + ("random",) * 2


def phase_opponents(phase: int, n_envs: int) -> list[str]:
    """Opponent of each env index: phase 1 is all RandomAgent, phase 2 cycles the mix."""
    if phase == 1:
        return ["random"] * n_envs
    return [PHASE2_OPPONENTS[i % len(PHASE2_OPPONENTS)] for i in range(n_envs)]


def make_training_env(
    seed: int | None = None, opponent: str = "logic", gamma: float = DEFAULT_GAMMA
) -> ClashRoyaleEnv:
    return ClashRoyaleEnv(seed=seed, opponent=opponent, gamma=gamma)


def make_vec_training_env(
    seed: int,
    opponents: list[str],
    gamma: float = DEFAULT_GAMMA,
    index_offset: int = 0,
) -> SubprocVecEnv:
    """One subprocess env per opponent; masks come from each env's own `action_masks()`."""

    def _factory(env_index: int, opponent: str):
        def _init():
            return ClashRoyaleEnv(seed=seed, env_index=env_index, opponent=opponent, gamma=gamma)

        return _init

    return SubprocVecEnv([_factory(index_offset + i, name) for i, name in enumerate(opponents)])


class WinRateCallback(BaseCallback):
    """Logs P0's win % per Training opponent type over its last `window` decisive matches."""

    def __init__(self, window: int = 200, verbose: int = 0):
        super().__init__(verbose)
        self.window = window
        self._outcomes: dict[str, deque[int]] = defaultdict(lambda: deque(maxlen=window))
        self._last_logged = 0

    def _on_step(self) -> bool:
        for done, info in zip(self.locals.get("dones", []), self.locals.get("infos", [])):
            if done and info.get("winner") is not None:
                self._outcomes[info["opponent"]].append(1 if info["winner"] == 0 else 0)
        # Env steps advance num_timesteps by n_envs, so test the interval, not a multiple.
        if self.num_timesteps - self._last_logged >= LOG_INTERVAL:
            self._last_logged = self.num_timesteps
            for opponent, outcomes in self._outcomes.items():
                self.logger.record(f"rollout/win_rate_vs_{opponent}", sum(outcomes) / len(outcomes))
        return True


def train(
    *,
    timesteps: int = 1_000_000,
    n_envs: int = DEFAULT_N_ENVS,
    seed: int = 42,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 256,
    ent_coef: float = 0.02,
    gamma: float = DEFAULT_GAMMA,
    net_arch: tuple[int, ...] = (256, 256),
    curriculum: bool = True,
    phase1_fraction: float = DEFAULT_PHASE1_FRACTION,
    model_dir: Path = DEFAULT_MODEL_DIR,
    resume: str | None = None,
    selection_interval: int = SELECTION_INTERVAL,
    selection_seeds: Sequence[int] = SELECTION_SEEDS,
) -> MaskablePPO:
    model_dir.mkdir(parents=True, exist_ok=True)
    # Seed the libraries here, not through MaskablePPO(seed=...), which would also pin each
    # env's first episode to seed + index instead of its derived episode seed.
    set_random_seed(seed)

    phase1_steps = int(timesteps * phase1_fraction) if curriculum else 0
    phases = ([(1, phase1_steps)] if curriculum else []) + [(2, timesteps - phase1_steps)]
    policy_kwargs = dict(net_arch=dict(pi=list(net_arch), vf=list(net_arch)))

    selection = SelectionCallback(model_dir, interval=selection_interval, seeds=selection_seeds)
    callbacks = [selection, WinRateCallback()]

    model = None
    for phase, steps in phases:
        # Env indices are unique across phases, so no episode seed is replayed.
        train_env = make_vec_training_env(
            seed, phase_opponents(phase, n_envs), gamma=gamma, index_offset=(phase - 1) * n_envs
        )
        if model is not None:
            model.set_env(train_env)
        elif resume:
            model = MaskablePPO.load(resume, env=train_env)
        else:
            model = MaskablePPO(
                "MlpPolicy",
                train_env,
                learning_rate=learning_rate,
                n_steps=max(n_steps // n_envs, 256),
                batch_size=batch_size,
                n_epochs=10,
                gamma=gamma,
                gae_lambda=0.95,
                ent_coef=ent_coef,
                clip_range=0.2,
                verbose=1,
                policy_kwargs=policy_kwargs,
                tensorboard_log=str(model_dir / "tb"),
            )
        print(f"=== Phase {phase}: {dict(Counter(phase_opponents(phase, n_envs)))} ({steps:,} steps) ===")
        model.learn(
            total_timesteps=steps,
            reset_num_timesteps=False,
            callback=callbacks,
            progress_bar=True,
        )
        train_env.close()

    final_path = model_dir / "ppo_cr_final.zip"
    model.save(str(final_path))
    print(f"Saved final model: {final_path}")

    selection.evaluate()  # the final model takes part in the selection too
    selected = select_checkpoint(selection.results)
    selected_path = model_dir / "checkpoints" / selected.path
    shutil.copy2(selected_path, model_dir / "best_model.zip")
    print(f"Selected checkpoint: {selected.path} (win rate {selected.win_rate:.1%})")
    if model_dir == DEFAULT_MODEL_DIR:
        # only the main training updates ppo_cr_best.zip — sweeps have their own directories
        shutil.copy2(selected_path, DEFAULT_BEST_PATH)
        print(f"Best model: {DEFAULT_BEST_PATH}")

    return model


def evaluate_model(
    model_path: str | Path,
    *,
    n_episodes: int = 100,
    deterministic: bool = True,
    seed: int = 0,
) -> dict[str, float]:
    model = MaskablePPO.load(str(model_path))
    seeds = range(seed, seed + n_episodes)
    return summarize_matches(play_matches(model, "logic", seeds, deterministic=deterministic))


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO training against a mix of opponents")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=DEFAULT_N_ENVS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Skip phase 1 (RandomAgent) and start in phase 2",
    )
    parser.add_argument("--phase1-fraction", type=float, default=DEFAULT_PHASE1_FRACTION)
    parser.add_argument("--resume", type=str, default=None, help="Path to a .zip to resume training")
    parser.add_argument(
        "--eval-only",
        type=str,
        default=None,
        metavar="MODEL.zip",
        help="Evaluate the trained model only",
    )
    parser.add_argument("--eval-episodes", type=int, default=100)
    args = parser.parse_args()

    if args.eval_only:
        stats = evaluate_model(args.eval_only, n_episodes=args.eval_episodes)
        print(
            f"Result vs LogicAgent ({stats['episodes']} matches): "
            f"wins={stats['wins']} losses={stats['losses']} draws={stats['draws']} "
            f"win_rate={stats['win_rate']:.1%} "
            f"mean_reward={stats['mean_reward']:.4f}"
        )
        return

    train(
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        learning_rate=args.lr,
        curriculum=not args.no_curriculum,
        phase1_fraction=args.phase1_fraction,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
