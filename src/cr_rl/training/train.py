"""
Training (MaskablePPO) against a mix of Training opponents.

A run goes through the Training phases of a layout (see `cr_rl.training.phases`): warm-up
against RandomAgent, the scripted mix, then the self-play mix with a Past-self pool
(see `cr_rl.training.self_play`). `--layout` selects the full, no-curriculum or
no-self-play arrangement.

Usage:
  cr-rl-train
  cr-rl-train --timesteps 500000 --n-envs 12
  cr-rl-train --layout no-curriculum
  cr-rl-evaluate --model models/ppo_cr_best.zip
"""

from __future__ import annotations

import argparse
import logging
import shutil
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Sequence

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from cr_rl.env.gym_env import ClashRoyaleEnv
from cr_rl.game.rewards import DEFAULT_GAMMA, RewardWeights
from cr_rl.paths import MODELS_DIR
from cr_rl.training.matches import play_matches, summarize_matches
from cr_rl.training.phases import FULL, PAST_SELF, PHASE_LAYOUTS, phase_opponents, phase_steps
from cr_rl.training.self_play import (
    ADMISSION_MATCHES,
    POOL_LOG_FILENAME,
    POOL_UPDATE_INTERVAL,
    PastSelfPool,
    load_past_self,
)
from cr_rl.training.selection import (
    SELECTION_INTERVAL,
    SELECTION_SEEDS,
    SelectionCallback,
    select_checkpoint,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = MODELS_DIR
DEFAULT_BEST_PATH = DEFAULT_MODEL_DIR / "ppo_cr_best.zip"

DEFAULT_N_ENVS = 12
LOG_INTERVAL = 5_000  # env steps between per-opponent win-rate logs


def make_vec_training_env(
    seed: int,
    opponents: list[str | Path],
    gamma: float = DEFAULT_GAMMA,
    index_offset: int = 0,
    reward_weights: RewardWeights = RewardWeights(),
) -> SubprocVecEnv:
    """One subprocess env per opponent; masks come from each env's own `action_masks()`.

    An opponent is a bot name or the checkpoint file of a Past self, loaded in the subprocess.
    """

    def _factory(env_index: int, opponent: str | Path):
        def _init():
            policy = load_past_self(opponent) if isinstance(opponent, Path) else opponent
            return ClashRoyaleEnv(
                seed=seed, env_index=env_index, opponent=policy, gamma=gamma, reward_weights=reward_weights
            )

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
    layout: str = FULL,
    reward_weights: RewardWeights = RewardWeights(),
    model_dir: Path = DEFAULT_MODEL_DIR,
    resume: str | None = None,
    selection_interval: int = SELECTION_INTERVAL,
    selection_seeds: Sequence[int] = SELECTION_SEEDS,
    pool_interval: int = POOL_UPDATE_INTERVAL,
    pool_matches: int = ADMISSION_MATCHES,
) -> MaskablePPO:
    model_dir.mkdir(parents=True, exist_ok=True)
    # Seed the libraries here, not through MaskablePPO(seed=...), which would also pin each
    # env's first episode to seed + index instead of its derived episode seed.
    set_random_seed(seed)

    policy_kwargs = dict(net_arch=dict(pi=list(net_arch), vf=list(net_arch)))

    selection = SelectionCallback(model_dir, interval=selection_interval, seeds=selection_seeds)
    callbacks = [selection, WinRateCallback()]

    model = None
    env_offset = 0  # env indices are unique across the run, so no episode seed is replayed
    for phase, steps in phase_steps(layout, timesteps):
        pool = None
        if phase.uses_pool:
            selection.evaluate()  # the model so far joins the checkpoints the pool is seeded from
            pool = PastSelfPool.from_selection(
                selection.results, model_dir / "checkpoints", run_seed=seed, matches=pool_matches
            )
        remaining = steps
        while remaining > 0:
            # The envs are rebuilt after every pool check, so a chunk ends at each check.
            chunk = min(remaining, pool_interval) if pool else remaining
            opponents = phase_opponents(phase, n_envs, pool.members if pool else ())
            train_env = make_vec_training_env(
                seed, opponents, gamma=gamma, index_offset=env_offset, reward_weights=reward_weights
            )
            env_offset += n_envs
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
            mix = Counter(PAST_SELF if isinstance(o, Path) else o for o in opponents)
            logger.info("=== %s: %s (%s steps) ===", phase.name, dict(mix), f"{chunk:,}")
            model.learn(
                total_timesteps=chunk,
                reset_num_timesteps=False,
                callback=callbacks,
                progress_bar=True,
            )
            train_env.close()
            remaining -= chunk
            if pool and remaining > 0:
                _check_pool(pool, model, model_dir)

    final_path = model_dir / "ppo_cr_final.zip"
    model.save(str(final_path))
    logger.info("Saved final model: %s", final_path)

    selection.evaluate()  # the final model takes part in the selection too
    selected = select_checkpoint(selection.results)
    selected_path = model_dir / "checkpoints" / selected.path
    shutil.copy2(selected_path, model_dir / "best_model.zip")
    logger.info("Selected checkpoint: %s (win rate %.1f%%)", selected.path, selected.win_rate * 100)
    if model_dir == DEFAULT_MODEL_DIR:
        # only the main training updates ppo_cr_best.zip — sweeps have their own directories
        shutil.copy2(selected_path, DEFAULT_BEST_PATH)
        logger.info("Best model: %s", DEFAULT_BEST_PATH)

    return model


def _check_pool(pool: PastSelfPool, model: MaskablePPO, model_dir: Path) -> None:
    """Admission check of the current model; every decision is logged to `pool.json`."""
    step = model.num_timesteps
    candidate = model_dir / "pool" / f"ppo_cr_{step}_steps.zip"
    candidate.parent.mkdir(exist_ok=True)
    model.save(str(candidate))
    decision = pool.consider(candidate, step)
    pool.save_log(model_dir / POOL_LOG_FILENAME)
    verdict = f"joins, replacing {decision.replaced}" if decision.joined else "stays out"
    logger.info(
        "Pool check @ %s steps: %d/%d wins, candidate %s",
        f"{step:,}", decision.wins, decision.matches, verdict,
    )


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
        "--layout",
        choices=sorted(PHASE_LAYOUTS),
        default=FULL,
        help="Training phases: full, no-curriculum (no warm-up) or no-self-play",
    )
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
    logging.basicConfig(level=logging.INFO, format="%(message)s")

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
        layout=args.layout,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
