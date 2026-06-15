"""
Trening PPO (MaskablePPO) — agent RL vs LogicAgent.

Uruchomienie:
  python train.py
  python train.py --timesteps 500000 --n-envs 8
  python evaluate.py --model models/ppo_cr_best.zip
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from gym_env import ClashRoyaleEnv


DEFAULT_MODEL_DIR = Path("models")
DEFAULT_BEST_PATH = DEFAULT_MODEL_DIR / "ppo_cr_best.zip"
DEFAULT_CHECKPOINT_PREFIX = DEFAULT_MODEL_DIR / "ppo_cr_checkpoint"


def mask_fn(env: ClashRoyaleEnv) -> np.ndarray:
    return env.unwrapped.action_masks()


def make_training_env(seed: int | None = None) -> ActionMasker:
    env = ClashRoyaleEnv(seed=seed)
    return ActionMasker(env, mask_fn)


def make_vec_training_env(n_envs: int, seed: int) -> DummyVecEnv:
    def _factory(rank: int):
        def _init():
            env_seed = seed + rank
            return make_training_env(env_seed)

        return _init

    return DummyVecEnv([_factory(i) for i in range(n_envs)])


class WinRateCallback(BaseCallback):
    """Loguje % wygranych P0 w ostatnich meczach (z bufora info env)."""

    def __init__(self, window: int = 200, verbose: int = 0):
        super().__init__(verbose)
        self.window = window
        self._outcomes: list[int] = []

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for done, info in zip(dones, infos):
            if not done:
                continue
            winner = info.get("winner")
            if winner is None:
                continue
            self._outcomes.append(1 if winner == 0 else 0)
            if len(self._outcomes) > self.window:
                self._outcomes.pop(0)
        if self._outcomes and self.num_timesteps % 5000 == 0:
            rate = sum(self._outcomes) / len(self._outcomes)
            self.logger.record("rollout/win_rate_vs_logic", rate)
        return True


def train(
    *,
    timesteps: int = 1_000_000,
    n_envs: int = 8,
    seed: int = 42,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 256,
    ent_coef: float = 0.02,
    model_dir: Path = DEFAULT_MODEL_DIR,
    resume: str | None = None,
) -> MaskablePPO:
    model_dir.mkdir(parents=True, exist_ok=True)

    train_env = make_vec_training_env(n_envs, seed)
    eval_env = make_training_env(seed + 10_000)

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))

    if resume:
        model = MaskablePPO.load(resume, env=train_env)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            train_env,
            learning_rate=learning_rate,
            n_steps=max(n_steps // n_envs, 256),
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=ent_coef,
            clip_range=0.2,
            verbose=1,
            seed=seed,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(model_dir / "tb"),
        )

    eval_callback = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(model_dir / "eval"),
        eval_freq=max(25_000 // n_envs, 1),
        n_eval_episodes=50,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100_000 // n_envs, 1),
        save_path=str(model_dir / "checkpoints"),
        name_prefix="ppo_cr",
    )
    win_callback = WinRateCallback()

    model.learn(
        total_timesteps=timesteps,
        callback=[eval_callback, checkpoint_callback, win_callback],
        progress_bar=True,
    )

    final_path = model_dir / "ppo_cr_final.zip"
    model.save(str(final_path))
    print(f"Zapisano model koncowy: {final_path}")

    best_eval = model_dir / "best_model.zip"
    if best_eval.exists():
        import shutil

        shutil.copy2(best_eval, DEFAULT_BEST_PATH)
        print(f"Najlepszy model ewaluacji: {DEFAULT_BEST_PATH}")

    train_env.close()
    eval_env.close()
    return model


def evaluate_model(
    model_path: str | Path,
    *,
    n_episodes: int = 100,
    deterministic: bool = True,
    seed: int = 0,
) -> dict[str, float]:
    env = make_training_env(seed)
    model = MaskablePPO.load(str(model_path))

    wins = 0
    losses = 0
    draws = 0
    rewards: list[float] = []
    lengths: list[int] = []

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(
                obs,
                deterministic=deterministic,
                action_masks=env.unwrapped.action_masks(),
            )
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated
        rewards.append(total_reward)
        lengths.append(steps)
        winner = info.get("winner")
        if winner == 0:
            wins += 1
        elif winner == 1:
            losses += 1
        else:
            draws += 1

    env.close()
    return {
        "episodes": n_episodes,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "mean_length": float(np.mean(lengths)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Trening PPO vs LogicAgent")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--resume", type=str, default=None, help="Sciezka do .zip aby wznowic trening")
    parser.add_argument(
        "--eval-only",
        type=str,
        default=None,
        metavar="MODEL.zip",
        help="Tylko ewaluacja wytrenowanego modelu",
    )
    parser.add_argument("--eval-episodes", type=int, default=100)
    args = parser.parse_args()

    if args.eval_only:
        stats = evaluate_model(args.eval_only, n_episodes=args.eval_episodes)
        print(
            f"Wynik vs LogicAgent ({stats['episodes']} meczy): "
            f"wygrane={stats['wins']} przegrane={stats['losses']} remisy={stats['draws']} "
            f"win_rate={stats['win_rate']:.1%} "
            f"srednia_nagroda={stats['mean_reward']:.4f}"
        )
        return

    train(
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        learning_rate=args.lr,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
