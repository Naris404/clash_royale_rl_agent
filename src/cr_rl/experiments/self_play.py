"""Self-play: agent trenuje przeciwko własnemu świeżemu checkpointowi.

Przeciwnik co --reload-freq kroków przeładowuje się najnowszym checkpointem
bieżącego treningu — zapobiega overfittingowi do jednego bota regułowego.

Uruchomienie:
  python -m experiments.self_play --timesteps 1000000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from cr_rl.agents.rl import RLAgent
from cr_rl.game.board import Board
from cr_rl.paths import EXPERIMENT_RUNS_DIR
from cr_rl.training.train import evaluate_model, make_vec_training_env

RUN_DIR = EXPERIMENT_RUNS_DIR / "self_play"


def checkpoint_metadata(path: str | Path) -> dict:
    """Return portable provenance for a checkpoint used by an experiment."""
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = checkpoint.stat()
    return {
        "path": str(checkpoint),
        "sha256": digest.hexdigest(),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
    }


def aggregate_match_results(matches: list[dict]) -> dict:
    """Aggregate match rows without depending on SB3 or the simulator."""
    episodes = len(matches)
    wins = sum(match["winner"] == 0 for match in matches)
    losses = sum(match["winner"] == 1 for match in matches)
    draws = episodes - wins - losses
    return {
        "episodes": episodes,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / episodes if episodes else None,
        "mean_tower_hp_margin": (
            sum(float(match["tower_hp_margin"]) for match in matches) / episodes
            if episodes else None
        ),
        "mean_steps": (
            sum(int(match["steps"]) for match in matches) / episodes if episodes else None
        ),
    }


def run_self_play_trial(
    player0_checkpoint: str | Path,
    older_player1_checkpoint: str | Path,
    *,
    episodes: int,
    seed: int,
    max_steps: int = 2500,
) -> dict:
    """Evaluate P0 against a fixed older MaskablePPO checkpoint as P1."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    player0_meta = checkpoint_metadata(player0_checkpoint)
    player1_meta = checkpoint_metadata(older_player1_checkpoint)
    if player0_meta["sha256"] == player1_meta["sha256"]:
        raise ValueError("Player 1 must use a distinct older checkpoint")

    player0 = RLAgent(seed=seed, model_path=player0_meta["path"], deterministic=True)
    player1 = RLAgent(seed=seed + 1, model_path=player1_meta["path"], deterministic=True)
    if not player0.is_trained or not player1.is_trained:
        raise RuntimeError("Both self-play players must load MaskablePPO checkpoints")

    matches: list[dict] = []
    for episode in range(episodes):
        episode_seed = seed + episode
        board = Board(seed=episode_seed)
        board.reset(seed=episode_seed)
        player0.reset(seed=episode_seed)
        player1.reset(seed=episode_seed + 10_000)
        steps = 0
        for steps in range(1, max_steps + 1):
            action0 = player0.choose_action(board, player=0)
            action1 = player1.choose_action(board, player=1)
            result = board.step(action0, action1)
            if result.terminated or result.truncated:
                break
        hp0 = sum(float(tower.hp) for tower in board.towers if tower.owner == 0)
        hp1 = sum(float(tower.hp) for tower in board.towers if tower.owner == 1)
        matches.append(
            {
                "episode": episode,
                "seed": episode_seed,
                "winner": board.winner,
                "steps": steps,
                "tower_hp_player0": hp0,
                "tower_hp_player1": hp1,
                "tower_hp_margin": hp0 - hp1,
            }
        )

    return {
        "schema_version": 1,
        "experiment": "self_play_checkpoint_trial",
        "label": "Self-play vs older checkpoint",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "settings": {
            "episodes": episodes,
            "max_steps": max_steps,
            "deterministic": True,
            "player0": "candidate",
            "player1": "fixed_older_checkpoint",
        },
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform()},
        "checkpoints": {"player0": player0_meta, "player1_older": player1_meta},
        "metrics": aggregate_match_results(matches),
        "matches": matches,
    }


def write_json_result(payload: dict, output: str | Path) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    return destination


class SelfPlayOpponent:
    """Przeciwnik grający najnowszym checkpointem bieżącego treningu."""

    def __init__(self, *, seed: int = 0, initial_checkpoint: str | Path | None = None):
        self._agent = RLAgent(
            seed=seed,
            model_path=initial_checkpoint,
            deterministic=False,
            play_chance=0.5,
        )
        initial = Path(initial_checkpoint) if initial_checkpoint else None
        self._loaded_path: Optional[Path] = initial if initial and initial.is_file() else None

    def maybe_reload(self, checkpoint_dir: Path) -> None:
        checkpoints = sorted(
            checkpoint_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        if not checkpoints:
            return
        latest = checkpoints[0]
        if latest == self._loaded_path:
            return
        try:
            self._agent = RLAgent(model_path=latest, deterministic=False)
            self._loaded_path = latest
        except Exception:
            pass  # checkpoint w trakcie zapisu — spróbujemy ponownie

    def choose_action(self, board, player: int = 1) -> int:
        return self._agent.choose_action(board, player)

    def reset(self) -> None:
        self._agent.reset()


class ReloadOpponentCallback(BaseCallback):
    def __init__(self, opponent: SelfPlayOpponent, checkpoint_dir: Path, reload_freq: int):
        super().__init__()
        self.opponent = opponent
        self.checkpoint_dir = checkpoint_dir
        self.reload_freq = reload_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.reload_freq == 0:
            self.opponent.maybe_reload(self.checkpoint_dir)
        return True


def train_self_play(
    *,
    timesteps: int,
    n_envs: int,
    seed: int,
    reload_freq: int,
    initial_checkpoint: str | Path | None = None,
) -> MaskablePPO:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = RUN_DIR / "selfplay_checkpoints"

    opponent = SelfPlayOpponent(seed=seed, initial_checkpoint=initial_checkpoint)
    train_env = make_vec_training_env(n_envs, seed, opponent=opponent)

    model = MaskablePPO(
        "MlpPolicy",
        train_env,
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

    checkpoint_cb = CheckpointCallback(
        save_freq=max(reload_freq // n_envs, 1000),
        save_path=str(checkpoint_dir),
        name_prefix="selfplay",
    )
    reload_cb = ReloadOpponentCallback(opponent, checkpoint_dir, max(reload_freq, 1000))

    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_cb, reload_cb],
        progress_bar=True,
    )
    train_env.close()

    final_path = RUN_DIR / "ppo_cr_selfplay_final.zip"
    model.save(str(final_path))
    print(f"Zapisano: {final_path}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-play trening PPO")
    parser.add_argument(
        "--trial-player0",
        type=str,
        default=None,
        metavar="MODEL.zip",
        help="Run a no-training checkpoint trial with this model as player 0",
    )
    parser.add_argument(
        "--trial-player1-old",
        type=str,
        default=None,
        metavar="OLDER.zip",
        help="Fixed older MaskablePPO checkpoint used as player 1",
    )
    parser.add_argument("--output", type=str, default=str(RUN_DIR / "trial_results.json"))
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reload-freq", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument(
        "--initial-opponent",
        type=str,
        default="models/ppo_cr_best.zip",
        help="Starszy checkpoint przeciwnika; brak pliku oznacza losowy start",
    )
    args = parser.parse_args()

    if args.trial_player0 or args.trial_player1_old:
        if not args.trial_player0 or not args.trial_player1_old:
            parser.error("--trial-player0 and --trial-player1-old must be provided together")
        payload = run_self_play_trial(
            args.trial_player0,
            args.trial_player1_old,
            episodes=args.eval_episodes,
            seed=args.seed,
            max_steps=args.max_steps,
        )
        output = write_json_result(payload, args.output)
        print(f"Saved self-play trial: {output}")
        return

    train_self_play(
        timesteps=args.timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        reload_freq=args.reload_freq,
        initial_checkpoint=args.initial_opponent,
    )
    model_path = RUN_DIR / "ppo_cr_selfplay_final.zip"
    stats = evaluate_model(model_path, n_episodes=args.eval_episodes, seed=args.seed + 500)
    print(
        f"[self_play] win_rate vs LogicAgent={stats['win_rate']:.1%} "
        f"(W:{stats['wins']} L:{stats['losses']} D:{stats['draws']})"
    )
    payload = {
        "experiment": "self_play",
        "label": "Self-play",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "reload_freq": args.reload_freq,
        "initial_opponent": args.initial_opponent,
        "model": str(model_path),
        **stats,
    }
    (RUN_DIR / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
