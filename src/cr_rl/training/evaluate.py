"""Evaluation of a trained PPO model vs LogicAgent."""

from __future__ import annotations

import argparse

from cr_rl.training.train import DEFAULT_BEST_PATH, evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=str(DEFAULT_BEST_PATH))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    stats = evaluate_model(args.model, n_episodes=args.episodes, seed=args.seed)
    print(f"Model: {args.model}")
    print(
        f"Matches: {stats['episodes']} | "
        f"W: {stats['wins']} L: {stats['losses']} D: {stats['draws']} | "
        f"Win rate: {stats['win_rate']:.1%}"
    )
    print(f"Mean reward (tower HP): {stats['mean_reward']:.4f}")
    print(f"Mean match length (ticks): {stats['mean_length']:.0f}")


if __name__ == "__main__":
    main()
