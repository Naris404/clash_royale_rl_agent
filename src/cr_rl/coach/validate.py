"""Offline coach validation — agreement of move grades with the real match outcome.

LogicAgent plays as a "human proxy" (P0) against LogicAgent (P1);
every P0 move is graded by CoachEngine. Thesis hypothesis:
won matches should have a higher average play quality than lost ones.

Usage:
  python -m coach.validate --games 20
  python -m coach.validate --games 50 --model models/ppo_cr_best.zip
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cr_rl.game.board import Board
from cr_rl.coach.evaluator import GRADE_SCORE, GRADE_UNKNOWN, CoachEngine
from cr_rl.coach.inference import PolicyInspector
from cr_rl.agents.logic import LogicAgent
from cr_rl.paths import MODELS_DIR

STATS_DIR = MODELS_DIR / "stats"


def play_graded_match(coach: CoachEngine, seed: int, max_steps: int = 2500) -> dict:
    board = Board(seed=seed)
    board.reset(seed=seed)
    human_proxy = LogicAgent()
    opponent = LogicAgent()

    grades: list[str] = []
    exact_agreements = 0
    card_agreements = 0
    for _ in range(max_steps):
        play = human_proxy._decide_play(board, 0)
        if play is not None:
            suggestion = coach.suggest(board, player=0)
            grade = coach.grade_move(board, play[0], play[1], play[2], player=0)
            grades.append(grade.grade)
            played_zone = grade.zone
            if suggestion.card == play[0]:
                card_agreements += 1
                if suggestion.zone == played_zone:
                    exact_agreements += 1
            board.set_pending_play(0, play)
        opponent.choose_action(board, player=1)
        result = board.step(0, 0)
        if result.terminated or result.truncated:
            break

    scores = [GRADE_SCORE[g] for g in grades if g != GRADE_UNKNOWN]
    return {
        "winner": board.winner,
        "moves": len(grades),
        "mean_score": float(np.mean(scores)) if scores else 0.5,
        "grades": {g: grades.count(g) for g in sorted(set(grades))},
        "card_agreements": card_agreements,
        "exact_agreements": exact_agreements,
    }


def pairwise_outcome_accuracy(matches: list[dict]) -> float | None:
    """Fraction of win/loss pairs where the winner has the higher move grade."""
    wins = [m for m in matches if m["winner"] == 0]
    losses = [m for m in matches if m["winner"] == 1]
    if not wins or not losses:
        return None
    comparisons = [
        1.0 if win["mean_score"] > loss["mean_score"] else
        0.5 if win["mean_score"] == loss["mean_score"] else 0.0
        for win in wins
        for loss in losses
    ]
    return float(np.mean(comparisons))


def run_validation(games: int, model_path: str | None, seed: int) -> dict:
    inspector = PolicyInspector(model_path)
    coach = CoachEngine(inspector)
    if not coach.uses_neural_net:
        print("WARNING: no model — heuristic grading (LogicAgent self-agreement).")

    matches = []
    for game in range(games):
        match = play_graded_match(coach, seed=seed + game)
        matches.append(match)
        print(
            f"  match {game + 1}/{games}: winner={match['winner']} "
            f"moves={match['moves']} mean_quality={match['mean_score']:.3f}"
        )

    wins = [m for m in matches if m["winner"] == 0]
    losses = [m for m in matches if m["winner"] == 1]
    draws = [m for m in matches if m["winner"] is None]

    win_scores = np.array([m["mean_score"] for m in wins]) if wins else np.array([0.5])
    loss_scores = np.array([m["mean_score"] for m in losses]) if losses else np.array([0.5])

    # point-biserial correlation: play quality vs win
    outcomes = np.array([1.0 if m["winner"] == 0 else 0.0 for m in matches if m["winner"] is not None])
    qualities = np.array([m["mean_score"] for m in matches if m["winner"] is not None])
    correlation = None
    if len(outcomes) >= 3 and outcomes.std() > 0 and qualities.std() > 0:
        correlation = float(np.corrcoef(outcomes, qualities)[0, 1])

    summary = {
        "experiment": "coach_grading_accuracy",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "model": str(model_path) if model_path else None,
        "neural_net": coach.uses_neural_net,
        "games": games,
        "wins": len(wins),
        "losses": len(losses),
        "draws": len(draws),
        "mean_score_wins": float(win_scores.mean()),
        "mean_score_losses": float(loss_scores.mean()),
        "score_gap": float(win_scores.mean() - loss_scores.mean()),
        "outcome_quality_correlation": correlation,
        "outcome_pairwise_accuracy": pairwise_outcome_accuracy(matches),
        "card_agreement": (
            sum(m["card_agreements"] for m in matches)
            / max(1, sum(m["moves"] for m in matches))
        ),
        "exact_action_agreement": (
            sum(m["exact_agreements"] for m in matches)
            / max(1, sum(m["moves"] for m in matches))
        ),
        "method": (
            "LogicAgent proxy moves are graded before execution; mean grades are "
            "compared with final tower outcomes. Agreement is a proxy metric, "
            "not human-labelled ground truth."
        ),
        "matches": matches,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Walidacja offline CoachEngine")
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    summary = run_validation(args.games, args.model, args.seed)

    print("\n=== Coach validation summary ===")
    print(f"Matches: {summary['games']} (W:{summary['wins']} L:{summary['losses']} D:{summary['draws']})")
    print(f"Mean quality in wins:       {summary['mean_score_wins']:.3f}")
    print(f"Mean quality in losses:     {summary['mean_score_losses']:.3f}")
    print(f"Gap (should be > 0):        {summary['score_gap']:+.3f}")
    if summary["outcome_quality_correlation"] is not None:
        print(f"Outcome-quality correlation:{summary['outcome_quality_correlation']:+.3f}")
    if summary["outcome_pairwise_accuracy"] is not None:
        print(f"Win/loss pair accuracy:     {summary['outcome_pairwise_accuracy']:.1%}")
    print(f"Card / action agreement:    {summary['card_agreement']:.1%} / {summary['exact_action_agreement']:.1%}")

    STATS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = STATS_DIR / "coach_validation.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
