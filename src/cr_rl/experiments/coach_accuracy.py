"""State-level CoachEngine validation against counterfactual tower-HP outcomes.

The experiment samples deterministic game states, replays every legal card
action from an identical snapshot, and compares coach grades with short-horizon
tower damage. It does not train a model.
"""
from __future__ import annotations

import argparse
import copy
import json
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np

from cr_rl.agents.logic import LogicAgent
from cr_rl.coach.evaluator import CoachEngine, decode_action
from cr_rl.coach.inference import PolicyInspector
from cr_rl.game.board import DEPLOY_ZONES, MAX_TOWER_HP_PER_PLAYER, Board
from cr_rl.experiments.self_play import checkpoint_metadata
from cr_rl.paths import EXPERIMENT_RUNS_DIR

RUN_DIR = EXPERIMENT_RUNS_DIR / "coach_accuracy"


def pairwise_ranking_accuracy(rows: Iterable[dict]) -> dict:
    """Measure whether grade-score ordering agrees with outcome ordering."""
    items = list(rows)
    correct = 0.0
    comparable = 0
    score_ties = 0
    outcome_ties = 0
    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            outcome_delta = float(left["hp_outcome"]) - float(right["hp_outcome"])
            if abs(outcome_delta) <= 1e-12:
                outcome_ties += 1
                continue
            score_delta = float(left["grade_score"]) - float(right["grade_score"])
            comparable += 1
            if abs(score_delta) <= 1e-12:
                score_ties += 1
                correct += 0.5
            elif score_delta * outcome_delta > 0:
                correct += 1.0
    return {
        "accuracy": correct / comparable if comparable else None,
        "comparable_pairs": comparable,
        "score_ties": score_ties,
        "outcome_ties": outcome_ties,
    }


def top_action_accuracy(state_rows: Iterable[list[dict]]) -> dict:
    """Fraction of states where any top-graded action is outcome-optimal."""
    all_states = [rows for rows in state_rows if rows]
    states = [
        rows
        for rows in all_states
        if len({float(row["hp_outcome"]) for row in rows}) > 1
    ]
    hits = 0
    for rows in states:
        best_score = max(float(row["grade_score"]) for row in rows)
        best_outcome = max(float(row["hp_outcome"]) for row in rows)
        predicted = {int(row["action"]) for row in rows if float(row["grade_score"]) == best_score}
        actual = {int(row["action"]) for row in rows if float(row["hp_outcome"]) == best_outcome}
        hits += bool(predicted & actual)
    return {
        "accuracy": hits / len(states) if states else None,
        "evaluable_states": len(states),
        "skipped_outcome_ties": len(all_states) - len(states),
        "hits": hits,
    }


def pearson_correlation(rows: Iterable[dict]) -> float | None:
    items = list(rows)
    if len(items) < 2:
        return None
    scores = np.asarray([float(row["grade_score"]) for row in items], dtype=float)
    outcomes = np.asarray([float(row["hp_outcome"]) for row in items], dtype=float)
    if scores.std() == 0 or outcomes.std() == 0:
        return None
    return float(np.corrcoef(scores, outcomes)[0, 1])


def tower_hp(board: Board, player: int) -> float:
    return sum(float(tower.hp) for tower in board.towers if tower.owner == player)


def counterfactual_hp_outcome(board: Board, action: int, rollout_steps: int) -> float:
    """Replay one action and deterministic continuations from a copied state."""
    replay = copy.deepcopy(board)
    hp0_before, hp1_before = tower_hp(replay, 0), tower_hp(replay, 1)
    continuation0, continuation1 = LogicAgent(), LogicAgent()

    opponent_action = continuation1.choose_action(replay, player=1)
    result = replay.step(action, opponent_action)
    for _ in range(max(0, rollout_steps - 1)):
        if result.terminated or result.truncated:
            break
        action0 = continuation0.choose_action(replay, player=0)
        action1 = continuation1.choose_action(replay, player=1)
        result = replay.step(action0, action1)

    own_damage = hp0_before - tower_hp(replay, 0)
    enemy_damage = hp1_before - tower_hp(replay, 1)
    return float((enemy_damage - own_damage) / MAX_TOWER_HP_PER_PLAYER)


def sample_decision_states(
    *,
    games: int,
    states: int,
    seed: int,
    sample_every: int,
    max_game_steps: int,
) -> list[dict]:
    """Sample reproducible legal-action snapshots from deterministic games."""
    if games < 1 or states < 1 or sample_every < 1 or max_game_steps < 1:
        raise ValueError("games, states, sample_every and max_game_steps must be positive")
    candidates: list[dict] = []
    for game in range(games):
        game_seed = seed + game
        board = Board(seed=game_seed)
        board.reset(seed=game_seed)
        player0, player1 = LogicAgent(), LogicAgent()
        for step in range(max_game_steps):
            legal = np.flatnonzero(board.valid_action_mask(0)).tolist()
            legal_cards = [int(action) for action in legal if action != 0]
            if step % sample_every == 0 and legal_cards:
                candidates.append(
                    {
                        "game": game,
                        "game_seed": game_seed,
                        "step": step,
                        "time": float(board.time),
                        "board": copy.deepcopy(board),
                        "actions": legal_cards,
                    }
                )
            action0 = player0.choose_action(board, player=0)
            action1 = player1.choose_action(board, player=1)
            result = board.step(action0, action1)
            if result.terminated or result.truncated:
                break
    rng = random.Random(seed)
    if len(candidates) > states:
        candidates = rng.sample(candidates, states)
    return sorted(candidates, key=lambda row: (row["game"], row["step"]))


def evaluate_state(coach: CoachEngine, sampled: dict, rollout_steps: int) -> dict:
    board = sampled["board"]
    action_rows: list[dict] = []
    for action in sampled["actions"]:
        card, _slot, zone = decode_action(board, 0, action)
        if card is None or zone is None:
            continue
        x, y = DEPLOY_ZONES[0][zone]
        grade = coach.grade_move(board, card, x, y, player=0)
        action_rows.append(
            {
                "action": action,
                "card": card,
                "zone": zone,
                "grade": grade.grade,
                "grade_score": float(grade.score),
                "policy_probability": grade.human_prob,
                "delta_value": grade.delta_value,
                "hp_outcome": counterfactual_hp_outcome(board, action, rollout_steps),
            }
        )
    pairwise = pairwise_ranking_accuracy(action_rows)
    return {
        "game": sampled["game"],
        "game_seed": sampled["game_seed"],
        "step": sampled["step"],
        "time": sampled["time"],
        "hand": list(board.get_hand(0)),
        "elixir": float(board.elixir[0]),
        "pairwise_accuracy": pairwise["accuracy"],
        "comparable_pairs": pairwise["comparable_pairs"],
        "actions": action_rows,
    }


def run_coach_accuracy(
    *,
    model_path: str | Path | None,
    games: int,
    states: int,
    seed: int,
    sample_every: int,
    rollout_steps: int,
    max_game_steps: int = 2500,
) -> dict:
    if rollout_steps < 1:
        raise ValueError("rollout_steps must be positive")
    model_metadata = checkpoint_metadata(model_path) if model_path is not None else None
    inspector = PolicyInspector(model_metadata["path"] if model_metadata else None)
    coach = CoachEngine(inspector)
    sampled = sample_decision_states(
        games=games,
        states=states,
        seed=seed,
        sample_every=sample_every,
        max_game_steps=max_game_steps,
    )
    evaluations = [evaluate_state(coach, state, rollout_steps) for state in sampled]
    action_rows = [action for state in evaluations for action in state["actions"]]
    pairwise = pairwise_ranking_accuracy(action_rows)
    within_state_pairs = sum(int(state["comparable_pairs"]) for state in evaluations)
    weighted_state_accuracy = (
        sum(float(state["pairwise_accuracy"]) * int(state["comparable_pairs"]) for state in evaluations
            if state["pairwise_accuracy"] is not None)
        / within_state_pairs
        if within_state_pairs else None
    )
    return {
        "schema_version": 1,
        "experiment": "coach_counterfactual_accuracy",
        "label": "Coach grading vs counterfactual tower HP",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform()},
        "model": model_metadata,
        "neural_net": coach.uses_neural_net,
        "settings": {
            "games": games,
            "requested_states": states,
            "sample_every": sample_every,
            "rollout_steps": rollout_steps,
            "max_game_steps": max_game_steps,
            "continuation_policy": "LogicAgent vs LogicAgent",
            "outcome": "normalized enemy tower damage minus own tower damage",
        },
        "metrics": {
            "sampled_states": len(evaluations),
            "evaluated_actions": len(action_rows),
            "within_state_pairwise_accuracy": weighted_state_accuracy,
            "within_state_comparable_pairs": within_state_pairs,
            "global_pairwise_accuracy": pairwise["accuracy"],
            "global_comparable_pairs": pairwise["comparable_pairs"],
            "top_action": top_action_accuracy([state["actions"] for state in evaluations]),
            "grade_outcome_pearson": pearson_correlation(action_rows),
        },
        "states": evaluations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Counterfactual CoachEngine accuracy evaluation")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--states", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-every", type=int, default=20)
    parser.add_argument("--rollout-steps", type=int, default=100)
    parser.add_argument("--max-game-steps", type=int, default=2500)
    parser.add_argument("--output", type=str, default=str(RUN_DIR / "results.json"))
    args = parser.parse_args()
    payload = run_coach_accuracy(
        model_path=args.model,
        games=args.games,
        states=args.states,
        seed=args.seed,
        sample_every=args.sample_every,
        rollout_steps=args.rollout_steps,
        max_game_steps=args.max_game_steps,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Saved coach accuracy evaluation: {output}")


if __name__ == "__main__":
    main()
