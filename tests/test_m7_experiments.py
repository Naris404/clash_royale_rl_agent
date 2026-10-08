from __future__ import annotations

import json

import pytest

from cr_rl.coach.evaluator import CoachEngine
from cr_rl.coach.inference import PolicyInspector
from cr_rl.coach.validate import pairwise_outcome_accuracy, play_graded_match
from cr_rl.experiments.coach_accuracy import (
    checkpoint_metadata,
    pairwise_ranking_accuracy,
    pearson_correlation,
    run_coach_accuracy,
    sample_decision_states,
    top_action_accuracy,
)
from cr_rl.experiments import sweep
from cr_rl.stats.reporting import load_experiment_results


def test_pairwise_outcome_accuracy() -> None:
    matches = [
        {"winner": 0, "mean_score": 0.8},
        {"winner": 0, "mean_score": 0.4},
        {"winner": 1, "mean_score": 0.5},
    ]
    assert pairwise_outcome_accuracy(matches) == pytest.approx(0.5)
    assert pairwise_outcome_accuracy([matches[0]]) is None


def test_graded_replay_exposes_accuracy_counts() -> None:
    coach = CoachEngine(PolicyInspector(model_path=None))
    result = play_graded_match(coach, seed=3, max_steps=30)
    assert result["moves"] >= 0
    assert 0 <= result["card_agreements"] <= result["moves"]
    assert 0 <= result["exact_agreements"] <= result["card_agreements"]


def test_load_experiment_results_normalizes_schemas(tmp_path) -> None:
    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    (sweep_dir / "results.json").write_text(
        json.dumps({"results": [{"config": "baseline", "win_rate": 0.6}]}),
        encoding="utf-8",
    )
    curriculum_dir = tmp_path / "curriculum"
    curriculum_dir.mkdir()
    (curriculum_dir / "results.json").write_text(
        json.dumps({"experiment": "curriculum", "label": "Curriculum", "win_rate": 0.7}),
        encoding="utf-8",
    )

    rows = load_experiment_results(tmp_path)
    assert {row["label"] for row in rows} == {"baseline", "Curriculum"}


def test_sweep_writes_reproducibility_manifest(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(sweep, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(sweep, "train", lambda **_kwargs: None)
    monkeypatch.setattr(
        sweep,
        "evaluate_model",
        lambda *_args, **_kwargs: {
            "episodes": 2,
            "wins": 1,
            "losses": 1,
            "draws": 0,
            "win_rate": 0.5,
            "mean_reward": 0.0,
            "mean_length": 10.0,
        },
    )

    results = sweep.run_sweep(
        timesteps=10,
        n_envs=1,
        eval_episodes=2,
        configs=["baseline"],
        seed=17,
    )
    manifest = json.loads((tmp_path / "baseline" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 17
    assert results[0]["seed"] == 17


def test_checkpoint_metadata(tmp_path) -> None:
    checkpoint = tmp_path / "older.zip"
    checkpoint.write_bytes(b"fixed checkpoint bytes")
    metadata = checkpoint_metadata(checkpoint)
    assert metadata["path"] == str(checkpoint.resolve())
    assert len(metadata["sha256"]) == 64
    with pytest.raises(FileNotFoundError, match="Checkpoint does not exist"):
        checkpoint_metadata(tmp_path / "missing.zip")


def test_counterfactual_ranking_metrics() -> None:
    rows = [
        {"action": 1, "grade_score": 1.0, "hp_outcome": 0.3},
        {"action": 2, "grade_score": 0.4, "hp_outcome": 0.1},
        {"action": 3, "grade_score": 0.0, "hp_outcome": -0.2},
    ]
    ranking = pairwise_ranking_accuracy(rows)
    assert ranking["accuracy"] == 1.0
    assert ranking["comparable_pairs"] == 3
    assert top_action_accuracy([rows]) == {
        "accuracy": 1.0,
        "evaluable_states": 1,
        "skipped_outcome_ties": 0,
        "hits": 1,
    }
    assert pearson_correlation(rows) > 0.9


def test_counterfactual_metrics_handle_ties_and_empty_input() -> None:
    rows = [
        {"action": 1, "grade_score": 0.5, "hp_outcome": 1.0},
        {"action": 2, "grade_score": 0.5, "hp_outcome": 0.0},
    ]
    assert pairwise_ranking_accuracy(rows)["accuracy"] == 0.5
    assert pairwise_ranking_accuracy([])["accuracy"] is None
    assert pearson_correlation([]) is None
    assert top_action_accuracy([])["accuracy"] is None


def test_coach_accuracy_smoke_is_reproducible_and_json_safe() -> None:
    first = sample_decision_states(
        games=1, states=2, seed=9, sample_every=1, max_game_steps=5
    )
    second = sample_decision_states(
        games=1, states=2, seed=9, sample_every=1, max_game_steps=5
    )
    identity = lambda rows: [(row["game_seed"], row["step"], row["actions"]) for row in rows]
    assert identity(first) == identity(second)

    result = run_coach_accuracy(
        model_path=None,
        games=1,
        states=1,
        seed=9,
        sample_every=1,
        rollout_steps=2,
        max_game_steps=5,
    )
    assert result["metrics"]["sampled_states"] <= 1
    assert result["settings"]["rollout_steps"] == 2
    json.dumps(result, allow_nan=False)
