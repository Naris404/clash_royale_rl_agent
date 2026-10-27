"""Reference-pool selection: checkpoints are scored against LogicAgent and frozen models (ADR-0007)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from cr_rl.training.matches import MatchResult
from cr_rl.training.reference_selection import (
    ReferenceScore,
    load_scores,
    save_scores,
    score_checkpoints,
    select_reference,
)
from cr_rl.training.selection import CheckpointResult, save_selection

SEEDS = range(10)
OPPONENTS = {"logic": "logic", "final-0": Path("final-0.zip")}


def _make_run(run_dir: Path, steps: list[int]) -> Path:
    results = [CheckpointResult(step, 0, 100, f"ppo_cr_{step}_steps.zip") for step in steps]
    save_selection(results, run_dir / "selection.json")
    return run_dir


def _fake_play(wins_by_step: dict[int, dict[str, int]]):
    """A match player with fixed wins per (checkpoint step, opponent name)."""

    def play(checkpoint: Path, opponent, seeds):
        step = int(checkpoint.name.split("_")[2])
        name = "logic" if opponent == "logic" else opponent.stem
        wins = wins_by_step[step][name]
        return [MatchResult(seed, 0 if i < wins else 1, 0.0, 1) for i, seed in enumerate(seeds)]

    return play


def _score(step: int, logic: int, frozen: int) -> ReferenceScore:
    return ReferenceScore(step, f"ppo_cr_{step}_steps.zip", {"logic": logic, "final-0": frozen}, 10)


class TestScoreCheckpoints:
    def test_scores_every_checkpoint_against_every_opponent(self, tmp_path):
        run = _make_run(tmp_path, [100, 200])
        play = _fake_play({100: {"logic": 10, "final-0": 2}, 200: {"logic": 6, "final-0": 4}})

        scores = score_checkpoints(run, OPPONENTS, SEEDS, play=play)

        assert [score.step for score in scores] == [100, 200]
        assert scores[0].win_rates == {"logic": 1.0, "final-0": 0.2}
        assert scores[0].average == pytest.approx(0.6)
        assert scores[1].average == pytest.approx(0.5)
        assert select_reference(scores).step == 100

    def test_step_range_limits_the_scored_checkpoints(self, tmp_path):
        run = _make_run(tmp_path, [100, 200, 300, 400])
        play = _fake_play({step: {"logic": 5, "final-0": 5} for step in (200, 300)})

        scores = score_checkpoints(run, OPPONENTS, SEEDS, play=play, min_step=200, max_step=300)

        assert [score.step for score in scores] == [200, 300]


class TestSelectReference:
    def test_picks_the_highest_average_win_rate(self):
        scores = [_score(100, 10, 0), _score(200, 7, 7), _score(300, 2, 2)]
        assert select_reference(scores).step == 200

    def test_a_tie_goes_to_the_later_checkpoint(self):
        scores = [_score(100, 6, 4), _score(300, 4, 6), _score(200, 5, 5)]
        assert select_reference(scores).step == 300


def test_scores_survive_a_save_and_load(tmp_path):
    scores = [_score(100, 10, 2), _score(200, 6, 4)]
    save_scores(scores, tmp_path / "reference_selection.json")
    assert load_scores(tmp_path / "reference_selection.json") == scores


@pytest.mark.slow
def test_scores_a_real_checkpoint_against_logic_and_a_frozen_model(tmp_path, untrained_model_path):
    (tmp_path / "checkpoints").mkdir()
    shutil.copy(untrained_model_path, tmp_path / "checkpoints" / "ppo_cr_100_steps.zip")
    _make_run(tmp_path, [100])
    opponents = {"logic": "logic", "frozen": untrained_model_path}

    scores = score_checkpoints(tmp_path, opponents, range(1_000_000, 1_000_002), workers=2)

    assert len(scores) == 1
    assert set(scores[0].wins) == {"logic", "frozen"}
    assert scores[0].episodes == 2
    assert all(0 <= wins <= 2 for wins in scores[0].wins.values())
