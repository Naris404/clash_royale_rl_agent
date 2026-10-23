"""Checkpoint selection: win rate on fixed Selection seeds picks the run's model."""

from __future__ import annotations

import pytest

from cr_rl.training import selection
from cr_rl.training.selection import (
    SELECTION_INTERVAL,
    SELECTION_SEEDS,
    CheckpointResult,
    load_selection,
    save_selection,
    select_checkpoint,
)
from cr_rl.training.train import train


def _result(step: int, wins: int, episodes: int = 100) -> CheckpointResult:
    return CheckpointResult(step=step, wins=wins, episodes=episodes, path=f"ppo_cr_{step}_steps.zip")


class TestSelectionSeeds:
    def test_are_100_fixed_seeds_from_one_million(self):
        assert list(SELECTION_SEEDS) == [1_000_000 + i for i in range(100)]

    def test_a_round_runs_every_50k_steps(self):
        assert SELECTION_INTERVAL == 50_000


class TestSelectCheckpoint:
    def test_picks_the_highest_win_rate(self):
        results = [_result(50_000, 30), _result(100_000, 70), _result(150_000, 55)]
        assert select_checkpoint(results).step == 100_000

    def test_a_tie_goes_to_the_later_checkpoint(self):
        results = [_result(50_000, 60), _result(150_000, 60), _result(100_000, 60)]
        assert select_checkpoint(results).step == 150_000

    def test_win_rate_is_wins_over_episodes(self):
        assert _result(50_000, 3, episodes=4).win_rate == 0.75


def test_records_survive_a_save_and_load(tmp_path):
    results = [_result(50_000, 30), _result(100_000, 70)]
    save_selection(results, tmp_path / "selection.json")
    assert load_selection(tmp_path / "selection.json") == results


class TestSmokeRun:
    @pytest.fixture(scope="class")
    def run(self, tmp_path_factory):
        model_dir = tmp_path_factory.mktemp("run")
        seeds = []
        original = selection.play_matches

        def spy(player, opponent, match_seeds, **kwargs):
            seeds.append(list(match_seeds))
            return original(player, opponent, match_seeds, **kwargs)

        selection.play_matches = spy
        try:
            train(
                timesteps=1024, n_envs=2, seed=0, model_dir=model_dir,
                selection_interval=1024, selection_seeds=range(1_000_000, 1_000_002),
            )
        finally:
            selection.play_matches = original
        return model_dir, seeds

    def test_every_round_plays_the_same_selection_seeds(self, run):
        _, seeds = run
        assert len(seeds) >= 2  # at least one periodic round plus the final model
        assert all(round_seeds == [1_000_000, 1_000_001] for round_seeds in seeds)

    def test_every_evaluated_checkpoint_is_kept_and_recorded(self, run):
        model_dir, _ = run
        results = load_selection(model_dir / "selection.json")
        steps = [result.step for result in results]
        assert steps == sorted(set(steps)) and len(steps) >= 2
        assert all((model_dir / "checkpoints" / result.path).is_file() for result in results)

    def test_the_final_model_is_evaluated_last(self, run):
        model_dir, _ = run
        results = load_selection(model_dir / "selection.json")
        assert results[-1].step == 2048  # one 2 x 1024 step rollout

    def test_best_model_is_the_selected_checkpoint(self, run):
        model_dir, _ = run
        results = load_selection(model_dir / "selection.json")
        selected = model_dir / "checkpoints" / select_checkpoint(results).path
        assert (model_dir / "best_model.zip").read_bytes() == selected.read_bytes()
