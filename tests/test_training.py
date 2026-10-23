"""Training entry point: subprocess envs, env-side masks and the smoke runs through the phase layouts."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from cr_rl.game.board import NUM_ACTIONS
from cr_rl.training import train as training
from cr_rl.training.phases import PAST_SELF
from cr_rl.training.train import make_vec_training_env, train


class TestSubprocessEnvs:
    def test_each_process_gets_its_opponent_and_serves_its_own_masks(self):
        vec_env = make_vec_training_env(seed=0, opponents=["random", "logic", "logic-eps"])
        try:
            kinds = [type(opponent).__name__ for opponent in vec_env.get_attr("opponent")]
            assert kinds == ["RandomAgent", "LogicAgent", "EpsilonLogicAgent"]
            masks = vec_env.env_method("action_masks")
            assert len(masks) == 3 and all(mask.shape == (NUM_ACTIONS,) for mask in masks)
        finally:
            vec_env.close()

    def test_every_env_starts_its_episodes_from_its_own_seeds(self):
        vec_env = make_vec_training_env(seed=0, opponents=["random"] * 3)
        try:
            first = [vec_env.env_method("reset", indices=i)[0][1]["seed"] for i in range(3)]
            second = [vec_env.env_method("reset", indices=i)[0][1]["seed"] for i in range(3)]
        finally:
            vec_env.close()
        assert len(set(first + second)) == 6


def smoke_run(
    model_dir: Path, layout: str, *, n_envs: int, timesteps: int, pool_interval: int
) -> list[Counter]:
    """Train a tiny run and return the opponent mix of every env set built along the way."""
    built = []
    original = training.make_vec_training_env

    def spy(seed, opponents, **kwargs):
        built.append(Counter(PAST_SELF if isinstance(o, Path) else o for o in opponents))
        return original(seed, opponents, **kwargs)

    training.make_vec_training_env = spy
    try:
        train(
            timesteps=timesteps, n_envs=n_envs, seed=0, layout=layout, model_dir=model_dir,
            n_steps=256 * n_envs, selection_seeds=range(1_000_000, 1_000_001),
            pool_interval=pool_interval, pool_matches=2,
        )
    finally:
        training.make_vec_training_env = original
    return built


SCRIPTED = {"logic": 8, "logic-eps": 2, "random": 2}
SELF_PLAY = {PAST_SELF: 4, "logic": 6, "logic-eps": 1, "random": 1}


class TestFullLayoutRun:
    @pytest.fixture(scope="class")
    def run(self, tmp_path_factory):
        model_dir = tmp_path_factory.mktemp("full")
        built = smoke_run(model_dir, "full", n_envs=12, timesteps=5_000, pool_interval=1_000)
        return model_dir, built

    def test_passes_through_all_three_phases_with_the_specified_opponents(self, run):
        model_dir, built = run
        assert built[:2] == [{"random": 12}, SCRIPTED]
        assert built[2:] == [SELF_PLAY] * 2  # self-play: rebuilt after the pool check
        assert (model_dir / "ppo_cr_final.zip").is_file()

    def test_the_pool_is_checked_once_between_the_chunks_and_logged(self, run):
        model_dir, _ = run
        (decision,) = json.loads((model_dir / "pool.json").read_text(encoding="utf-8"))
        assert decision["matches"] == 2
        assert len(list((model_dir / "pool").glob("*.zip"))) == 1


def test_no_self_play_run_never_builds_a_past_self_env(tmp_path):
    built = smoke_run(tmp_path, "no-self-play", n_envs=2, timesteps=1_000, pool_interval=300)
    assert built == [{"random": 2}, {"logic": 2}]
    assert not (tmp_path / "pool.json").exists()


def test_no_curriculum_run_starts_in_the_scripted_mix(tmp_path):
    built = smoke_run(tmp_path, "no-curriculum", n_envs=2, timesteps=1_000, pool_interval=300)
    assert built[0] == {"logic": 2} and set(built[1]) == {PAST_SELF}
