"""Training entry point: per-env opponent assignment, curriculum phases and env-side masks."""

from __future__ import annotations

from collections import Counter

import pytest

from cr_rl.game.board import NUM_ACTIONS
from cr_rl.training.train import make_vec_training_env, phase_opponents, train


class TestPhaseOpponents:
    def test_phase_1_is_all_random(self):
        assert phase_opponents(1, 12) == ["random"] * 12

    def test_phase_2_assigns_the_mix_per_env_index(self):
        opponents = phase_opponents(2, 12)
        assert Counter(opponents) == {"logic": 8, "logic-eps": 2, "random": 2}
        assert opponents == phase_opponents(2, 12)


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


@pytest.mark.parametrize("curriculum", [True, False])
def test_smoke_run_trains_and_saves_a_model(tmp_path, curriculum):
    train(
        timesteps=1024, n_envs=2, seed=0, curriculum=curriculum, model_dir=tmp_path,
        selection_seeds=range(1_000_000, 1_000_001),
    )
    assert (tmp_path / "ppo_cr_final.zip").is_file()
