"""A checkpoint trained on an older observation layout is ignored, not crashed on."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from cr_rl.agents.rl import load_compatible_model
from cr_rl.game.board import NUM_ACTIONS, OBS_DIM


class _Env(gym.Env):
    def __init__(self, obs_dim: int):
        self.observation_space = spaces.Box(-1.0, 1.0, (obs_dim,), np.float32)
        self.action_space = spaces.Discrete(NUM_ACTIONS)


def _save_model(path, obs_dim: int) -> None:
    from sb3_contrib import MaskablePPO

    MaskablePPO("MlpPolicy", _Env(obs_dim), n_steps=8, batch_size=8).save(str(path))


@pytest.mark.parametrize(
    "old_obs_dim",
    [
        153,  # 6-card layout: 3 + 6 + 4 x 6 cards + 24 x 5
        169,  # 8-card layout with 24 distance-sorted unit slots (before observation v2)
    ],
)
def test_checkpoint_from_an_older_observation_layout_is_ignored(tmp_path, old_obs_dim):
    pytest.importorskip("sb3_contrib")
    _save_model(tmp_path / "old.zip", old_obs_dim)

    assert load_compatible_model(tmp_path / "old.zip") is None


def test_checkpoint_with_the_current_layout_loads(tmp_path):
    pytest.importorskip("sb3_contrib")
    _save_model(tmp_path / "current.zip", OBS_DIM)

    assert load_compatible_model(tmp_path / "current.zip") is not None
