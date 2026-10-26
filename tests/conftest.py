"""Fixtures shared by several test modules."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def untrained_model_path(tmp_path_factory):
    """Fresh (untrained) MaskablePPO — exercises the NN path without training."""
    from sb3_contrib import MaskablePPO

    from cr_rl.env.gym_env import ClashRoyaleEnv

    env = ClashRoyaleEnv(seed=0)
    model = MaskablePPO("MlpPolicy", env, n_steps=64, n_epochs=1, batch_size=64, verbose=0)
    path = tmp_path_factory.mktemp("models") / "untrained.zip"
    model.save(str(path))
    env.close()
    return path
