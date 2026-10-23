"""Ablation arms: each differs from the full configuration in exactly one setting."""

from __future__ import annotations

from dataclasses import fields

import pytest

from cr_rl.experiments.arms import ARMS, Arm
from cr_rl.game.rewards import RewardWeights
from cr_rl.training.phases import PHASE_LAYOUTS
from cr_rl.training.train import make_vec_training_env

OTHER_ARMS = [name for name in ARMS if name != "full"]


def differing_settings(arm: Arm, other: Arm) -> list[str]:
    return [f.name for f in fields(Arm) if getattr(arm, f.name) != getattr(other, f.name)]


def test_there_are_four_arms():
    assert list(ARMS) == ["full", "no-shaping", "no-curriculum", "no-self-play"]


def test_full_is_the_default_configuration():
    assert ARMS["full"] == Arm(layout="full", reward_weights=RewardWeights())


@pytest.mark.parametrize("name", OTHER_ARMS)
def test_each_arm_differs_from_full_in_exactly_one_setting(name):
    assert len(differing_settings(ARMS[name], ARMS["full"])) == 1


def test_no_shaping_zeroes_both_shaping_weights_and_keeps_the_outcome_weight():
    weights = ARMS["no-shaping"].reward_weights
    assert (weights.crown, weights.hp) == (0.0, 0.0)
    assert weights.win == RewardWeights().win
    assert ARMS["no-shaping"].layout == "full"


def test_the_layout_arms_use_the_matching_phase_layout_and_the_default_reward():
    for name in ("no-curriculum", "no-self-play"):
        assert name in PHASE_LAYOUTS
        assert ARMS[name].layout == name
        assert ARMS[name].reward_weights == RewardWeights()


def test_training_envs_receive_the_reward_weights():
    weights = RewardWeights(crown=0.0, hp=0.0)
    env = make_vec_training_env(0, ["random", "random"], reward_weights=weights)
    try:
        assert env.get_attr("reward_weights") == [weights, weights]
    finally:
        env.close()
