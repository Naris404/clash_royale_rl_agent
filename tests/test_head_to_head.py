"""Head-to-head between two models on the Report seeds (ADR-0002 eval protocol)."""

from __future__ import annotations

import inspect

import pytest
from sb3_contrib import MaskablePPO

from cr_rl.env.gym_env import ClashRoyaleEnv
from cr_rl.game.board import NUM_ACTIONS, OBS_DIM
from cr_rl.training.matches import wilson_interval
from cr_rl.training.head_to_head import evaluate_head_to_head
from cr_rl.training.report import ENV_VERSION, REPORT_SEEDS

TINY_SEEDS = range(2_000_000, 2_000_002)


def _policy(seed: int) -> MaskablePPO:
    return MaskablePPO("MlpPolicy", ClashRoyaleEnv(seed=0), n_steps=64, batch_size=64, seed=seed, device="cpu")


@pytest.fixture(scope="module")
def policies() -> tuple[MaskablePPO, MaskablePPO]:
    return _policy(0), _policy(1)


class TestHeadToHead:
    @pytest.fixture(scope="class")
    def result(self, policies):
        return evaluate_head_to_head(*policies, name_a="a.zip", name_b="b.zip", seeds=TINY_SEEDS, config={"arm": "full"})

    def test_plays_every_seed_from_both_sides(self, result):
        assert result.as_player_0.episodes == result.as_player_1.episodes == len(TINY_SEEDS)
        assert result.overall.episodes == 2 * len(TINY_SEEDS)
        for field in ("wins", "losses", "draws"):
            assert getattr(result.overall, field) == getattr(result.as_player_0, field) + getattr(
                result.as_player_1, field
            )

    def test_overall_carries_a_wilson_interval(self, result):
        overall = result.overall
        assert overall.opponent == "b.zip"
        assert overall.win_rate == overall.wins / overall.episodes
        assert (overall.ci_low, overall.ci_high) == wilson_interval(overall.wins, overall.episodes)

    def test_metadata_matches_the_report_seed_evaluation(self, result):
        metadata = result.metadata
        assert metadata.model == "a.zip"
        assert (metadata.seed_start, metadata.episodes) == (2_000_000, len(TINY_SEEDS))
        assert metadata.deterministic is True
        assert (metadata.env_version, metadata.obs_dim, metadata.num_actions) == (ENV_VERSION, OBS_DIM, NUM_ACTIONS)
        assert metadata.config == {"arm": "full"}

    def test_is_deterministic(self, policies, result):
        again = evaluate_head_to_head(*policies, name_a="a.zip", name_b="b.zip", seeds=TINY_SEEDS, config={"arm": "full"})
        assert again == result

    def test_a_model_against_itself_is_symmetric(self, policies):
        result = evaluate_head_to_head(policies[0], policies[0], seeds=TINY_SEEDS)
        assert result.overall.wins == result.overall.losses
        assert (result.as_player_0.wins, result.as_player_0.losses) == (
            result.as_player_1.losses,
            result.as_player_1.wins,
        )

    def test_defaults_to_the_200_report_seeds_for_400_matches(self):
        assert inspect.signature(evaluate_head_to_head).parameters["seeds"].default is REPORT_SEEDS
        assert 2 * len(REPORT_SEEDS) == 400
