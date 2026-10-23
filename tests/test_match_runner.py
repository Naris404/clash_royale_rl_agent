"""Match runner: one loop for every evaluation, plus the Wilson interval for win counts."""

from __future__ import annotations

from dataclasses import asdict

import pytest
from sb3_contrib import MaskablePPO

from cr_rl.agents.rl import RLAgent
from cr_rl.env.gym_env import ClashRoyaleEnv
from cr_rl.stats.reporting import evaluate_random_baseline, evaluate_saved_model
from cr_rl.training.matches import play_matches, wilson_interval
from cr_rl.training.train import evaluate_model

SEEDS = [3, 4, 5]


@pytest.fixture(scope="module")
def policy() -> MaskablePPO:
    return MaskablePPO("MlpPolicy", ClashRoyaleEnv(seed=0), n_steps=64, batch_size=64, seed=0, device="cpu")


def _random_player() -> RLAgent:
    return RLAgent(seed=1, model_path=None)


class TestPlayMatches:
    @pytest.mark.parametrize("opponent", ["logic", "logic-eps", "random"])
    def test_plays_each_seed_against_a_named_bot(self, opponent):
        results = play_matches(_random_player(), opponent, SEEDS)
        assert [result.seed for result in results] == SEEDS
        for result in results:
            assert result.winner in (0, 1, None)
            assert result.length > 0

    @pytest.mark.parametrize("opponent", ["logic", "logic-eps", "random"])
    def test_same_seeds_give_identical_results(self, opponent):
        first = play_matches(_random_player(), opponent, SEEDS)
        second = play_matches(_random_player(), opponent, SEEDS)
        assert first == second

    def test_plays_a_trained_policy_as_player_zero(self, policy):
        results = play_matches(policy, "logic", SEEDS)
        assert results == play_matches(policy, "logic", SEEDS)
        assert [result.seed for result in results] == SEEDS

    def test_plays_against_a_frozen_policy_as_player_one(self, policy):
        results = play_matches(_random_player(), policy, SEEDS)
        assert results == play_matches(_random_player(), policy, SEEDS)
        assert [result.seed for result in results] == SEEDS

    def test_an_empty_seed_list_plays_nothing(self):
        assert play_matches(_random_player(), "logic", []) == []


class TestWilsonInterval:
    @pytest.mark.parametrize(
        ("wins", "n", "lower", "upper"),
        [
            (0, 10, 0.0, 0.2775),
            (10, 10, 0.7225, 1.0),
            (50, 100, 0.4038, 0.5962),
            (130, 200, 0.5816, 0.7127),
        ],
    )
    def test_matches_reference_values(self, wins, n, lower, upper):
        low, high = wilson_interval(wins, n)
        assert low == pytest.approx(lower, abs=1e-4)
        assert high == pytest.approx(upper, abs=1e-4)

    def test_no_matches_gives_the_uninformative_interval(self):
        assert wilson_interval(0, 0) == (0.0, 1.0)


class TestExistingEvaluationsAreUnchanged:
    def test_evaluate_model_matches_the_original_loop(self, policy, tmp_path):
        path = tmp_path / "model.zip"
        policy.save(str(path))
        assert evaluate_model(path, n_episodes=3, seed=7) == pytest.approx(_original_evaluate_model(policy, 3, 7))

    def test_saved_model_report_matches_evaluate_model(self, policy, tmp_path):
        path = tmp_path / "model.zip"
        policy.save(str(path))
        stats = evaluate_model(path, n_episodes=3, seed=7)
        report = evaluate_saved_model("model", path, episodes=3, seed=7)
        assert (report.wins, report.losses, report.draws) == (stats["wins"], stats["losses"], stats["draws"])
        assert report.mean_length == stats["mean_length"]

    def test_random_baseline_keeps_its_results(self):
        result = asdict(evaluate_random_baseline(3, 5))
        assert result["wins"] == 0
        assert result["losses"] == 3
        assert result["draws"] == 0
        assert result["mean_reward"] == pytest.approx(-0.9727681338457783)
        assert result["mean_length"] == pytest.approx(46.333333333333336)


def _original_evaluate_model(model: MaskablePPO, n_episodes: int, seed: int) -> dict[str, float]:
    """The evaluation loop as it was before the runner existed."""
    env = ClashRoyaleEnv(seed=seed)
    wins = losses = draws = 0
    rewards: list[float] = []
    lengths: list[int] = []
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        total_reward = 0.0
        steps = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True, action_masks=env.action_masks())
            obs, reward, terminated, truncated, info = env.step(int(action))
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated
        rewards.append(total_reward)
        lengths.append(steps)
        winner = info.get("winner")
        wins += winner == 0
        losses += winner == 1
        draws += winner is None
    return {
        "episodes": n_episodes,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / n_episodes,
        "mean_reward": sum(rewards) / n_episodes,
        "mean_length": sum(lengths) / n_episodes,
    }
