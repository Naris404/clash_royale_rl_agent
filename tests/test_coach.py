"""Testy CoachEngine — sugestie, ocena zagrań, kluczowe momenty, fallback."""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from cr_rl.game.board import DEPLOY_ZONES, NUM_ACTIONS, NUM_ZONES, Board
from cr_rl.coach.evaluator import (
    GRADE_BEST,
    GRADE_BLUNDER,
    GRADE_GOOD,
    GRADE_INACCURACY,
    CoachEngine,
    KeyMomentTracker,
    decode_action,
    encode_action,
    nearest_zone,
)
from cr_rl.coach.inference import PolicyInspector


def _board(seed: int = 42) -> Board:
    env = Board(seed=seed)
    env.reset(seed=seed)
    return env


def _zone_name(zone_idx: int, player: int = 0) -> str:
    return DEPLOY_ZONES[player][zone_idx].name


def _zone_xy(zone_idx: int, player: int = 0) -> tuple[float, float]:
    zone = DEPLOY_ZONES[player][zone_idx]
    return zone.x, zone.y


def _force_hand(board: Board, player: int, hand: list[str], queue: list[str]) -> None:
    board.hand[player] = list(hand)
    board.hand_queue[player] = deque(queue)


@pytest.fixture(scope="module")
def untrained_model_path(tmp_path_factory):
    """Świeży (nietrenowany) MaskablePPO — testuje ścieżkę NN bez treningu."""
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker

    from cr_rl.training.train import make_training_env

    env = make_training_env(0)
    model = MaskablePPO("MlpPolicy", env, n_steps=64, n_epochs=1, batch_size=64, verbose=0)
    path = tmp_path_factory.mktemp("models") / "untrained.zip"
    model.save(str(path))
    env.close()
    return path


class TestActionMapping:
    def test_nearest_zone_for_troops_by_distance(self):
        assert _zone_name(nearest_zone(0, "Knight", 2.0, 4.0)) == "back-L"
        assert _zone_name(nearest_zone(0, "Knight", 14.5, 9.0)) == "mid-R"
        assert _zone_name(nearest_zone(0, "Knight", 3.0, 13.0)) == "bridge-L"
        assert _zone_name(nearest_zone(0, "Cannon", 5.5, 9.5)) == "pull-L"
        assert _zone_name(nearest_zone(0, "Cannon", 12.0, 10.0)) == "pull-R"

    def test_nearest_zone_for_spell_only_considers_spell_zones(self):
        assert _zone_name(nearest_zone(0, "Fireball", 3.2, 25.0)) == "spell-tower-L"
        assert _zone_name(nearest_zone(0, "Fireball", 13.8, 25.0)) == "spell-tower-R"
        assert _zone_name(nearest_zone(0, "Fireball", 9.0, 28.0)) == "spell-king"
        assert _zone_name(nearest_zone(0, "Fireball", 3.0, 9.0)) == "spell-tower-L"  # nie "mid-L"

    def test_nearest_zone_for_player1_mirrors(self):
        assert _zone_name(nearest_zone(1, "Knight", 3.0, 23.0)) == "mid-L"
        assert _zone_name(nearest_zone(1, "Fireball", 3.5, 8.0)) == "spell-tower-L"

    def test_encode_decode_roundtrip(self):
        board = _board()
        _force_hand(board, 0, ["Knight", "Giant", "Cannon", "Musketeer"], ["Hog_Rider", "Fireball"])
        action = encode_action(board, 0, "Giant", 7)
        assert action == 1 + 1 * 11 + 7
        card, slot, zone = decode_action(board, 0, action)
        assert (card, slot, zone) == ("Giant", 1, 7)

    def test_every_zone_has_hint_name_in_both_languages(self):
        from cr_rl.coach.explainer import zone_name

        for lang in ("pl", "en"):
            names = {zone_name(i, lang) for i in range(NUM_ZONES)}
            assert len(names) == NUM_ZONES
            assert not any(name.isdigit() for name in names)

    def test_hint_references_pull_zone(self):
        from cr_rl.coach.explainer import format_hint

        zone = next(i for i, z in enumerate(DEPLOY_ZONES[0]) if z.name == "pull-L")
        assert "pull" in format_hint("Cannon", zone, 0.5, lang="en")

    def test_encode_returns_none_for_card_not_in_hand(self):
        board = _board()
        _force_hand(board, 0, ["Knight", "Giant", "Cannon", "Musketeer"], ["Hog_Rider", "Fireball"])
        assert encode_action(board, 0, "Fireball", 0) is None

    def test_decode_noop(self):
        board = _board()
        assert decode_action(board, 0, 0) == (None, None, None)


class TestOutdatedCheckpoint:
    def test_v1_layout_checkpoint_is_ignored(self, tmp_path):
        import gymnasium as gym
        from gymnasium import spaces
        from sb3_contrib import MaskablePPO

        class V1Env(gym.Env):
            observation_space = spaces.Box(-1.0, 1.0, shape=(153,), dtype=np.float32)
            action_space = spaces.Discrete(13)

            def reset(self, *, seed=None, options=None):
                return np.zeros(153, dtype=np.float32), {}

            def step(self, action):
                return np.zeros(153, dtype=np.float32), 0.0, True, False, {}

        path = tmp_path / "v1.zip"
        MaskablePPO("MlpPolicy", V1Env(), n_steps=8, batch_size=8, verbose=0).save(str(path))
        inspector = PolicyInspector(path)
        assert not inspector.available
        assert CoachEngine(inspector).suggest(_board()).source == "heuristic"


class TestHeuristicFallback:
    def test_suggestion_without_model_is_legal(self):
        coach = CoachEngine(PolicyInspector(model_path=None))
        assert not coach.uses_neural_net
        board = _board()
        board.elixir[0] = 10.0
        for _ in range(30):
            suggestion = coach.suggest(board)
            if suggestion.card is not None:
                assert board.card_in_hand(0, suggestion.card)
                assert board.zone_allows(0, suggestion.card, suggestion.zone)
                assert suggestion.source == "heuristic"
            board.step(0, action_p1=0)
            if board.done:
                break

    def test_grade_matches_heuristic_choice(self):
        coach = CoachEngine(PolicyInspector(model_path=None))
        board = _board()
        board.elixir[0] = 10.0
        suggestion = coach.suggest(board)
        assert suggestion.card is not None
        grade = coach.grade_move(board, suggestion.card, *_zone_xy(suggestion.zone), player=0)
        assert grade.grade in (GRADE_BEST, GRADE_GOOD)


class TestNeuralPath:
    def test_estimate_probs_and_value(self, untrained_model_path):
        inspector = PolicyInspector(untrained_model_path)
        assert inspector.available
        board = _board()
        estimate = inspector.estimate(board, player=0)
        assert estimate is not None
        assert estimate.probs.shape == (NUM_ACTIONS,)
        assert estimate.probs.sum() == pytest.approx(1.0, abs=1e-5)
        mask = board.valid_action_mask(0)
        assert (estimate.probs[~mask] == 0.0).all()
        assert np.isfinite(estimate.value)
        assert len(estimate.top_k) >= 1
        assert estimate.top_k[0][0] == estimate.action

    def test_estimate_mirrored_player1(self, untrained_model_path):
        inspector = PolicyInspector(untrained_model_path)
        board = _board()
        estimate = inspector.estimate(board, player=1)
        assert estimate is not None and np.isfinite(estimate.value)

    def test_suggestion_and_grade_with_model(self, untrained_model_path):
        coach = CoachEngine(PolicyInspector(untrained_model_path))
        assert coach.uses_neural_net
        board = _board()
        board.elixir[0] = 10.0

        suggestion = coach.suggest(board)
        assert suggestion.source == "rl"
        assert suggestion.value is not None
        if suggestion.card is not None:
            assert suggestion.slot is not None and suggestion.zone is not None
            assert suggestion.top

        _force_hand(board, 0, ["Knight", "Giant", "Cannon", "Musketeer"], ["Hog_Rider", "Fireball"])
        grade = coach.grade_move(board, "Knight", 3.0, 6.0, player=0)
        assert grade.grade in (GRADE_BEST, GRADE_GOOD, GRADE_INACCURACY, GRADE_BLUNDER)
        assert grade.human_prob is not None and 0.0 <= grade.human_prob <= 1.0
        assert grade.value_before is not None
        assert grade.value_after is not None
        assert grade.delta_value == pytest.approx(grade.value_after - grade.value_before)

    def test_best_move_graded_best(self, untrained_model_path):
        coach = CoachEngine(PolicyInspector(untrained_model_path))
        board = _board()
        board.elixir[0] = 10.0
        suggestion = coach.suggest(board)
        if suggestion.card is None:
            pytest.skip("polityka sugeruje noop — wymuś kartę")
        grade = coach.grade_move(board, suggestion.card, *_zone_xy(suggestion.zone), player=0)
        assert grade.grade == GRADE_BEST


class TestKeyMoments:
    def test_flags_large_swings_only(self):
        tracker = KeyMomentTracker(threshold=0.01)
        assert tracker.update(1.0, 0.10) is None  # pierwsza wartość
        assert tracker.update(2.0, 0.105) is None  # mała zmiana
        moment = tracker.update(3.0, 0.20)  # duży skok
        assert moment is not None
        assert moment.delta_value == pytest.approx(0.095)
        assert len(tracker.moments) == 1

    def test_ignores_none_values(self):
        tracker = KeyMomentTracker()
        assert tracker.update(1.0, None) is None
        assert tracker.update(2.0, 0.5) is None

    def test_reset(self):
        tracker = KeyMomentTracker()
        tracker.update(1.0, 0.1)
        tracker.update(2.0, 0.5)
        tracker.reset()
        assert not tracker.moments
