"""Testy CoachEngine — sugestie, ocena zagrań, kluczowe momenty, fallback."""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from cr_rl.game.board import Board, NUM_ACTIONS
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
    def test_nearest_zone_by_distance(self):
        # strefy P0: (3,6), (9,6), (14,6)
        assert nearest_zone(0, "Knight", 2.0, 5.0) == 0
        assert nearest_zone(0, "Knight", 9.0, 9.0) == 1
        assert nearest_zone(0, "Knight", 16.0, 4.0) == 2

    def test_nearest_zone_for_spell_uses_lane_only(self):
        assert nearest_zone(0, "Fireball", 3.2, 25.0) == 0
        assert nearest_zone(0, "Fireball", 13.8, 25.0) == 2

    def test_encode_decode_roundtrip(self):
        board = _board()
        _force_hand(board, 0, ["Knight", "Giant", "Cannon", "Musketeer"], ["Hog_Rider", "Fireball"])
        action = encode_action(board, 0, "Giant", 2)
        assert action == 1 + 1 * 3 + 2
        card, slot, zone = decode_action(board, 0, action)
        assert (card, slot, zone) == ("Giant", 1, 2)

    def test_encode_returns_none_for_card_not_in_hand(self):
        board = _board()
        _force_hand(board, 0, ["Knight", "Giant", "Cannon", "Musketeer"], ["Hog_Rider", "Fireball"])
        assert encode_action(board, 0, "Fireball", 0) is None

    def test_decode_noop(self):
        board = _board()
        assert decode_action(board, 0, 0) == (None, None, None)


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
                assert suggestion.zone in (0, 1, 2)
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
        grade = coach.grade_move(board, suggestion.card, *[(3.0, 6.0), (9.0, 6.0), (14.0, 6.0)][suggestion.zone], player=0)
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
        zones = [(3.0, 6.0), (9.0, 6.0), (14.0, 6.0)]
        grade = coach.grade_move(board, suggestion.card, *zones[suggestion.zone], player=0)
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
