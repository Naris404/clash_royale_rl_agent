"""Backend tests — REST, WS protocol, session logic."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cr_rl.coach.evaluator import Suggestion
from cr_rl.coach.inference import PolicyInspector
from cr_rl.game.board import REGULATION_TIME
from cr_rl.game.cards import DECK, cards_dic
from cr_rl.server.app import create_app
from cr_rl.server.protocol import parse_client_message
from cr_rl.server.sessions import (
    HINT_EVERY_TICKS,
    MODE_COACH,
    MODE_VS_BOT,
    MODE_VS_RL,
    GameSession,
    SessionManager,
)
from scenario import place_unit


@pytest.fixture()
def inspector() -> PolicyInspector:
    # no model — heuristic coach, tests are deterministic and fast
    return PolicyInspector(model_path=None)


@pytest.fixture()
def client(inspector, tmp_path) -> TestClient:
    with TestClient(create_app(inspector, models_dir=tmp_path)) as test_client:
        yield test_client


LIVE_COACH_MESSAGES = {"hint", "feedback", "key_moment"}


def _session(mode: str, model_path) -> GameSession:
    """A session of `mode` whose Coach (and RL Opponent) run the model at `model_path`."""
    inspector = PolicyInspector(model_path)
    return GameSession(f"s-{mode}", mode, inspector, seed=1, rl_model=inspector.model)


def _end_match(session: GameSession):
    """Ends the match by killing the enemy King Tower; returns the match_end event."""
    king = next(t for t in session.board.towers if t.owner == 1 and t.name == "King_Tower")
    king.hp = 0.0
    king.alive = False
    return next(m for m in session.tick() if m.type == "event" and m.kind == "match_end")


class TestRest:
    def test_health(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["shipped_model_available"] is False

    def test_create_session(self, client):
        response = client.post("/api/sessions", json={"mode": "coach", "lang": "pl", "seed": 1})
        assert response.status_code == 200
        body = response.json()
        assert body["coach_enabled"] is True

    def test_create_session_rejects_bad_mode(self, client):
        response = client.post("/api/sessions", json={"mode": "ranked"})
        assert response.status_code == 422

    def test_delete_session(self, client):
        sid = client.post("/api/sessions", json={"mode": "vs_bot"}).json()["session_id"]
        assert client.delete(f"/api/sessions/{sid}").status_code == 200
        assert client.delete(f"/api/sessions/{sid}").status_code == 404

    def test_training_metrics(self, client):
        response = client.get("/api/training/metrics")
        assert response.status_code == 200
        body = response.json()
        assert {"training", "latest_step", "series", "runs", "models"} <= body.keys()
        assert isinstance(body["series"], list)


class TestProtocol:
    def test_parse_play(self):
        msg = parse_client_message({"type": "play", "card": "Knight", "x": 9.0, "y": 6.0})
        assert msg.card == "Knight"

    def test_parse_rejects_unknown_type(self):
        with pytest.raises(ValueError):
            parse_client_message({"type": "cheat"})

    def test_speed_bounds(self):
        with pytest.raises(Exception):
            parse_client_message({"type": "speed", "value": 100.0})


class TestGameSession:
    def test_tick_returns_snapshot_with_ids(self, inspector):
        session = GameSession("s1", MODE_VS_BOT, inspector, seed=1)
        messages = session.tick()
        snapshot = messages[0]
        assert snapshot.type == "snapshot"
        assert len(snapshot.towers) == 6
        assert {t.id for t in snapshot.towers} == {"p0_l", "p0_r", "p0_king", "p1_l", "p1_r", "p1_king"}
        assert len(snapshot.hand) == 4
        assert snapshot.next_card == session.board.get_next_card(0)
        assert snapshot.next_card not in snapshot.hand
        assert snapshot.hits == []
        assert snapshot.king_active == [False, False]
        assert snapshot.phase == "regulation"
        assert snapshot.clock == pytest.approx(REGULATION_TIME - snapshot.time)

    def test_snapshot_carries_hits_of_last_tick(self, inspector):
        session = GameSession("s1", MODE_VS_BOT, inspector, seed=1)
        tower = next(t for t in session.board.towers if t.owner == 1 and t.name == "Tower")
        place_unit(session.board, "Knight", 0, tower.x, tower.y - 1.0)
        hits = []
        for _ in range(30):
            hits.extend(session.tick()[0].hits)
        assert any(not h.ranged and h.owner == 0 for h in hits)  # knight → tower
        assert any(h.ranged and h.owner == 1 for h in hits)  # tower → knight

    def test_snapshot_carries_deploy_countdown_of_a_placed_unit(self, inspector):
        session = GameSession("s1", MODE_VS_BOT, inspector, seed=1)
        card = next(c for c in session.board.get_hand(0) if cards_dic[c]["type"] != "spell")
        session.board.elixir[0] = 10.0
        session.handle_play(card, 9.0, 6.0)

        troop = session.tick()[0].troops[0]
        assert 0 < troop.deploy_remaining <= cards_dic[card]["deploy_time"]
        for _ in range(15):
            troop = session.tick()[0].troops[0]
        assert troop.deploy_remaining == 0

    def test_coach_mode_emits_hint(self, inspector):
        session = GameSession("s2", MODE_COACH, inspector, seed=1)
        hints = []
        for _ in range(HINT_EVERY_TICKS):
            hints.extend(m for m in session.tick() if m.type == "hint")
        assert len(hints) == 1

    def test_vs_bot_mode_has_no_hints(self, inspector):
        session = GameSession("s3", MODE_VS_BOT, inspector, seed=1)
        card = session.board.get_hand(0)[0]
        assert session.handle_play(card, 9.0, 6.0) == []
        for _ in range(HINT_EVERY_TICKS * 2):
            assert all(m.type not in LIVE_COACH_MESSAGES for m in session.tick())

    def test_handle_play_grades_and_logs_move(self, inspector):
        session = GameSession("s4", MODE_COACH, inspector, seed=1)
        card = session.board.get_hand(0)[0]
        messages = session.handle_play(card, 9.0, 6.0)
        feedback = next(m for m in messages if m.type == "feedback")
        assert feedback.card == card
        assert len(session.move_log) == 1

    def test_handle_play_rejects_card_not_in_hand(self, inspector):
        session = GameSession("s5", MODE_COACH, inspector, seed=1)
        hand = session.board.get_hand(0)
        missing = next(c for c in DECK if c not in hand)
        messages = session.handle_play(missing, 9.0, 6.0)
        assert messages[0].type == "error"

    def test_handle_play_rejects_invalid_zone_and_low_elixir(self, inspector):
        session = GameSession("s-invalid", MODE_COACH, inspector, seed=1)
        hand = session.board.get_hand(0)
        card = next(name for name in hand if cards_dic[name]["type"] != "spell")
        assert session.handle_play(card, 9.0, 24.0)[0].type == "error"
        session.board.elixir[0] = 0.0
        assert session.handle_play(card, 9.0, 6.0)[0].type == "error"
        session.tick()
        assert session.board.get_hand(0) == hand
        assert not any(t.owner == 0 for t in session.board.troops)

    def test_handle_play_executes_on_next_tick(self, inspector):
        session = GameSession("s-exec", MODE_VS_BOT, inspector, seed=1)
        session.board.elixir[0] = 10.0
        card = next(name for name in session.board.get_hand(0) if cards_dic[name]["type"] != "spell")
        session.handle_play(card, 9.0, 6.0)
        session.tick()
        assert card not in session.board.get_hand(0)
        assert [(t.name, t.x, t.y) for t in session.board.troops if t.owner == 0] == [(card, 9.0, 6.0)]

    def test_rematch_resets_state(self, inspector):
        session = GameSession("s6", MODE_COACH, inspector, seed=1)
        for _ in range(10):
            session.tick()
        card = session.board.get_hand(0)[0]
        session.handle_play(card, 9.0, 6.0)
        session.rematch()
        assert session.tick_count == 0
        assert session.board.time == 0.0
        assert not session.move_log
        assert not session.board.done

    def test_speed_clamped(self, inspector):
        session = GameSession("s7", MODE_VS_BOT, inspector, seed=1)
        session.set_speed(100.0)
        assert session.speed == 4.0
        session.set_speed(0.01)
        assert session.speed == 0.25

    def test_tower_down_event(self, inspector):
        session = GameSession("s8", MODE_VS_BOT, inspector, seed=1)
        tower = next(t for t in session.board.towers if t.owner == 1 and t.name == "Tower")
        tower.hp = 0.0
        tower.alive = False
        messages = session.tick()
        events = [m for m in messages if m.type == "event" and m.kind == "tower_down"]
        assert events and events[0].detail["tower"].startswith("p1_")
        snapshot = next(m for m in messages if m.type == "snapshot")
        assert snapshot.crowns == [1, 0]

    def test_match_end_event_has_summary(self, inspector):
        session = GameSession("s9", MODE_COACH, inspector, seed=1)
        king = next(t for t in session.board.towers if t.owner == 1 and t.name == "King_Tower")
        king.hp = 0.0
        king.alive = False
        messages = session.tick()
        end = next(m for m in messages if m.type == "event" and m.kind == "match_end")
        assert end.detail["winner"] == 0
        assert end.detail["crowns"] == [3, 0]
        assert "grade_counts" in end.detail

    def test_match_end_move_log_keeps_the_full_grade_without_a_model(self, inspector):
        session = GameSession("s-log-heuristic", MODE_COACH, inspector, seed=1)
        session.board.elixir[0] = 10.0  # the heuristic plays a card instead of waiting
        card = session.board.get_hand(0)[0]
        session.handle_play(card, 9.0, 6.0)
        move = _end_match(session).detail["move_log"][0]
        assert (move["time"], move["card"]) == (0.0, card)
        assert move["grade"] in {"best", "good", "inaccuracy", "blunder"}
        assert move["value_before"] is None and move["value_after"] is None
        assert move["delta_value"] is None and move["human_prob"] is None
        assert move["best_card"] in DECK
        assert isinstance(move["best_zone"], int)
        assert move["reason"]

    def test_match_end_move_log_keeps_the_full_grade_with_a_model(self, untrained_model_path):
        session = GameSession("s-log-model", MODE_COACH, PolicyInspector(untrained_model_path), seed=1)
        card = session.board.get_hand(0)[0]
        session.handle_play(card, 9.0, 6.0)
        move = _end_match(session).detail["move_log"][0]
        assert move["value_before"] is not None and move["value_after"] is not None
        assert move["delta_value"] == pytest.approx(move["value_after"] - move["value_before"])
        assert 0.0 <= move["human_prob"] <= 1.0
        assert 0.0 <= move["best_prob"] <= 1.0
        assert move["best_card"] in DECK and isinstance(move["best_zone"], int)
        assert move["reason"]

    def test_match_end_evaluation_is_sampled_at_the_hint_interval(self, untrained_model_path):
        session = GameSession("s-eval", MODE_COACH, PolicyInspector(untrained_model_path), seed=1)
        hint_values = []
        for _ in range(3 * HINT_EVERY_TICKS):
            hint_values.extend(m.value for m in session.tick() if m.type == "hint")
        evaluation = _end_match(session).detail["evaluation"]
        assert [sample["value"] for sample in evaluation] == hint_values
        times = [sample["time"] for sample in evaluation]
        assert len(times) == 3 and times == sorted(times)

    def test_match_end_evaluation_is_empty_without_a_model(self, inspector):
        session = GameSession("s-eval-none", MODE_COACH, inspector, seed=1)
        for _ in range(2 * HINT_EVERY_TICKS):
            session.tick()
        assert _end_match(session).detail["evaluation"] == []

    def test_match_end_key_moments_carry_their_value(self, inspector):
        session = GameSession("s-moment", MODE_COACH, inspector, seed=1)
        for value in (0.0, 0.5):  # a swing of 0.5 is a Key moment
            session.coach.suggest = lambda board, player=0, v=value: Suggestion(
                card=None, slot=None, zone=None, prob=None, value=v, reason="", hint=""
            )
            for _ in range(HINT_EVERY_TICKS):
                session.tick()
        (moment,) = _end_match(session).detail["key_moments"]
        assert moment["value"] == 0.5

    @pytest.mark.parametrize("mode", [MODE_VS_BOT, MODE_VS_RL])
    def test_review_is_built_quietly_in_modes_without_the_coach(self, mode, untrained_model_path):
        session = _session(mode, untrained_model_path)
        card = session.board.get_hand(0)[0]
        messages = session.handle_play(card, 9.0, 6.0)
        for _ in range(3 * HINT_EVERY_TICKS):
            messages.extend(session.tick())
        end = _end_match(session)

        assert all(m.type not in LIVE_COACH_MESSAGES for m in messages)
        (move,) = end.detail["move_log"]
        assert move["card"] == card
        assert move["value_before"] is not None and move["delta_value"] is not None
        assert move["best_card"] in DECK and move["reason"]
        times = [sample["time"] for sample in end.detail["evaluation"]]
        assert len(times) == 3 and times == sorted(times)

    def test_vs_bot_samples_the_same_evaluation_as_coach_mode(self, untrained_model_path):
        evaluations = []
        for mode in (MODE_COACH, MODE_VS_BOT):
            session = _session(mode, untrained_model_path)
            for _ in range(3 * HINT_EVERY_TICKS):
                session.tick()
            evaluations.append(_end_match(session).detail["evaluation"])
        assert evaluations[0] == evaluations[1] != []

    def test_vs_bot_tracks_key_moments_without_announcing_them(self, inspector):
        session = GameSession("s-quiet-moment", MODE_VS_BOT, inspector, seed=1)
        messages = []
        for value in (0.0, 0.5):  # a swing of 0.5 is a Key moment
            session.coach.value = lambda board, player=0, v=value: v
            for _ in range(HINT_EVERY_TICKS):
                messages.extend(session.tick())
        assert all(m.type != "key_moment" for m in messages)
        (moment,) = _end_match(session).detail["key_moments"]
        assert moment["value"] == 0.5

    def test_only_coach_mode_is_enabled_for_the_client(self, inspector):
        assert GameSession("s-c", MODE_COACH, inspector).hello().coach_enabled
        assert not GameSession("s-b", MODE_VS_BOT, inspector).hello().coach_enabled
        assert not GameSession("s-r", MODE_VS_RL, inspector).hello().coach_enabled

    def test_rematch_clears_the_evaluation(self, untrained_model_path):
        session = GameSession("s-eval-rematch", MODE_COACH, PolicyInspector(untrained_model_path), seed=1)
        for _ in range(HINT_EVERY_TICKS):
            session.tick()
        session.rematch()
        assert _end_match(session).detail["evaluation"] == []

    @staticmethod
    def _logged(score: float, delta: float | None = None) -> dict:
        return {"time": 0.0, "card": "Knight", "zone": 0, "grade": "good", "score": score, "delta_value": delta}

    def _review_of(self, inspector, moves: list[dict]) -> dict:
        session = GameSession("s-review", MODE_COACH, inspector, seed=1)
        session.move_log.extend(moves)
        return _end_match(session).detail

    def test_match_end_picks_strongest_and_weakest_by_evaluation_change(self, inspector):
        deltas = [0.05, -0.30, 0.20, 0.00, -0.10, 0.40, 0.10]
        detail = self._review_of(inspector, [self._logged(0.5, d) for d in deltas])
        assert detail["strongest"] == [5, 2, 6]
        assert detail["weakest"] == [1, 4, 3]

    def test_match_end_ranks_by_grade_score_without_evaluations(self, inspector):
        scores = [0.75, 0.0, 1.0, 0.4, 0.75, 1.0]
        detail = self._review_of(inspector, [self._logged(s) for s in scores])
        assert sorted(detail["strongest"]) == [2, 4, 5]
        assert sorted(detail["weakest"]) == [0, 1, 3]
        assert detail["weakest"][0] == 1 and detail["strongest"][0] in (2, 5)

    def test_match_end_strongest_and_weakest_never_share_a_move(self, inspector):
        for count in range(0, 9):
            detail = self._review_of(inspector, [self._logged(0.5, 0.0) for _ in range(count)])
            assert not set(detail["strongest"]) & set(detail["weakest"])
            assert len(detail["strongest"]) == len(detail["weakest"]) == min(3, count // 2)

    @pytest.mark.parametrize("count", [0, 1])
    def test_match_end_has_no_strongest_or_weakest_with_under_two_moves(self, inspector, count):
        detail = self._review_of(inspector, [self._logged(1.0, 0.1) for _ in range(count)])
        assert detail["strongest"] == [] and detail["weakest"] == []
        assert detail["moves"] == count

    def test_match_end_picks_from_real_moves_with_a_model(self, untrained_model_path):
        session = GameSession("s-pick-model", MODE_COACH, PolicyInspector(untrained_model_path), seed=1)
        for i in range(4):
            session.board.elixir[0] = 10.0
            session.handle_play(session.board.get_hand(0)[0], 9.0, 6.0 + i)
        detail = _end_match(session).detail
        deltas = [m["delta_value"] for m in detail["move_log"]]
        assert [deltas[i] for i in detail["strongest"]] == sorted(deltas, reverse=True)[:2]
        assert [deltas[i] for i in detail["weakest"]] == sorted(deltas)[:2]


class TestWebSocket:
    def test_full_flow(self, client):
        sid = client.post("/api/sessions", json={"mode": "coach", "seed": 1}).json()["session_id"]
        with client.websocket_connect(f"/ws/sessions/{sid}") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "hello"
            assert hello["coach_enabled"] is True
            assert hello["config"]["arena_width"] == 18.0
            zones = hello["config"]["deploy_zones"]["0"]
            assert len(zones) == 13
            assert zones[6] == {"name": "pull-L", "x": 8.0, "y": 11.0, "card_types": ["air", "building", "ground"]}
            assert zones[10]["card_types"] == ["spell"]

            snapshot = ws.receive_json()
            assert snapshot["type"] == "snapshot"
            assert len(snapshot["hand"]) == 4

            ws.send_json({"type": "play", "card": snapshot["hand"][0], "x": 9.0, "y": 6.0})
            feedback = None
            for _ in range(50):
                message = ws.receive_json()
                if message["type"] == "feedback":
                    feedback = message
                    break
            assert feedback is not None
            assert feedback["card"] == snapshot["hand"][0]

            ws.send_json({"type": "ping"})
            types = []
            for _ in range(50):
                message = ws.receive_json()
                types.append(message["type"])
                if message["type"] == "pong":
                    break
            assert "pong" in types

    def test_unknown_session_closed(self, client):
        with pytest.raises(Exception):
            with client.websocket_connect("/ws/sessions/nope"):
                pass


class TestSessionManager:
    def test_create_get_discard(self, inspector):
        manager = SessionManager(inspector)
        session = manager.create(MODE_COACH, seed=1)
        assert manager.get(session.id) is session
        assert len(manager) == 1
        manager.discard(session.id)
        assert manager.get(session.id) is None
        assert len(manager) == 0
