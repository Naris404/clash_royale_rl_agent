"""Testy backendu — REST, protokół WS, logika sesji."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from cr_rl.coach.inference import PolicyInspector
from cr_rl.server.app import create_app
from cr_rl.server.protocol import parse_client_message
from cr_rl.server.sessions import (
    HINT_EVERY_TICKS,
    MODE_COACH,
    MODE_VS_BOT,
    GameSession,
    SessionManager,
)


@pytest.fixture()
def inspector() -> PolicyInspector:
    # bez modelu — trener heurystyczny, testy deterministyczne i szybkie
    return PolicyInspector(model_path=None)


@pytest.fixture()
def client(inspector) -> TestClient:
    with TestClient(create_app(inspector)) as test_client:
        yield test_client


class TestRest:
    def test_health(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is False

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

    def test_models_list(self, client):
        assert client.get("/api/models").status_code == 200

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

    def test_coach_mode_emits_hint(self, inspector):
        session = GameSession("s2", MODE_COACH, inspector, seed=1)
        hints = []
        for _ in range(HINT_EVERY_TICKS):
            hints.extend(m for m in session.tick() if m.type == "hint")
        assert len(hints) == 1

    def test_vs_bot_mode_has_no_hints(self, inspector):
        session = GameSession("s3", MODE_VS_BOT, inspector, seed=1)
        for _ in range(HINT_EVERY_TICKS * 2):
            assert all(m.type != "hint" for m in session.tick())

    def test_handle_play_grades_and_sets_pending(self, inspector):
        session = GameSession("s4", MODE_COACH, inspector, seed=1)
        card = session.board.get_hand(0)[0]
        messages = session.handle_play(card, 9.0, 6.0)
        feedback = next(m for m in messages if m.type == "feedback")
        assert feedback.card == card
        assert session.board._pending_play[0] == (card, 9.0, 6.0)
        assert len(session.move_log) == 1

    def test_handle_play_rejects_card_not_in_hand(self, inspector):
        session = GameSession("s5", MODE_COACH, inspector, seed=1)
        hand = session.board.get_hand(0)
        missing = next(c for c in ("Knight", "Giant", "Cannon", "Musketeer", "Hog_Rider", "Fireball") if c not in hand)
        messages = session.handle_play(missing, 9.0, 6.0)
        assert messages[0].type == "error"

    def test_handle_play_rejects_invalid_zone_and_low_elixir(self, inspector):
        session = GameSession("s-invalid", MODE_COACH, inspector, seed=1)
        card = next(
            name
            for name in session.board.get_hand(0)
            if session.board._is_spell(name) is False
        )
        assert session.handle_play(card, 9.0, 24.0)[0].type == "error"
        session.board.elixir[0] = 0.0
        assert session.handle_play(card, 9.0, 6.0)[0].type == "error"
        assert session.board._pending_play[0] is None

    def test_handle_play_executes_on_next_tick(self, inspector):
        session = GameSession("s-exec", MODE_VS_BOT, inspector, seed=1)
        session.board.elixir[0] = 10.0
        card = next(
            name
            for name in session.board.get_hand(0)
            if session.board._is_spell(name) is False
        )
        session.handle_play(card, 9.0, 6.0)
        session.tick()
        assert session.board._pending_play[0] is None
        assert any(t.owner == 0 and t.name == card for t in session.board.troops)

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

    def test_match_end_event_has_summary(self, inspector):
        session = GameSession("s9", MODE_COACH, inspector, seed=1)
        king = next(t for t in session.board.towers if t.owner == 1 and t.name == "King_Tower")
        king.hp = 0.0
        king.alive = False
        messages = session.tick()
        end = next(m for m in messages if m.type == "event" and m.kind == "match_end")
        assert end.detail["winner"] == 0
        assert "grade_counts" in end.detail


class TestWebSocket:
    def test_full_flow(self, client):
        sid = client.post("/api/sessions", json={"mode": "coach", "seed": 1}).json()["session_id"]
        with client.websocket_connect(f"/ws/sessions/{sid}") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "hello"
            assert hello["coach_enabled"] is True
            assert hello["config"]["arena_width"] == 18.0

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
