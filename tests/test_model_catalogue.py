"""Model catalogue — which models a player may pick as the RL Opponent."""

from __future__ import annotations

import shutil

import pytest
from fastapi.testclient import TestClient

from cr_rl.agents.logic import LogicAgent
from cr_rl.coach.inference import PolicyInspector
from cr_rl.server import model_catalogue
from cr_rl.server.app import create_app
from cr_rl.server.model_catalogue import CATALOGUE, DEFAULT_MODEL_ID, ModelCatalogue, ModelNotAvailable

SELF_PLAY_ID = "v2-self-play"
SELF_PLAY_FILE = "ppo_cr_self_play.zip"
V2_SHIPPED_ID = "v2-shipped"
V2_SHIPPED_FILE = "ppo_cr_v2_shipped.zip"
SHIPPED_FILE = "ppo_cr_best.zip"
V3_ID = "v3"
V3_FILE = "ppo_cr_v3.zip"


@pytest.fixture()
def models_dir(tmp_path, untrained_model_path):
    """A models/ folder holding every catalogue file."""
    for filename in (SHIPPED_FILE, V2_SHIPPED_FILE, SELF_PLAY_FILE, V3_FILE):
        shutil.copy(untrained_model_path, tmp_path / filename)
    return tmp_path


@pytest.fixture()
def client(models_dir):
    inspector = PolicyInspector(models_dir / SHIPPED_FILE)
    with TestClient(create_app(inspector, models_dir=models_dir)) as test_client:
        yield test_client


def _opponent_model(client, session_id):
    return client.app.state.sessions.get(session_id).opponent.agent._model


def _create(client, **body):
    return client.post("/api/sessions", json={"mode": "vs_rl", "seed": 1, **body})


class TestCatalogueEndpoint:
    def test_lists_entries_with_copy_and_availability_but_no_paths(self, client):
        body = client.get("/api/models").json()

        assert body["default"] == DEFAULT_MODEL_ID
        assert [m["id"] for m in body["models"]] == [e.id for e in CATALOGUE]
        for model in body["models"]:
            assert model["available"] is True
            assert set(model["label"]) == {"en", "pl"}
            assert set(model["description"]) == {"en", "pl"}
        assert ".zip" not in client.get("/api/models").text

    def test_missing_file_is_listed_as_unavailable(self, models_dir):
        (models_dir / SELF_PLAY_FILE).unlink()
        inspector = PolicyInspector(models_dir / SHIPPED_FILE)
        with TestClient(create_app(inspector, models_dir=models_dir)) as client:
            availability = {m["id"]: m["available"] for m in client.get("/api/models").json()["models"]}

        assert availability == {DEFAULT_MODEL_ID: True, V2_SHIPPED_ID: True, SELF_PLAY_ID: False, V3_ID: True}

    def test_health_reports_no_file_path(self, client):
        body = client.get("/api/health").json()

        assert body["shipped_model_available"] is True
        assert "model_path" not in body


class TestSessionModel:
    def test_no_model_id_plays_the_shipped_model(self, client):
        body = _create(client).json()

        assert body["model"] == DEFAULT_MODEL_ID
        shipped = client.app.state.inspector.model
        assert _opponent_model(client, body["session_id"]) is shipped

    def test_catalogue_id_plays_that_model(self, client):
        body = _create(client, model=SELF_PLAY_ID).json()

        assert body["model"] == SELF_PLAY_ID
        playing = _opponent_model(client, body["session_id"])
        assert playing is not client.app.state.inspector.model

    def test_v3_id_plays_the_v3_model_file(self, models_dir, monkeypatch):
        real_load = model_catalogue.load_compatible_model
        loaded = {}
        monkeypatch.setattr(
            model_catalogue, "load_compatible_model", lambda path: loaded.setdefault(path.name, real_load(path))
        )
        inspector = PolicyInspector(models_dir / SHIPPED_FILE)
        with TestClient(create_app(inspector, models_dir=models_dir)) as client:
            body = _create(client, model=V3_ID).json()

            assert body["model"] == V3_ID
            assert _opponent_model(client, body["session_id"]) is loaded[V3_FILE]

    def test_coach_keeps_the_shipped_model_whatever_is_picked(self, client):
        _create(client, model=SELF_PLAY_ID)
        coach = client.post("/api/sessions", json={"mode": "coach", "model": SELF_PLAY_ID}).json()

        session = client.app.state.sessions.get(coach["session_id"])
        assert session.coach is not None
        assert client.app.state.inspector.model_path.name == SHIPPED_FILE
        assert client.app.state.inspector.model is client.app.state.sessions.catalogue.resolve(DEFAULT_MODEL_ID)[1]

    def test_modes_without_an_rl_opponent_report_no_model(self, client):
        body = client.post("/api/sessions", json={"mode": "vs_bot", "model": SELF_PLAY_ID}).json()

        assert body["model"] is None

    @pytest.mark.parametrize("model", ["nope", "../models/ppo_cr_best.zip", "C:/models/ppo_cr_best.zip"])
    def test_unknown_id_or_path_is_rejected(self, client, model):
        assert _create(client, model=model).status_code == 422

    def test_unavailable_id_is_rejected(self, models_dir):
        (models_dir / SELF_PLAY_FILE).unlink()
        inspector = PolicyInspector(models_dir / SHIPPED_FILE)
        with TestClient(create_app(inspector, models_dir=models_dir)) as client:
            assert _create(client, model=SELF_PLAY_ID).status_code == 422

    def test_unavailable_shipped_model_falls_back_to_the_rule_based_bot(self, tmp_path):
        with TestClient(create_app(PolicyInspector(model_path=None), models_dir=tmp_path)) as client:
            body = _create(client).json()

            assert body["model"] is None
            assert isinstance(client.app.state.sessions.get(body["session_id"]).opponent, LogicAgent)
            assert _create(client, model=DEFAULT_MODEL_ID).status_code == 422

    def test_each_model_is_loaded_once_across_sessions(self, models_dir, monkeypatch):
        loads = []
        real_load = model_catalogue.load_compatible_model
        monkeypatch.setattr(model_catalogue, "load_compatible_model", lambda path: loads.append(path) or real_load(path))
        inspector = PolicyInspector(models_dir / SHIPPED_FILE)

        with TestClient(create_app(inspector, models_dir=models_dir)) as client:
            for _ in range(3):
                _create(client, model=SELF_PLAY_ID)
                _create(client)

        # the Shipped model comes from the inspector
        assert loads == [models_dir / V2_SHIPPED_FILE, models_dir / SELF_PLAY_FILE, models_dir / V3_FILE]


def test_the_v2_shipped_model_is_its_own_entry_that_does_not_follow_the_shipped_model_path(models_dir):
    """If another model is shipped, the v2 Shipped model stays selectable from its own file."""
    catalogue = ModelCatalogue(shipped_model="v3", models_dir=models_dir)

    assert catalogue.resolve(None) == (DEFAULT_MODEL_ID, "v3")
    model_id, model = catalogue.resolve(V2_SHIPPED_ID)
    assert model_id == V2_SHIPPED_ID and model not in (None, "v3")


def test_resolve_rejects_unknown_ids(tmp_path):
    with pytest.raises(ModelNotAvailable):
        ModelCatalogue(shipped_model=None, models_dir=tmp_path).resolve("nope")
