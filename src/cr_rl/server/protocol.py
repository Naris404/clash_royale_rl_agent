"""Protokół WebSocket — modele pydantic wiadomości klient↔serwer."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# --- klient → serwer ---


class PlayCardMsg(BaseModel):
    type: Literal["play"] = "play"
    card: str
    x: float
    y: float


class PauseMsg(BaseModel):
    type: Literal["pause"] = "pause"
    paused: bool


class SpeedMsg(BaseModel):
    type: Literal["speed"] = "speed"
    value: float = Field(gt=0.1, le=8.0)


class RematchMsg(BaseModel):
    type: Literal["rematch"] = "rematch"


class PingMsg(BaseModel):
    type: Literal["ping"] = "ping"


ClientMessage = PlayCardMsg | PauseMsg | SpeedMsg | RematchMsg | PingMsg


def parse_client_message(data: dict) -> ClientMessage:
    """Walidacja wiadomości klienta; ValueError przy nieznanym typie."""
    msg_type = data.get("type")
    models = {
        "play": PlayCardMsg,
        "pause": PauseMsg,
        "speed": SpeedMsg,
        "rematch": RematchMsg,
        "ping": PingMsg,
    }
    model = models.get(msg_type)
    if model is None:
        raise ValueError(f"Nieznany typ wiadomości: {msg_type!r}")
    return model.model_validate(data)


# --- serwer → klient ---


class TowerState(BaseModel):
    id: str
    name: str
    owner: int
    x: float
    y: float
    hp: float
    max_hp: float
    alive: bool


class TroopState(BaseModel):
    id: int
    card: str
    owner: int
    x: float
    y: float
    hp: float
    max_hp: float


class SpellState(BaseModel):
    id: int
    card: str
    owner: int
    x: float
    y: float
    start_x: float
    start_y: float
    radius: float
    age: float
    travel_time: float


class HitState(BaseModel):
    owner: int
    from_x: float
    from_y: float
    x: float
    y: float
    damage: float
    ranged: bool


class SnapshotMsg(BaseModel):
    type: Literal["snapshot"] = "snapshot"
    tick: int
    time: float
    time_limit: float
    elixir: list[float]
    towers: list[TowerState]
    troops: list[TroopState]
    spells: list[SpellState]
    hits: list[HitState] = Field(default_factory=list)
    hand: list[str]
    next_card: Optional[str]
    done: bool
    winner: Optional[int]


class HintMsg(BaseModel):
    type: Literal["hint"] = "hint"
    card: Optional[str]
    slot: Optional[int]
    zone: Optional[int]
    prob: Optional[float]
    value: Optional[float]
    reason: str
    hint: str
    top: list[dict] = Field(default_factory=list)
    source: str


class FeedbackMsg(BaseModel):
    type: Literal["feedback"] = "feedback"
    grade: str
    score: float
    card: str
    zone: int
    human_prob: Optional[float]
    delta_value: Optional[float]
    best_card: Optional[str]
    best_zone: Optional[int]
    reason: str


class KeyMomentMsg(BaseModel):
    type: Literal["key_moment"] = "key_moment"
    time: float
    delta_value: float
    note: str


class EventMsg(BaseModel):
    type: Literal["event"] = "event"
    kind: str  # "tower_down" | "match_end"
    detail: dict = Field(default_factory=dict)


class HelloMsg(BaseModel):
    type: Literal["hello"] = "hello"
    session_id: str
    mode: str
    coach_enabled: bool
    lang: str
    config: dict


class PongMsg(BaseModel):
    type: Literal["pong"] = "pong"


class ErrorMsg(BaseModel):
    type: Literal["error"] = "error"
    message: str


ServerMessage = SnapshotMsg | HintMsg | FeedbackMsg | KeyMomentMsg | EventMsg | HelloMsg | PongMsg | ErrorMsg


def dump(message: ServerMessage) -> dict:
    return message.model_dump(exclude_none=True)
