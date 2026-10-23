"""WebSocket protocol — pydantic models for client↔server messages."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# --- client → server ---


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
    """Validates a client message; ValueError on an unknown type."""
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
        raise ValueError(f"Unknown message type: {msg_type!r}")
    return model.model_validate(data)


# --- server → client ---


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
    deploy_remaining: float  # seconds until the unit can move and attack; 0 once deployed


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


class PocketState(BaseModel):
    """Extra legal deploy rectangle for the human (an enemy princess tower fell)."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float


class SnapshotMsg(BaseModel):
    type: Literal["snapshot"] = "snapshot"
    tick: int
    time: float
    phase: Literal["regulation", "double_elixir", "overtime"]
    clock: float  # seconds left in the current period (regulation or overtime)
    elixir: list[float]
    king_active: list[bool]  # per player: the king tower has woken up and shoots
    towers: list[TowerState]
    troops: list[TroopState]
    spells: list[SpellState]
    hits: list[HitState] = Field(default_factory=list)
    pockets: list[PocketState] = Field(default_factory=list)  # the human's extra deploy areas
    hand: list[str]
    next_card: str
    crowns: list[int]
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
    aim: Optional[tuple[float, float]] = None  # where the suggested move lands (moves with a push zone)


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
