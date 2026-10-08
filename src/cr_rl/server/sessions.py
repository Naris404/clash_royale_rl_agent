"""Game sessions — asyncio loop ticking the Board in real time.

One session = one human (P0) vs bot/RL (P1) match. The game engine is
shared with RL training (no risk of logic divergence); the session adds
unit identifiers, real-time cadence and the coaching layer.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from typing import Optional

from cr_rl.game.board import (
    ARENA_LENGTH,
    ARENA_WIDTH,
    BRIDGE_HALF_WIDTH,
    BRIDGE_LANE_X,
    DEPLOY_ZONES,
    MATCH_TIME_LIMIT,
    RIVER_HALF_WIDTH,
    RIVER_Y,
    TICK_DT,
    TOWER_LAYOUT,
    Board,
)
from cr_rl.game.cards import PLAYABLE_CARDS, cards_dic
from cr_rl.coach.evaluator import GRADE_SCORE, CoachEngine, KeyMomentTracker
from cr_rl.coach.inference import PolicyInspector
from cr_rl.agents.logic import LogicAgent
from cr_rl.agents.rl import RLAgent
from cr_rl.server.protocol import (
    ErrorMsg,
    EventMsg,
    FeedbackMsg,
    HelloMsg,
    HintMsg,
    KeyMomentMsg,
    ServerMessage,
    HitState,
    SnapshotMsg,
    SpellState,
    TowerState,
    TroopState,
)

logger = logging.getLogger(__name__)

MODE_COACH = "coach"  # vs LogicAgent + coach hints
MODE_VS_BOT = "vs_bot"  # vs LogicAgent
MODE_VS_RL = "vs_rl"  # vs trained PPO model
VALID_MODES = frozenset({MODE_COACH, MODE_VS_BOT, MODE_VS_RL})

HINT_EVERY_TICKS = 5  # hint every 0.5 s at 10 Hz
MIN_SPEED = 0.25
MAX_SPEED = 4.0


def arena_config() -> dict:
    """Static arena configuration for the client (hello message)."""
    return {
        "arena_width": ARENA_WIDTH,
        "arena_length": ARENA_LENGTH,
        "river_y": RIVER_Y,
        "river_half_width": RIVER_HALF_WIDTH,
        "bridge_lane_x": list(BRIDGE_LANE_X),
        "bridge_half_width": BRIDGE_HALF_WIDTH,
        "deploy_zones": {
            str(p): [
                {"name": z.name, "x": z.x, "y": z.y, "card_types": sorted(z.card_types)}
                for z in zones
            ]
            for p, zones in DEPLOY_ZONES.items()
        },
        "tower_layout": {
            str(p): {k: list(v) for k, v in layout.items()} for p, layout in TOWER_LAYOUT.items()
        },
        "cards": {name: stats for name, stats in cards_dic.items()},
        "playable_cards": list(PLAYABLE_CARDS),
        "tick_dt": TICK_DT,
        "time_limit": MATCH_TIME_LIMIT,
    }


class GameSession:
    """One match: Board state + opponent + coach; tick() is thread-safe."""

    def __init__(
        self,
        session_id: str,
        mode: str,
        inspector: PolicyInspector,
        *,
        lang: str = "en",
        seed: Optional[int] = None,
    ):
        if mode not in VALID_MODES:
            raise ValueError(f"Nieznany tryb gry: {mode!r}")
        self.id = session_id
        self.mode = mode
        self.lang = lang
        self._seed = seed

        self.board = Board(seed=seed)
        self.board.reset(seed=seed)
        self.coach: Optional[CoachEngine] = (
            CoachEngine(inspector, lang=lang) if mode == MODE_COACH else None
        )
        self.opponent = self._make_opponent(inspector)

        self.paused = False
        self.speed = 1.0
        self.tick_count = 0

        self._lock = threading.Lock()
        self._running = False
        self._next_entity_id = 1
        self._key_moments = KeyMomentTracker(lang=lang)
        self.move_log: list[dict] = []
        self._alive_towers = self._tower_ids_alive()
        self._tower_id_map = self._build_tower_id_map()

    def _make_opponent(self, inspector: PolicyInspector):
        if self.mode == MODE_VS_RL and inspector.available:
            return RLAgent(model_path=inspector.model_path, deterministic=True)
        return LogicAgent()

    def _build_tower_id_map(self) -> dict[int, str]:
        """Tower object → stable id mapping (e.g. 'p0_king', 'p1_l')."""
        suffix = {"Tower_L": "l", "Tower_R": "r", "King_Tower": "king"}
        by_pos = {
            (player, pos[0], pos[1]): f"p{player}_{suffix[key]}"
            for player, layout in TOWER_LAYOUT.items()
            for key, pos in layout.items()
        }
        return {id(t): by_pos[(t.owner, t.x, t.y)] for t in self.board.towers}

    def _tower_ids_alive(self) -> set[str]:
        suffix = {"Tower_L": "l", "Tower_R": "r", "King_Tower": "king"}
        alive = set()
        for player, layout in TOWER_LAYOUT.items():
            for key, pos in layout.items():
                tower = next(
                    (t for t in self.board.towers if t.owner == player and t.x == pos[0] and t.y == pos[1]),
                    None,
                )
                if tower is not None and tower.alive:
                    alive.add(f"p{player}_{suffix[key]}")
        return alive

    # --- client interaction (thread-safe, called from executors) ---

    def handle_play(self, card: str, x: float, y: float) -> list[ServerMessage]:
        """Human play: graded by the coach BEFORE execution, then pending."""
        with self._lock:
            if self.board.done:
                return [ErrorMsg(message="Match finished")]
            if not self.board.card_in_hand(0, card):
                return [ErrorMsg(message=f"Card not in hand: {card}")]
            if not self.board.can_play_card_at(0, card, x, y):
                return [
                    ErrorMsg(
                        message="Cannot play this card: check elixir and the deploy area"
                    )
                ]

            messages: list[ServerMessage] = []
            if self.coach is not None:
                grade = self.coach.grade_move(self.board, card, x, y, player=0)
                self.move_log.append(
                    {
                        "time": self.board.time,
                        "card": card,
                        "zone": grade.zone,
                        "grade": grade.grade,
                        "score": grade.score,
                        "delta_value": grade.delta_value,
                    }
                )
                messages.append(
                    FeedbackMsg(
                        grade=grade.grade,
                        score=grade.score,
                        card=grade.card,
                        zone=grade.zone,
                        human_prob=grade.human_prob,
                        delta_value=grade.delta_value,
                        best_card=grade.best_card,
                        best_zone=grade.best_zone,
                        reason=grade.reason,
                    )
                )
            self.board.set_pending_play(0, (card, x, y))
            return messages

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self.paused = paused

    def set_speed(self, value: float) -> None:
        with self._lock:
            self.speed = max(MIN_SPEED, min(MAX_SPEED, value))

    def rematch(self, seed: Optional[int] = None) -> None:
        with self._lock:
            new_seed = seed if seed is not None else (self._seed or 0) + 1000
            self._seed = new_seed
            self.board.reset(seed=new_seed)
            self.tick_count = 0
            self.paused = False
            self.move_log.clear()
            self._key_moments.reset()
            self.opponent.reset()
            self._tower_id_map = self._build_tower_id_map()
            self._alive_towers = self._tower_ids_alive()

    # --- game loop ---

    def tick(self) -> list[ServerMessage]:
        """One simulation step + messages (snapshot, hint, events). Thread-safe."""
        with self._lock:
            self.tick_count += 1

            action_p1 = self.opponent.choose_action(self.board, player=1)
            result = self.board.step(0, action_p1=action_p1)

            messages: list[ServerMessage] = [self.build_snapshot()]

            if self.coach is not None and self.tick_count % HINT_EVERY_TICKS == 0:
                suggestion = self.coach.suggest(self.board, player=0)
                messages.append(
                    HintMsg(
                        card=suggestion.card,
                        slot=suggestion.slot,
                        zone=suggestion.zone,
                        prob=suggestion.prob,
                        value=suggestion.value,
                        reason=suggestion.reason,
                        hint=suggestion.hint,
                        top=suggestion.top,
                        source=suggestion.source,
                    )
                )
                moment = self._key_moments.update(self.board.time, suggestion.value)
                if moment is not None:
                    messages.append(
                        KeyMomentMsg(time=moment.time, delta_value=moment.delta_value, note=moment.note)
                    )

            messages.extend(self._detect_tower_events())

            if result.terminated or result.truncated:
                messages.append(self._match_end_event())

            return messages

    async def run(self, outbox: asyncio.Queue) -> None:
        """Real-time loop; tick runs in an executor so it does not block the loop."""
        self._running = True
        loop = asyncio.get_running_loop()
        try:
            while self._running:
                await asyncio.sleep(TICK_DT / self.speed)
                with self._lock:
                    skip = self.paused or self.board.done
                if skip:
                    continue
                messages = await loop.run_in_executor(None, self.tick)
                for message in messages:
                    outbox.put_nowait(message.model_dump(exclude_none=True))
        except asyncio.CancelledError:
            raise
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False

    def hello(self) -> HelloMsg:
        return HelloMsg(
            session_id=self.id,
            mode=self.mode,
            coach_enabled=self.coach is not None,
            lang=self.lang,
            config=arena_config(),
        )

    # --- messages ---

    def build_snapshot(self) -> SnapshotMsg:
        board = self.board
        troops = []
        for troop in board.troops:
            if not troop.alive:
                continue
            sid = getattr(troop, "_sid", None)
            if sid is None:
                sid = self._next_entity_id
                self._next_entity_id += 1
                troop._sid = sid
            troops.append(
                TroopState(
                    id=sid,
                    card=troop.name,
                    owner=troop.owner,
                    x=round(troop.x, 3),
                    y=round(troop.y, 3),
                    hp=round(troop.hp, 1),
                    max_hp=float(troop.max_hp),
                )
            )

        spells = []
        for effect in board.spell_effects:
            sid = getattr(effect, "_sid", None)
            if sid is None:
                sid = self._next_entity_id
                self._next_entity_id += 1
                effect._sid = sid
            spells.append(
                SpellState(
                    id=sid,
                    card=effect.card,
                    owner=effect.owner,
                    x=effect.x,
                    y=effect.y,
                    start_x=effect.start_x,
                    start_y=effect.start_y,
                    radius=effect.radius,
                    age=round(effect.age, 3),
                    travel_time=effect.travel_time,
                )
            )

        towers = [
            TowerState(
                id=self._tower_id_map[id(tower)],
                name=tower.name,
                owner=tower.owner,
                x=tower.x,
                y=tower.y,
                hp=round(tower.hp, 1),
                max_hp=float(tower.max_hp),
                alive=tower.alive,
            )
            for tower in board.towers
        ]

        hits = [
            HitState(
                owner=hit.owner,
                from_x=round(hit.from_x, 3),
                from_y=round(hit.from_y, 3),
                x=round(hit.x, 3),
                y=round(hit.y, 3),
                damage=hit.damage,
                ranged=hit.ranged,
            )
            for hit in board.hit_events
        ]

        queue = board.hand_queue[0]
        return SnapshotMsg(
            tick=self.tick_count,
            time=round(board.time, 2),
            time_limit=MATCH_TIME_LIMIT,
            elixir=[round(e, 2) for e in board.elixir],
            towers=towers,
            troops=troops,
            spells=spells,
            hits=hits,
            hand=board.get_hand(0),
            next_card=queue[0] if queue else None,
            done=board.done,
            winner=board.winner,
        )

    def _detect_tower_events(self) -> list[EventMsg]:
        alive_now = self._tower_ids_alive()
        destroyed = self._alive_towers - alive_now
        self._alive_towers = alive_now
        return [
            EventMsg(kind="tower_down", detail={"tower": tower_id, "time": self.board.time})
            for tower_id in sorted(destroyed)
        ]

    def _match_end_event(self) -> EventMsg:
        scores = [m["score"] for m in self.move_log]
        grade_counts: dict[str, int] = {}
        for entry in self.move_log:
            grade_counts[entry["grade"]] = grade_counts.get(entry["grade"], 0) + 1
        top_moments = sorted(
            self._key_moments.moments, key=lambda m: abs(m.delta_value), reverse=True
        )[:5]
        return EventMsg(
            kind="match_end",
            detail={
                "winner": self.board.winner,
                "time": self.board.time,
                "moves": len(self.move_log),
                "mean_score": (sum(scores) / len(scores)) if scores else None,
                "grade_counts": grade_counts,
                "key_moments": [
                    {"time": m.time, "delta_value": m.delta_value, "note": m.note}
                    for m in top_moments
                ],
                "move_log": self.move_log,
            },
        )


class SessionManager:
    """Registry of active sessions; shared PolicyInspector (one model in RAM)."""

    def __init__(self, inspector: Optional[PolicyInspector] = None):
        self.inspector = inspector or PolicyInspector()
        self._sessions: dict[str, GameSession] = {}

    def create(self, mode: str, lang: str = "en", seed: Optional[int] = None) -> GameSession:
        session = GameSession(str(uuid.uuid4()), mode, self.inspector, lang=lang, seed=seed)
        self._sessions[session.id] = session
        logger.info("Created session %s (mode=%s)", session.id, mode)
        return session

    def get(self, session_id: str) -> Optional[GameSession]:
        return self._sessions.get(session_id)

    def discard(self, session_id: str) -> None:
        session = self._sessions.pop(session_id, None)
        if session is not None:
            session.stop()
            logger.info("Closed session %s", session_id)

    def __len__(self) -> int:
        return len(self._sessions)
