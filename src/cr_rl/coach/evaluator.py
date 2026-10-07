"""CoachEngine — sugestie ruchów, ocena zagrań gracza, kluczowe momenty.

Z wytrenowaną siecią (PolicyInspector) używa rozkładu polityki i V(s);
bez modelu spada na heurystyki LogicAgent (trener działa zawsze).
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cr_rl.game.board import DEPLOY_ZONES, NUM_ZONES, Board
from cr_rl.game.cards import cards_dic
from cr_rl.coach.explainer import build_reason, format_hint
from cr_rl.coach.inference import PolicyEstimate, PolicyInspector
from cr_rl.agents.logic import LogicAgent

logger = logging.getLogger(__name__)

GRADE_BEST = "best"
GRADE_GOOD = "good"
GRADE_INACCURACY = "inaccuracy"
GRADE_BLUNDER = "blunder"
GRADE_UNKNOWN = "unknown"

GRADE_SCORE = {
    GRADE_BEST: 1.0,
    GRADE_GOOD: 0.75,
    GRADE_INACCURACY: 0.4,
    GRADE_BLUNDER: 0.0,
    GRADE_UNKNOWN: 0.5,
}

# Progi klasyfikacji — stroione na nagrodę w skali ułamka HP wież na mecz.
BLUNDER_PROB = 0.05
INACCURACY_PROB = 0.20
BLUNDER_DELTA_V = -0.02
INACCURACY_DELTA_V = -0.005
KEY_MOMENT_DELTA_V = 0.01


def nearest_zone(player: int, card: str, x: float, y: float) -> int:
    """Mapuje dowolną pozycję rzutu na najbliższą dyskretną strefę akcji RL."""
    zones = DEPLOY_ZONES[player]
    card_type = cards_dic.get(card, {}).get("type")
    allowed = [i for i, zone in enumerate(zones) if zone.allows(card_type)]
    return min(allowed, key=lambda i: (zones[i].x - x) ** 2 + (zones[i].y - y) ** 2)


def encode_action(board: Board, player: int, card: str, zone_idx: int) -> Optional[int]:
    """Indeks akcji RL dla (karta, strefa); None gdy karty nie ma na ręce."""
    hand = board.get_hand(player)
    if card not in hand:
        return None
    slot = hand.index(card)
    if not 0 <= zone_idx < NUM_ZONES:
        return None
    return 1 + slot * NUM_ZONES + zone_idx


def decode_action(board: Board, player: int, action: int) -> tuple[Optional[str], Optional[int], Optional[int]]:
    """Akcja → (karta, slot ręki, strefa); (None, None, None) dla noop."""
    if action <= 0:
        return None, None, None
    slot, zone_idx = divmod(action - 1, NUM_ZONES)
    hand = board.get_hand(player)
    if slot >= len(hand):
        return None, None, None
    return hand[slot], slot, zone_idx


@dataclass
class Suggestion:
    card: Optional[str]
    slot: Optional[int]
    zone: Optional[int]
    prob: Optional[float]
    value: Optional[float]
    reason: str
    hint: str
    top: list[dict] = field(default_factory=list)
    source: str = "rl"  # "rl" | "heuristic"


@dataclass
class MoveGrade:
    grade: str
    score: float
    card: str
    zone: int
    human_prob: Optional[float]
    value_before: Optional[float]
    value_after: Optional[float]
    delta_value: Optional[float]
    best_card: Optional[str]
    best_zone: Optional[int]
    best_prob: Optional[float]
    reason: str


@dataclass
class KeyMoment:
    time: float
    delta_value: float
    value: float
    note: str


class KeyMomentTracker:
    """Wykrywa duże wahania V(s) — do logu kluczowych momentów meczu."""

    def __init__(self, threshold: float = KEY_MOMENT_DELTA_V, lang: str = "pl"):
        self.threshold = threshold
        self.lang = lang
        self._prev: Optional[float] = None
        self.moments: list[KeyMoment] = []

    def update(self, time_s: float, value: Optional[float]) -> Optional[KeyMoment]:
        if value is None:
            return None
        moment = None
        if self._prev is not None:
            delta = value - self._prev
            if abs(delta) >= self.threshold:
                if self.lang == "en":
                    note = "Position improved" if delta > 0 else "Position worsened"
                else:
                    note = "Pozycja się poprawiła" if delta > 0 else "Pozycja się pogorszyła"
                moment = KeyMoment(time=time_s, delta_value=delta, value=value, note=note)
                self.moments.append(moment)
        self._prev = value
        return moment

    def reset(self) -> None:
        self._prev = None
        self.moments.clear()


class CoachEngine:
    """Fasada trenera: suggest() co tick, grade_move() po zagraniu gracza."""

    def __init__(
        self,
        inspector: Optional[PolicyInspector] = None,
        *,
        lang: str = "pl",
        sim_ticks_after_move: int = 3,
    ):
        self.inspector = inspector or PolicyInspector()
        self.lang = lang
        self.sim_ticks_after_move = sim_ticks_after_move
        self._fallback = LogicAgent() if not self.inspector.available else None

    @property
    def uses_neural_net(self) -> bool:
        return self.inspector.available

    # --- sugestie ---

    def suggest(self, board: Board, player: int = 0) -> Suggestion:
        if self.inspector.available:
            estimate = self.inspector.estimate(board, player)
            if estimate is not None:
                return self._suggestion_from_estimate(board, player, estimate)
        return self._suggestion_from_heuristic(board, player)

    def _suggestion_from_estimate(
        self, board: Board, player: int, estimate: PolicyEstimate
    ) -> Suggestion:
        card, slot, zone_idx = decode_action(board, player, estimate.action)
        top = []
        for action, prob in estimate.top_k:
            t_card, t_slot, t_zone = decode_action(board, player, action)
            top.append({"card": t_card, "slot": t_slot, "zone": t_zone, "prob": prob})
        prob = float(estimate.probs[estimate.action])
        return Suggestion(
            card=card,
            slot=slot,
            zone=zone_idx,
            prob=prob,
            value=estimate.value,
            reason=build_reason(board, player, card, lang=self.lang),
            hint=format_hint(card, zone_idx, prob, lang=self.lang),
            top=top,
            source="rl",
        )

    def _suggestion_from_heuristic(self, board: Board, player: int) -> Suggestion:
        assert self._fallback is not None
        self._fallback.reset()
        play = self._fallback._decide_play(board, player)
        if play is None:
            return Suggestion(
                card=None, slot=None, zone=None, prob=None, value=None,
                reason=build_reason(board, player, None, lang=self.lang),
                hint=format_hint(None, None, None, lang=self.lang),
                source="heuristic",
            )
        card, x, y = play
        zone_idx = nearest_zone(player, card, x, y)
        hand = board.get_hand(player)
        slot = hand.index(card) if card in hand else None
        return Suggestion(
            card=card,
            slot=slot,
            zone=zone_idx,
            prob=None,
            value=None,
            reason=self._fallback.last_decision or build_reason(board, player, card, lang=self.lang),
            hint=format_hint(card, zone_idx, None, lang=self.lang),
            source="heuristic",
        )

    # --- ocena zagrań ---

    def grade_move(self, board: Board, card: str, x: float, y: float, player: int = 0) -> MoveGrade:
        """Ocenia zagranie gracza WYWOŁANE PRZED wykonaniem ruchu na planszy."""
        zone_idx = nearest_zone(player, card, x, y)

        if not self.inspector.available:
            return self._grade_with_heuristic(board, card, zone_idx, player)

        estimate = self.inspector.estimate(board, player)
        assert estimate is not None
        human_action = encode_action(board, player, card, zone_idx)
        human_prob = float(estimate.probs[human_action]) if human_action is not None else 0.0

        value_after = self._simulate_value_after(board, player, card, x, y)
        delta_v = (value_after - estimate.value) if value_after is not None else None

        grade = self._classify(human_action, estimate, human_prob, delta_v)
        best_card, _, best_zone = decode_action(board, player, estimate.action)
        return MoveGrade(
            grade=grade,
            score=GRADE_SCORE[grade],
            card=card,
            zone=zone_idx,
            human_prob=human_prob,
            value_before=estimate.value,
            value_after=value_after,
            delta_value=delta_v,
            best_card=best_card,
            best_zone=best_zone,
            best_prob=float(estimate.probs[estimate.action]),
            reason=build_reason(board, player, best_card, lang=self.lang),
        )

    def _grade_with_heuristic(
        self, board: Board, card: str, zone_idx: int, player: int
    ) -> MoveGrade:
        assert self._fallback is not None
        self._fallback.reset()
        play = self._fallback._decide_play(board, player)
        if play is None:
            grade = GRADE_INACCURACY  # heurystyka woli czekać, gracz wydał eliksir
            best_card, best_zone = None, None
        else:
            best_card = play[0]
            best_zone = nearest_zone(player, play[0], play[1], play[2])
            if card == best_card and zone_idx == best_zone:
                grade = GRADE_BEST
            elif card == best_card:
                grade = GRADE_GOOD
            else:
                grade = GRADE_INACCURACY
        return MoveGrade(
            grade=grade,
            score=GRADE_SCORE[grade],
            card=card,
            zone=zone_idx,
            human_prob=None,
            value_before=None,
            value_after=None,
            delta_value=None,
            best_card=best_card,
            best_zone=best_zone,
            best_prob=None,
            reason=build_reason(board, player, best_card, lang=self.lang),
        )

    def _classify(
        self,
        human_action: Optional[int],
        estimate: PolicyEstimate,
        human_prob: float,
        delta_v: Optional[float],
    ) -> str:
        if human_action is not None and human_action == estimate.action:
            return GRADE_BEST
        if delta_v is not None and delta_v <= BLUNDER_DELTA_V:
            return GRADE_BLUNDER
        if human_prob <= BLUNDER_PROB:
            return GRADE_BLUNDER
        if delta_v is not None and delta_v <= INACCURACY_DELTA_V:
            return GRADE_INACCURACY
        if human_prob <= INACCURACY_PROB:
            return GRADE_INACCURACY
        return GRADE_GOOD

    def _simulate_value_after(
        self, board: Board, player: int, card: str, x: float, y: float
    ) -> Optional[float]:
        """V(s') po zagraniu karty i kilku tickach noop — przybliżona wartość ruchu."""
        try:
            sim = copy.deepcopy(board)
            if not sim.play_card_at(player, card, x, y):
                return None
            for _ in range(self.sim_ticks_after_move):
                sim.step(0, action_p1=0)
                if sim.done:
                    break
            return self.inspector.value(sim, player)
        except Exception:
            logger.exception("Symulacja wartości po ruchu nie powiodła się")
            return None
