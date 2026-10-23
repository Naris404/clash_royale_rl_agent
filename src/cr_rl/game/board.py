from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass, field, replace
from math import sqrt
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

from cr_rl.game.cards import (
    CARD_TO_ID,
    DEFAULT_SIGHT_RANGE,
    DECK,
    MIN_MOVE_DISTANCE,
    Troop,
    Tower,
    cards_dic,
)

# Board: 1 unit ≈ 1 tile in Clash Royale.
# Official CR arena: 18 tiles wide × 32 tiles long (wiki / datamining).
ARENA_WIDTH = 18.0   # tiles horizontally (X axis)
ARENA_LENGTH = 32.0  # tiles vertically (Y axis); river at Y = 16
TICK_DT = 0.1
MAX_ELIXIR = 10.0
START_ELIXIR = 5.0
ELIXIR_PER_SECOND = 1.0 / 2.8
REGULATION_TIME = 180.0  # 3:00; the leader on crowns wins, a tie goes to overtime
DOUBLE_ELIXIR_TIME = 120.0  # elixir regenerates twice as fast from 2:00 on, through overtime
OVERTIME_TIME = 120.0  # up to 2:00 of sudden death, then the tiebreak
RIVER_Y = ARENA_LENGTH / 2
RIVER_HALF_WIDTH = 1.2
# Two bridges (left / right), lined up with the princess towers so the arena is mirror-symmetric.
BRIDGE_LANE_X = (3.5, 14.5)
BRIDGES = [(BRIDGE_LANE_X[0], RIVER_Y), (BRIDGE_LANE_X[1], RIVER_Y)]
BRIDGE_HALF_WIDTH = 1.0  # bridges are 2 tiles wide
# Waypoint on the river bank before / after a bridge, set back from the water.
BRIDGE_BANK_Y_OFFSET = RIVER_HALF_WIDTH + 0.3
RANGED_MIN_RANGE = 2.0  # range above melee → hit is shown as a projectile
COLLISION_PASSES = 4  # relaxation passes per tick so chains of overlapping units settle

# Units and spells may be placed up to this far from the back edge (behind the king tower).
BACK_EDGE_MARGIN = 0.5
DEPLOY_X_MARGIN = 1.0  # units and spells stay this far from the side walls

# Pocket after an enemy princess tower falls (measured in the real game): 4 tiles deep from the
# enemy bank, as wide as one lane half. Humans only: agents keep the 11 deploy zones (ADR-0006).
POCKET_DEPTH = 4.0
POCKET_WIDTH = ARENA_WIDTH / 2

# Towers (as in CR): princess ~6 tiles from the river, king 4×4 with one free tile behind it.
# P0 = bottom (y grows upward), P1 = top.
TOWER_LAYOUT = {
    0: {
        "Tower_L": (3.5, 8.0),
        "Tower_R": (14.5, 8.0),
        "King_Tower": (9.0, 3.0),
    },
    1: {
        "Tower_L": (3.5, 24.0),
        "Tower_R": (14.5, 24.0),
        "King_Tower": (9.0, 29.0),
    },
}

TROOP_CARD_TYPES = frozenset({"ground", "air", "building"})
TROOP_ONLY_CARD_TYPES = frozenset({"ground", "air"})
SPELL_CARD_TYPES = frozenset({"spell"})


@dataclass(frozen=True)
class DeployZone:
    """Discrete placement spot with absolute coordinates and the card types it accepts."""

    name: str
    x: float
    y: float
    card_types: frozenset[str]

    def allows(self, card_type: str) -> bool:
        return card_type in self.card_types


# P0 layout (ADR-0002). Back zones sit behind the king tower (troops only: a building there would
# be wasted). Pull zones sit near the center so a building planted there is closer to an enemy
# building-targeter at the bridge exit than the princess tower is, with the same margin per lane.
_P0_ZONES = (
    DeployZone("back-L", 7.5, 0.5, TROOP_ONLY_CARD_TYPES),
    DeployZone("back-R", 10.5, 0.5, TROOP_ONLY_CARD_TYPES),
    DeployZone("mid-L", 3.5, 10.5, TROOP_CARD_TYPES),
    DeployZone("mid-R", 14.5, 10.5, TROOP_CARD_TYPES),
    DeployZone("bridge-L", 3.5, 14.0, TROOP_CARD_TYPES),
    DeployZone("bridge-R", 14.5, 14.0, TROOP_CARD_TYPES),
    DeployZone("pull-L", 8.0, 11.0, TROOP_CARD_TYPES),
    DeployZone("pull-R", 10.0, 11.0, TROOP_CARD_TYPES),
    DeployZone("spell-tower-L", 3.5, 24.0, SPELL_CARD_TYPES),
    DeployZone("spell-tower-R", 14.5, 24.0, SPELL_CARD_TYPES),
    DeployZone("spell-king", 9.0, 29.0, SPELL_CARD_TYPES),
)
DEPLOY_ZONES: dict[int, tuple[DeployZone, ...]] = {
    0: _P0_ZONES,
    1: tuple(replace(z, y=ARENA_LENGTH - z.y) for z in _P0_ZONES),
}
NUM_ZONES = len(_P0_ZONES)

HAND_SIZE = 4
DECK_SIZE = len(DECK)
NUM_ACTIONS = 1 + HAND_SIZE * NUM_ZONES  # noop + hand slot (0..3) × zone (44 plays)

# Observation layout v2 (ADR-0002). Every segment offset is pinned by tests/test_observation.py.
UNIT_CARDS = tuple(c for c in DECK if cards_dic[c]["type"] != "spell")  # spells have no grid cells
NUM_LANES = 2  # left / right of the board's center line
DEPTH_BINS = 8  # 4 rows each along the observer's view of y; the river sits on a bin edge
DEPTH_BIN_ROWS = ARENA_LENGTH / DEPTH_BINS
PLAY_HISTORY_LEN = 4  # the opponent's last plays; the Cycle makes them their hand and next card
NO_PLAY = DECK_SIZE  # "none" slot of a play one-hot: that play has not happened yet

OBS_GLOBAL_FEATURES = 4  # own elixir, enemy elixir, elixir advantage, time
OBS_TOWER_BASE = OBS_GLOBAL_FEATURES
OBS_TOWER_FEATURES = 6  # HP of the 6 towers (layout: own L/R/King, enemy L/R/King)
OBS_HAND_BASE = OBS_TOWER_BASE + OBS_TOWER_FEATURES
OBS_HAND_FEATURES = HAND_SIZE * DECK_SIZE
OBS_NEXT_BASE = OBS_HAND_BASE + OBS_HAND_FEATURES  # one-hot of the visible next card
OBS_PLAYS_BASE = OBS_NEXT_BASE + DECK_SIZE  # enemy's last plays, oldest first, each one-hot + none
OBS_PLAYS_FEATURES = PLAY_HISTORY_LEN * (DECK_SIZE + 1)
OBS_GRID_BASE = OBS_PLAYS_BASE + OBS_PLAYS_FEATURES  # [owner: own, enemy][lane][depth][unit card]
OBS_GRID_FEATURES = 2 * NUM_LANES * DEPTH_BINS * len(UNIT_CARDS)
OBS_DIM = OBS_GRID_BASE + OBS_GRID_FEATURES

# Crowns a player holds once the enemy king tower is down (this ends the match at once).
KING_CROWNS = 3

# One player's total max tower HP (2× princess + king); scales the tower-HP shaping potential.
MAX_TOWER_HP_PER_PLAYER = (
    2 * float(cards_dic["Tower"]["hp"]) + float(cards_dic["King_Tower"]["hp"])
)


@dataclass
class StepResult:
    observation: np.ndarray
    terminated: bool
    truncated: bool
    info: dict = field(default_factory=dict)


@dataclass
class SpellEffect:
    """Active spell effect on the board (for animation / preview)."""

    card: str
    x: float
    y: float
    start_x: float
    start_y: float
    radius: float
    owner: int
    age: float = 0.0
    travel_time: float = 0.28
    fade_time: float = 0.55


@dataclass(frozen=True)
class HitEvent:
    """A single hit from the last step (visualisation only)."""

    owner: int
    from_x: float
    from_y: float
    x: float
    y: float
    damage: float
    ranged: bool


class Board:
    """
    Simplified Clash Royale environment for RL.

    Akcja (int 0..44):
      0 — do nothing
      1..44 — play a card (hand slot 0..3 × zone 0..10); 4 cards in hand, cycling the 8-card Deck

    Player 0 learns; player 1 is a random bot by default (a second model can be plugged in).
    """

    def __init__(self, *, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.troops: list[Troop] = []
        self.towers: list[Tower] = []
        self.elixir = [START_ELIXIR, START_ELIXIR]
        self.time = 0.0
        self.done = False
        self.winner: Optional[int] = None
        self._pending_play: dict[int, Optional[tuple[str, float, float]]] = {0: None, 1: None}
        self.hand: dict[int, list[str]] = {0: [], 1: []}
        self.hand_queue: dict[int, deque[str]] = {0: deque(), 1: deque()}
        self.play_history: dict[int, list[str]] = {0: [], 1: []}  # every card played, oldest first
        self.spell_effects: list[SpellEffect] = []
        self.hit_events: list[HitEvent] = []

        self._init_towers()
        self._init_hands()

    @staticmethod
    def _tower_card_name(layout_key: str) -> str:
        if layout_key == "King_Tower":
            return "King_Tower"
        return "Tower"

    def _init_towers(self) -> None:
        self.towers = []
        for player, layout in TOWER_LAYOUT.items():
            for layout_key, pos in layout.items():
                tower = Tower(self._tower_card_name(layout_key), player)
                tower.place(pos)
                self.towers.append(tower)

    def reset(self, *, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self._rng.seed(seed)
            self.np_rng = np.random.default_rng(seed)

        self.troops = []
        self.elixir = [START_ELIXIR, START_ELIXIR]
        self.time = 0.0
        self.done = False
        self.winner = None
        self.spell_effects = []
        self.hit_events = []
        self._pending_play = {0: None, 1: None}
        self._init_towers()
        self._init_hands()
        return self.get_observation(player=0)

    def _init_hands(self) -> None:
        """Shuffled Deck: 4 cards in hand, the rest queued; a played card returns to the back of the queue."""
        for player in (0, 1):
            cards = list(DECK)
            self._rng.shuffle(cards)
            queue: deque[str] = deque(cards)
            self.hand[player] = [queue.popleft() for _ in range(HAND_SIZE)]
            self.hand_queue[player] = queue
            self.play_history[player] = []

    def get_hand(self, player: int) -> list[str]:
        return list(self.hand[player])

    def get_next_card(self, player: int) -> str:
        """The visible next card: the one drawn after the next play."""
        return self.hand_queue[player][0]

    def card_in_hand(self, player: int, card_name: str) -> bool:
        return card_name in self.hand[player]

    def _cycle_hand_after_play(self, player: int, card_name: str) -> None:
        hand = self.hand[player]
        if card_name not in hand:
            return
        hand.remove(card_name)
        self.play_history[player].append(card_name)
        queue = self.hand_queue[player]
        queue.append(card_name)
        if queue:
            hand.append(queue.popleft())

    def set_pending_play(
        self, player: int, play: Optional[tuple[str, float, float]]
    ) -> None:
        self._pending_play[player] = play

    def pocket_zones(self, player: int) -> list[tuple[float, float, float, float]]:
        """Enemy-side rectangles (x_min, x_max, y_min, y_max) opened by the enemy princess towers `player` destroyed."""
        direction = 1.0 if player == 0 else -1.0
        near = RIVER_Y + direction * RIVER_HALF_WIDTH  # the enemy bank, not the water
        far = near + direction * POCKET_DEPTH
        zones = []
        for tower in self.towers:
            if tower.owner == player or tower.name != "Tower" or tower.alive:
                continue
            lane_min = 0.0 if tower.x < ARENA_WIDTH / 2 else ARENA_WIDTH - POCKET_WIDTH
            x_min = max(lane_min, DEPLOY_X_MARGIN)
            x_max = min(lane_min + POCKET_WIDTH, ARENA_WIDTH - DEPLOY_X_MARGIN)
            zones.append((x_min, x_max, min(near, far), max(near, far)))
        return zones

    def _in_deploy_zone(self, player: int, x: float, y: float) -> bool:
        if not (DEPLOY_X_MARGIN <= x <= ARENA_WIDTH - DEPLOY_X_MARGIN):
            return False
        if player == 0:
            return BACK_EDGE_MARGIN <= y <= RIVER_Y - RIVER_HALF_WIDTH  # up to the water's edge, never in it
        return RIVER_Y + RIVER_HALF_WIDTH <= y <= ARENA_LENGTH - BACK_EDGE_MARGIN

    def _in_pocket(self, player: int, x: float, y: float) -> bool:
        return any(
            x_min <= x <= x_max and y_min <= y <= y_max for x_min, x_max, y_min, y_max in self.pocket_zones(player)
        )

    def _in_spell_zone(self, player: int, x: float, y: float) -> bool:
        if not (DEPLOY_X_MARGIN <= x <= ARENA_WIDTH - DEPLOY_X_MARGIN):
            return False
        return BACK_EDGE_MARGIN <= y <= ARENA_LENGTH - BACK_EDGE_MARGIN

    @staticmethod
    def _is_spell(card_name: str) -> bool:
        return cards_dic.get(card_name, {}).get("type") == "spell"

    def _cast_spell(self, player: int, card_name: str, x: float, y: float) -> bool:
        stats = cards_dic.get(card_name)
        if not stats or stats.get("type") != "spell":
            return False

        cost = stats["elisir"]
        if self.elixir[player] < cost:
            return False

        radius = float(stats["radius"])
        damage = float(stats.get("damage", 0))
        tower_damage = float(stats.get("tower_damage", damage))
        kx, ky = TOWER_LAYOUT[player]["King_Tower"]
        for unit in self._all_combatants():
            if unit.owner == player or not unit.alive:
                continue
            if unit.distance_to_xy(x, y) <= radius:
                dealt = tower_damage if unit.is_tower else damage
                unit.hp -= dealt
                if unit.hp <= 0:
                    unit.hp = 0
                    unit.alive = False
                self.hit_events.append(
                    HitEvent(player, kx, ky, unit.x, unit.y, dealt, ranged=False)
                )

        self.spell_effects.append(
            SpellEffect(
                card=card_name,
                x=x,
                y=y,
                start_x=kx,
                start_y=ky,
                radius=radius,
                owner=player,
            )
        )

        self.elixir[player] -= cost
        self._cycle_hand_after_play(player, card_name)
        return True

    def _tick_spell_effects(self, dt: float) -> None:
        for effect in self.spell_effects:
            effect.age += dt
        self.spell_effects = [
            effect
            for effect in self.spell_effects
            if effect.age < effect.travel_time + effect.fade_time
        ]

    def can_play_card_at(self, player: int, card_name: str, x: float, y: float) -> bool:
        """Checks a play without modifying the board state."""
        if self.done or not self.card_in_hand(player, card_name):
            return False
        stats = cards_dic.get(card_name)
        if not stats or self.elixir[player] < stats["elisir"]:
            return False
        if self._is_spell(card_name):
            return self._in_spell_zone(player, x, y)
        return self._in_deploy_zone(player, x, y) or self._in_pocket(player, x, y)

    def play_card_at(self, player: int, card_name: str, x: float, y: float) -> bool:
        if not self.can_play_card_at(player, card_name, x, y):
            return False

        if self._is_spell(card_name):
            return self._cast_spell(player, card_name, x, y)

        cost = cards_dic[card_name]["elisir"]

        troop = Troop(card_name, player)
        troop.place((x, y))
        self.elixir[player] -= cost
        self.troops.append(troop)
        self._cycle_hand_after_play(player, card_name)
        return True

    def action_to_card_zone(self, action: int, player: int) -> tuple[Optional[str], Optional[int]]:
        if action <= 0:
            return None, None
        hand = self.hand[player]
        idx = action - 1
        slot, zone_idx = divmod(idx, NUM_ZONES)
        if slot >= len(hand):
            return None, None
        return hand[slot], zone_idx

    def zone_allows(self, player: int, card_name: str, zone_idx: int) -> bool:
        """Whether a card of this type can be played in the zone (ignoring elixir and hand)."""
        if not 0 <= zone_idx < NUM_ZONES:
            return False
        card_type = cards_dic.get(card_name, {}).get("type")
        return DEPLOY_ZONES[player][zone_idx].allows(card_type)

    def play_card(self, player: int, card_name: str, zone_idx: int) -> bool:
        if not self.zone_allows(player, card_name, zone_idx):
            return False
        zone = DEPLOY_ZONES[player][zone_idx]
        return self.play_card_at(player, card_name, zone.x, zone.y)

    def step(
        self,
        action_p0: int,
        action_p1: Optional[int] = None,
    ) -> StepResult:
        if self.done:
            return StepResult(
                self.get_observation(0), True, False, {"winner": self.winner}
            )

        self.hit_events = []

        if action_p1 is None:
            action_p1 = self._random_bot_action(1)

        self._apply_action(0, action_p0)
        self._apply_action(1, action_p1)

        self._regen_elixir()
        self._simulate_combat(TICK_DT)

        self._remove_dead()
        self._tick_spell_effects(TICK_DT)
        self.time += TICK_DT

        terminated, truncated = self._check_end()

        return StepResult(
            observation=self.get_observation(0),
            terminated=terminated,
            truncated=truncated,
            info={
                "winner": self.winner if (terminated or truncated) else None,
                "time": self.time,
                "elixir": self.elixir[0],
                "troops_alive": sum(1 for t in self.troops if t.alive),
            },
        )

    def _apply_action(self, player: int, action: int) -> None:
        pending = self._pending_play[player]
        if pending is not None:
            card, x, y = pending
            if not self.play_card_at(player, card, x, y) and not self._is_spell(card):
                # Fallback: middle of own half (e.g. Cannon near the river)
                logger.warning(
                    "P%d: pending play %s at (%.1f, %.1f) illegal — fallback to mid",
                    player, card, x, y,
                )
                mid_x = ARENA_WIDTH / 2
                mid_y = (RIVER_Y - 5.0) if player == 0 else (RIVER_Y + 5.0)
                if not self.play_card_at(player, card, mid_x, mid_y):
                    logger.warning(
                        "P%d: fallback play of %s also failed — card not played",
                        player, card,
                    )
            self._pending_play[player] = None
            return

        card, zone = self.action_to_card_zone(action, player)
        if card is not None and zone is not None:
            self.play_card(player, card, zone)

    def _random_bot_action(self, player: int) -> int:
        affordable = np.flatnonzero(self.valid_action_mask(player))[1:].tolist()
        if affordable and self._rng.random() < 0.35:
            return self._rng.choice(affordable)
        return 0

    def _regen_elixir(self) -> None:
        rate = ELIXIR_PER_SECOND * (2 if self.time >= DOUBLE_ELIXIR_TIME else 1)
        gain = rate * TICK_DT
        for p in range(2):
            self.elixir[p] = min(MAX_ELIXIR, self.elixir[p] + gain)

    def _all_combatants(self) -> list[Troop]:
        return [t for t in self.troops if t.alive] + [t for t in self.towers if t.alive]

    def _nearest_bridge_x(self, x: float) -> float:
        if abs(x - BRIDGE_LANE_X[0]) <= abs(x - BRIDGE_LANE_X[1]):
            return BRIDGE_LANE_X[0]
        return BRIDGE_LANE_X[1]

    def _in_river_water(self, y: float) -> bool:
        return (RIVER_Y - RIVER_HALF_WIDTH) < y < (RIVER_Y + RIVER_HALF_WIDTH)

    def _on_bridge_lane(self, troop: Troop) -> bool:
        if not self._in_river_water(troop.y):
            return False
        return any(abs(troop.x - bx) <= BRIDGE_HALF_WIDTH for bx in BRIDGE_LANE_X)

    @staticmethod
    def _river_between(y: float, dest_y: float) -> bool:
        """Whether going from y to dest_y requires crossing the river (bank to bank).

        A unit within MIN_MOVE_DISTANCE of the far bank counts as across: it could not
        step onto that bank anyway and would otherwise freeze there.
        """
        low, high = min(y, dest_y), max(y, dest_y)
        near = BRIDGE_BANK_Y_OFFSET - MIN_MOVE_DISTANCE
        return low < RIVER_Y + near and high > RIVER_Y - near

    def _path_waypoint(self, troop: Troop, dest_x: float, dest_y: float) -> tuple[float, float]:
        """Diagonal to the bridge bank, straight over the bridge, then diagonal to the target."""
        if troop.crosses_river_freely or troop.speed <= 0:
            return dest_x, dest_y
        if not self._river_between(troop.y, dest_y):
            return dest_x, dest_y

        direction = 1.0 if dest_y > troop.y else -1.0
        near_bank_y = RIVER_Y - direction * BRIDGE_BANK_Y_OFFSET
        far_bank_y = RIVER_Y + direction * BRIDGE_BANK_Y_OFFSET
        bx = self._nearest_bridge_x(troop.x)
        at_bank = direction * (troop.y - near_bank_y) >= -0.1
        if at_bank and abs(troop.x - bx) <= BRIDGE_HALF_WIDTH:
            return bx, far_bank_y
        return bx, near_bank_y

    def get_move_waypoint(self, troop: Troop, target: Troop) -> tuple[float, float]:
        return self._path_waypoint(troop, target.x, target.y)

    def _in_river_off_bridge(self, troop: Troop) -> bool:
        """Whether a ground unit stands in the water away from any bridge lane."""
        if troop.crosses_river_freely or troop.is_building:
            return False
        return self._in_river_water(troop.y) and not self._on_bridge_lane(troop)

    def _enforce_no_river_cut(self, troop: Troop) -> None:
        """Puts a unit back on the bank of the nearest bridge if it entered the river outside one."""
        if not self._in_river_off_bridge(troop):
            return

        bx = self._nearest_bridge_x(troop.x)
        troop.x = bx
        if troop.owner == 0:
            troop.y = RIVER_Y - RIVER_HALF_WIDTH - 0.2
        else:
            troop.y = RIVER_Y + RIVER_HALF_WIDTH + 0.2

    def _current_target(self, unit: Troop, find_new: Callable[[Troop], Optional[Troop]]) -> Optional[Troop]:
        """Keeps attacking the locked target until it dies or leaves range, then picks a new one."""
        locked = unit.locked_target
        if locked is not None and unit.can_target(locked) and unit.is_in_range(locked):
            return locked
        target = find_new(unit)
        unit.locked_target = target if target is not None and unit.is_in_range(target) else None
        return target

    def find_nearest_target(self, troop: Troop) -> Optional[Troop]:
        """Target only within sight range (~9.5 tiles in CR) — no cross-lane aggro from the other side."""
        best: Optional[Troop] = None
        best_dist = float("inf")
        sight = getattr(troop, "sight_range", DEFAULT_SIGHT_RANGE)

        for other in self._all_combatants():
            if not troop.can_target(other):
                continue
            dist = troop.distance_to(other)
            if dist > sight:
                continue
            if dist < best_dist:
                best_dist = dist
                best = other
        return best

    def get_push_waypoint(self, troop: Troop) -> tuple[float, float]:
        """Marches along the lane toward the enemy King Tower when nothing is in sight range."""
        kx, ky = TOWER_LAYOUT[1 - troop.owner]["King_Tower"]
        return self._path_waypoint(troop, kx, ky)

    def find_tower_target(self, tower: Troop) -> Optional[Troop]:
        """The tower shoots the nearest enemy in range (never a target out of range)."""
        best: Optional[Troop] = None
        best_dist = float("inf")
        for enemy in self.troops:
            if not enemy.alive or enemy.owner == tower.owner:
                continue
            if not tower.can_target(enemy):
                continue
            if not tower.is_in_range(enemy):
                continue
            dist = tower.distance_to(enemy)
            if dist < best_dist:
                best_dist = dist
                best = enemy
        return best

    def is_king_active(self, player: int) -> bool:
        """A king tower wakes up once it has taken damage or a princess tower of its side fell.

        Derived from tower state: HP never recovers and towers never return, so it is permanent.
        """
        own = [t for t in self.towers if t.owner == player]
        king = next(t for t in own if t.name == "King_Tower")
        return king.hp < king.max_hp or any(not t.alive for t in own)

    def _simulate_combat(self, dt: float) -> tuple[float, float]:
        """Returns (damage dealt by P0, damage taken by P0).

        Every unit decides from the same start-of-tick state, all hits land, and only then
        do units move and deaths take effect, so list order never decides a duel.
        """
        dealt_p0 = 0.0
        taken_p0 = 0.0

        attackers: list[tuple[Troop, Troop]] = []
        movers: list[tuple[Troop, Optional[tuple[float, float]]]] = []
        idlers: list[Troop] = []

        for troop in list(self.troops):
            if not troop.alive or not troop.tick_age(dt):
                continue
            if troop.is_deploying:
                troop.tick_deploy(dt)
                continue
            target = self._current_target(troop, self.find_nearest_target)
            if target is not None and troop.is_in_range(target):
                attackers.append((troop, target))
            elif target is not None and (not troop.is_building or troop.speed > 0):
                movers.append((troop, self.get_move_waypoint(troop, target)))
            elif target is None and troop.speed > 0 and not troop.is_building:
                movers.append((troop, self.get_push_waypoint(troop)))
            else:
                idlers.append(troop)

        for tower in self.towers:
            if not tower.alive:
                continue
            if tower.name == "King_Tower" and not self.is_king_active(tower.owner):
                idlers.append(tower)
                continue
            target = self._current_target(tower, self.find_tower_target)
            if target is None:
                tower.stop_attacking()  # the next target starts a fresh first-hit delay
                idlers.append(tower)
            else:
                attackers.append((tower, target))

        for attacker, target in attackers:
            dmg = attacker.hit(target, dt)
            if dmg <= 0:
                continue
            self._apply_splash(attacker, target, dmg)
            self._record_hit(attacker, target, dmg)
            if attacker.owner == 0:
                dealt_p0 += dmg
            elif target.owner == 0:
                taken_p0 += dmg

        for troop, waypoint in movers:
            troop.move_towards_point(*waypoint, dt)
            self._enforce_no_river_cut(troop)
            troop.update_cooldown(dt)
        for unit in idlers:
            unit.update_cooldown(dt)

        for unit in self._all_combatants():
            if unit.hp <= 0:
                unit.alive = False

        self._resolve_collisions()
        return dealt_p0, taken_p0

    def _resolve_collisions(self) -> None:
        bodies = self._all_combatants()
        for _ in range(COLLISION_PASSES):
            for i, a in enumerate(bodies):
                for b in bodies[i + 1:]:
                    if (a.unit_type == "air") == (b.unit_type == "air"):
                        self._separate(a, b)

    def _separate(self, a: Troop, b: Troop) -> None:
        dx, dy = b.x - a.x, b.y - a.y
        dist = sqrt(dx * dx + dy * dy)
        overlap = a.collision_radius + b.collision_radius - dist
        if overlap <= 0:
            return
        nx, ny = (dx / dist, dy / dist) if dist > 1e-9 else (1.0, 0.0)
        # Heavier units give way less; a body without mass (building, tower) never moves.
        give_a = 1 / a.mass if a.mass > 0 else 0.0
        give_b = 1 / b.mass if b.mass > 0 else 0.0
        total = give_a + give_b
        if total == 0:
            return
        self._shove(a, -nx * overlap * give_a / total, -ny * overlap * give_a / total)
        self._shove(b, nx * overlap * give_b / total, ny * overlap * give_b / total)

    def _shove(self, unit: Troop, dx: float, dy: float) -> None:
        """Moves a unit by a push, keeping it in the arena and out of the river off a bridge."""
        if unit.mass <= 0:
            return
        r = unit.collision_radius
        old = unit.x, unit.y
        unit.x = min(max(unit.x + dx, r), ARENA_WIDTH - r)
        unit.y = min(max(unit.y + dy, r), ARENA_LENGTH - r)
        if self._in_river_off_bridge(unit):
            unit.x, unit.y = old

    def _apply_splash(self, attacker: Troop, target: Troop, damage: float) -> None:
        """Splash attackers also hit every other enemy the attacker can target near the target."""
        if attacker.splash_radius <= 0:
            return
        for other in self._all_combatants():
            if other is target or not attacker.can_target(other):
                continue
            if other.distance_to(target) <= attacker.splash_radius:
                other.hp = max(0.0, other.hp - damage)

    def _record_hit(self, attacker: Troop, target: Troop, damage: float) -> None:
        self.hit_events.append(
            HitEvent(
                attacker.owner,
                attacker.x,
                attacker.y,
                target.x,
                target.y,
                damage,
                ranged=attacker.range > RANGED_MIN_RANGE,
            )
        )

    def _remove_dead(self) -> None:
        self.troops = [t for t in self.troops if t.alive]

    @property
    def crowns(self) -> list[int]:
        """Crowns per player: 1 per enemy princess tower destroyed, 3 once the enemy king is down."""
        crowns = [0, 0]
        for tower in self.towers:
            if tower.alive:
                continue
            crowns[1 - tower.owner] += KING_CROWNS if tower.name == "King_Tower" else 1
        return [min(c, KING_CROWNS) for c in crowns]  # a king kill plus princess towers still caps at 3

    @property
    def phase(self) -> str:
        """Match phase: "regulation", "double_elixir" (last minute of regulation) or "overtime"."""
        if self.time >= REGULATION_TIME:
            return "overtime"
        return "double_elixir" if self.time >= DOUBLE_ELIXIR_TIME else "regulation"

    @property
    def clock(self) -> float:
        """Seconds left in the current period: regulation, or overtime once it has started."""
        end = REGULATION_TIME + OVERTIME_TIME if self.phase == "overtime" else REGULATION_TIME
        return max(0.0, end - self.time)

    def _check_end(self) -> tuple[bool, bool]:
        crowns = self.crowns
        if max(crowns) >= KING_CROWNS:
            self._end_match(crowns)
            return True, False
        if self.time < REGULATION_TIME:
            return False, False
        if crowns[0] != crowns[1]:
            # End of regulation, or the first crown of overtime (sudden death).
            self._end_match(crowns)
            return False, True
        if self.time >= REGULATION_TIME + OVERTIME_TIME:
            self._end_match(crowns)
            return False, True
        return False, False

    def _end_match(self, crowns: list[int]) -> None:
        """More crowns wins; equal crowns fall to the tiebreak (weakest tower), then a draw."""
        self.done = True
        score = crowns
        if crowns[0] == crowns[1] and crowns[0] < KING_CROWNS:  # both kings down is a draw
            score = self._weakest_tower_hp()
        if score[0] != score[1]:
            self.winner = 0 if score[0] > score[1] else 1

    def _weakest_tower_hp(self) -> list[float]:
        return [min(t.hp for t in self.towers if t.owner == p and t.alive) for p in (0, 1)]

    def get_observation(self, player: int) -> np.ndarray:
        """State vector from `player`'s perspective.

        For player=1 the view is mirrored: own towers first and depth measured from its own
        back edge (y' = L - y, x unchanged), the same way as the deploy zones, so a model
        trained as P0 can also play as P1 and left stays left.
        """
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        enemy = 1 - player
        obs[0] = self.elixir[player] / MAX_ELIXIR
        obs[1] = self.elixir[enemy] / MAX_ELIXIR
        obs[2] = (self.elixir[player] - self.elixir[enemy]) / MAX_ELIXIR
        obs[3] = min(1.0, self.time / REGULATION_TIME)

        towers = [t for t in self.towers if t.owner == player]
        towers += [t for t in self.towers if t.owner == enemy]
        for i, tower in enumerate(towers):
            obs[OBS_TOWER_BASE + i] = tower.hp / tower.max_hp if tower.max_hp else 0.0

        for slot, card_name in enumerate(self.get_hand(player)):
            obs[OBS_HAND_BASE + slot * DECK_SIZE + CARD_TO_ID[card_name]] = 1.0
        obs[OBS_NEXT_BASE + CARD_TO_ID[self.get_next_card(player)]] = 1.0

        recent = self.play_history[enemy][-PLAY_HISTORY_LEN:]
        recent = [None] * (PLAY_HISTORY_LEN - len(recent)) + recent  # missing plays come first
        for i, card_name in enumerate(recent):
            card_id = NO_PLAY if card_name is None else CARD_TO_ID[card_name]
            obs[OBS_PLAYS_BASE + i * (DECK_SIZE + 1) + card_id] = 1.0

        grid = obs[OBS_GRID_BASE:].reshape(2, NUM_LANES, DEPTH_BINS, len(UNIT_CARDS))
        for troop in self.troops:
            if not troop.alive or troop.name not in UNIT_CARDS:
                continue
            view_y = troop.y if player == 0 else ARENA_LENGTH - troop.y
            lane = int(troop.x >= ARENA_WIDTH / 2)
            depth = min(max(int(view_y // DEPTH_BIN_ROWS), 0), DEPTH_BINS - 1)
            cell = (int(troop.owner != player), lane, depth, UNIT_CARDS.index(troop.name))
            grid[cell] = min(1.0, grid[cell] + troop.hp / troop.max_hp)

        return obs

    def valid_action_mask(self, player: int = 0) -> np.ndarray:
        """Legal action mask: noop + (card, zone) when the card is affordable and the zone accepts its type."""
        mask = np.zeros(NUM_ACTIONS, dtype=np.bool_)
        mask[0] = True
        elixir = self.elixir[player]
        for slot, card_name in enumerate(self.get_hand(player)):
            if elixir < cards_dic[card_name]["elisir"]:
                continue
            for zone_idx in range(NUM_ZONES):
                if self.zone_allows(player, card_name, zone_idx):
                    mask[1 + slot * NUM_ZONES + zone_idx] = True
        return mask

    def render(self) -> None:
        grid = [["." for _ in range(int(ARENA_WIDTH))] for _ in range(int(ARENA_LENGTH))]
        for t in self.towers:
            if t.alive:
                gx, gy = int(t.x), int(t.y)
                if 0 <= gx < int(ARENA_WIDTH) and 0 <= gy < int(ARENA_LENGTH):
                    grid[gy][gx] = "K" if t.name == "King_Tower" else "T"
        for troop in self.troops:
            if troop.alive:
                gx, gy = int(troop.x), int(troop.y)
                if 0 <= gx < int(ARENA_WIDTH) and 0 <= gy < int(ARENA_LENGTH):
                    grid[gy][gx] = troop.name[0]
        print(f"t={self.time:.1f}s  elixir={self.elixir[0]:.1f}/{self.elixir[1]:.1f}")
        for row in reversed(grid):
            print("".join(row))


def run_random_episode(steps: int = 500, seed: int = 0) -> None:
    env = Board(seed=seed)
    obs = env.reset(seed=seed)
    for _ in range(steps):
        action = env.np_rng.integers(0, NUM_ACTIONS)
        result = env.step(action)
        if result.terminated or result.truncated:
            break
    print(f"Match over: winner={env.winner}, steps={env.time:.1f}s")


if __name__ == "__main__":
    run_random_episode()
