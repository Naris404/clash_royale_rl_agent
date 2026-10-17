from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cr_rl.game.board import Board

# Target detection range (as in CR — without it units "see" the whole map).
DEFAULT_SIGHT_RANGE = 9.5
MIN_MOVE_DISTANCE = 0.05  # closer than this to a waypoint counts as arrived

# Combat stats live in one data file that records its source and card level.
CARD_STATS_PATH = Path(__file__).with_name("card_stats.json")
cards_dic: dict[str, dict] = json.loads(CARD_STATS_PATH.read_text(encoding="utf-8"))["cards"]

# The fixed mirror Deck both players use (ADR-0004); the order defines the card ids.
DECK = (
    "Knight",
    "Giant",
    "Hog_Rider",
    "Musketeer",
    "Cannon",
    "Fireball",
    "Arrows",
    "Baby_Dragon",
)
CARD_TO_ID = {name: i for i, name in enumerate(DECK)}


class Troop:
    """Unit or building on the board."""

    def __init__(self, name: str, owner: int, *, is_tower: bool = False):
        self.stats = cards_dic.get(name, {})
        self.name = name
        self.owner = owner
        self.is_tower = is_tower
        self.max_hp = self.stats.get("hp", 0)
        self.hp = self.max_hp
        self.damage = self.stats.get("damage", 0)
        self.targets = list(self.stats.get("targets", []))
        self.hit_speed = self.stats.get("hit_speed", 0)
        self.first_hit_speed = self.stats.get("first_hit_speed", self.hit_speed)
        self.deploy_remaining = self.stats.get("deploy_time", 0.0)
        self.sight_range = self.stats.get("sight_range", DEFAULT_SIGHT_RANGE)
        self.elisir = self.stats.get("elisir", 0)
        self.life_span = self.stats.get("life_span", 0)
        self.unit_type = self.stats.get("type", "ground")
        self.range = self.stats.get("range", 0)
        self.splash_radius = self.stats.get("splash_radius", 0)
        self.speed = self.stats.get("speed", 0)
        self.collision_radius = self.stats.get("collision_radius", 0.0)
        self.mass = self.stats.get("mass", 0)

        self.x = 0.0
        self.y = 0.0
        self.cooldown = 0.0
        self.age = 0.0
        self.alive = True
        self.jumps_river = bool(self.stats.get("jumps_river", False))
        self._attack_winding = False

    @property
    def is_building(self) -> bool:
        return self.unit_type == "building" or self.is_tower

    @property
    def crosses_river_freely(self) -> bool:
        return self.jumps_river or self.unit_type == "air"

    def place(self, position: tuple[float, float]) -> None:
        self.x, self.y = position

    def distance_to(self, other: Troop) -> float:
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def distance_to_xy(self, x: float, y: float) -> float:
        return sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

    def is_in_range(self, target: Troop) -> bool:
        """Range is measured edge to edge, so both sides of a fight reach each other equally."""
        return self.distance_to(target) - self.collision_radius - target.collision_radius <= self.range

    def can_target(self, target: Troop) -> bool:
        if not target.alive or target.owner == self.owner:
            return False
        if target.is_building:
            return not self.is_building and ("buildings" in self.targets or "ground" in self.targets)
        if target.unit_type == "air":
            return "air" in self.targets
        return "ground" in self.targets

    def update_cooldown(self, dt: float) -> None:
        if self.cooldown > 0:
            self.cooldown -= dt

    def move_towards_point(self, x: float, y: float, dt: float) -> None:
        if self.speed <= 0:
            return

        dx = x - self.x
        dy = y - self.y
        distance = sqrt(dx * dx + dy * dy)
        if distance <= MIN_MOVE_DISTANCE:
            return

        step = min(self.speed * dt, distance)
        if step >= distance - 1e-6:
            self.x = x
            self.y = y
            return
        self.x += dx / distance * step
        self.y += dy / distance * step

    def move_towards(self, target: Troop, dt: float) -> None:
        if target is None:
            return
        self.move_towards_point(target.x, target.y, dt)

    @property
    def is_deploying(self) -> bool:
        return self.deploy_remaining > 0

    def tick_deploy(self, dt: float) -> None:
        # Rounded so ten 0.1 s ticks end a 1.0 s deploy exactly.
        self.deploy_remaining = round(max(0.0, self.deploy_remaining - dt), 6)

    def hit(self, target: Troop, dt: float) -> float:
        """Attacks the target; returns the damage dealt (0 if no attack)."""
        self.update_cooldown(dt)
        if not self.is_in_range(target):
            self._attack_winding = False
            return 0.0

        if not self._attack_winding:
            self._attack_winding = True
            if self.cooldown <= 0:
                self.cooldown = self.first_hit_speed
            return 0.0

        if self.cooldown > 0:
            return 0.0

        target.hp = max(0.0, target.hp - self.damage)  # the board marks deaths at end of tick
        self.cooldown = self.hit_speed
        return float(self.damage)

    def tick_age(self, dt: float) -> bool:
        """Returns False when a temporary building (e.g. Cannon) has expired."""
        self.age += dt
        if self.life_span <= 0:
            return True
        if self.age >= self.life_span:
            self.alive = False
            return False
        return True


class Tower(Troop):
    def __init__(self, name: str, owner: int):
        super().__init__(name, owner, is_tower=True)
