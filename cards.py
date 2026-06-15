from __future__ import annotations

from math import sqrt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from board import Board

# Zasięgi i prędkości w jednostkach planszy (metry uproszczone).
clash_royale_ranges = {
    "Melee_Short": 0.8,
    "Melee_Medium": 1.2,
    "Melee_Long": 1.6,
    "Ranged_Short": 4.5,
    "Ranged_Medium": 5.5,
    "Ranged_Long": 6.5,
}

# Zasięg wykrywania celu (jak w CR — bez tego jednostki „widzą” całą mapę).
DEFAULT_SIGHT_RANGE = 9.5

# Kafelki na sekundę (arena 18×32); poprzednie wartości ~×30 za szybkie.
clash_royale_speeds = {
    "slow": 1.2,
    "medium": 1.8,
    "fast": 2.6,
    "very_fast": 3.5,
}

cards_dic = {
    "Knight": {
        "damage": 293,
        "hp": 2566,
        "targets": ["ground"],
        "range": "Melee_Medium",
        "damage/sec": 244,
        "hit_speed": 1.2,
        "speed": "medium",
        "elisir": 3,
        "life_span": 0,
        "type": "ground",
    },
    "Giant": {
        "damage": 253,
        "hp": 3968,
        "targets": ["buildings"],
        "range": "Melee_Medium",
        "damage/sec": 168,
        "hit_speed": 1.5,
        "speed": "slow",
        "elisir": 5,
        "life_span": 0,
        "type": "ground",
    },
    "Cannon": {
        "damage": 167,
        "hp": 682,
        "targets": ["ground"],
        "range": 5.5,
        "damage/sec": 167,
        "hit_speed": 1.0,
        "first_hit_speed": 1.0,
        "deploy_time": 1.0,
        "speed": 0,
        "elisir": 3,
        "life_span": 30,
        "type": "building",
    },
    "Musketeer": {
        "damage": 238,
        "hp": 760,
        "targets": ["ground", "air"],
        "range": 6.0,
        "damage/sec": 238,
        "hit_speed": 1.0,
        "speed": "medium",
        "elisir": 4,
        "life_span": 0,
        "type": "ground",
    },
    "Hog_Rider": {
        "damage": 264,
        "hp": 1408,
        "targets": ["buildings"],
        "range": "Melee_Medium",
        "damage/sec": 165,
        "hit_speed": 1.6,
        "first_hit_speed": 0.6,
        "deploy_time": 1.0,
        "speed": "very_fast",
        "elisir": 4,
        "life_span": 0,
        "type": "ground",
        "jumps_river": True,
    },
    "Fireball": {
        "damage": 575,
        "radius": 2.5,
        "elisir": 4,
        "type": "spell",
    },
    "Tower": {
        "damage": 158,
        "hp": 4424,
        "targets": ["ground", "air"],
        "range": 7.5,
        "damage/sec": 197,
        "hit_speed": 0.8,
        "speed": 0,
        "elisir": 0,
        "life_span": 0,
        "type": "building",
    },
    "King_Tower": {
        "damage": 158,
        "hp": 7032,
        "targets": ["ground", "air"],
        "range": 7.0,
        "damage/sec": 197,
        "hit_speed": 1.0,
        "speed": 0,
        "elisir": 0,
        "life_span": 0,
        "type": "building",
    },
}

PLAYABLE_CARDS = (
    "Knight",
    "Giant",
    "Cannon",
    "Musketeer",
    "Hog_Rider",
    "Fireball",
)
CARD_TO_ID = {name: i for i, name in enumerate(PLAYABLE_CARDS)}


class Troop:
    """Jednostka lub budynek na planszy."""

    def __init__(self, name: str, owner: int, *, is_tower: bool = False):
        self.stats = cards_dic.get(name, {})
        self.name = name
        self.owner = owner
        self.is_tower = is_tower
        self.max_hp = self.stats.get("hp", 0)
        self.hp = self.max_hp
        self.damage = self.stats.get("damage", 0)
        self.targets = list(self.stats.get("targets", []))
        self.damage_sec = self.stats.get("damage/sec", 0)
        self.hit_speed = self.stats.get("hit_speed", 0)
        self.first_hit_speed = self.stats.get("first_hit_speed", self.hit_speed)
        self.deploy_time = self.stats.get("deploy_time", 0.0)
        self.sight_range = self.stats.get("sight_range", DEFAULT_SIGHT_RANGE)
        self.elisir = self.stats.get("elisir", 0)
        self.life_span = self.stats.get("life_span", 0)
        self.unit_type = self.stats.get("type", "ground")

        self.range = self.stats.get("range", 0)
        if self.range in clash_royale_ranges:
            self.range = clash_royale_ranges[self.range]

        self.speed = self.stats.get("speed", 0)
        if self.speed in clash_royale_speeds:
            self.speed = clash_royale_speeds[self.speed]

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

    def place(self, position: tuple[float, float]) -> None:
        self.x, self.y = position

    def distance_to(self, other: Troop) -> float:
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def distance_to_xy(self, x: float, y: float) -> float:
        return sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

    def is_in_range(self, target: Troop) -> bool:
        return self.distance_to(target) <= self.range

    def can_target(self, target: Troop) -> bool:
        if not target.alive or target.owner == self.owner:
            return False
        if target.is_tower:
            return "buildings" in self.targets or "ground" in self.targets
        if target.is_building:
            return "buildings" in self.targets
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
        if distance <= 0.05:
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

    def can_attack(self) -> bool:
        return self.age >= self.deploy_time

    def hit(self, target: Troop, dt: float) -> float:
        """Atakuje cel; zwraca zadane obrażenia (0 jeśli brak ataku)."""
        self.update_cooldown(dt)
        if not self.can_attack() or not self.is_in_range(target):
            self._attack_winding = False
            return 0.0

        if not self._attack_winding:
            self._attack_winding = True
            if self.cooldown <= 0:
                self.cooldown = self.first_hit_speed
            return 0.0

        if self.cooldown > 0:
            return 0.0

        target.hp -= self.damage
        if target.hp <= 0:
            target.hp = 0
            target.alive = False
        self.cooldown = self.hit_speed
        return float(self.damage)

    def tick_age(self, dt: float) -> bool:
        """Zwraca False, gdy budynek tymczasowy (np. Cannon) wygasł."""
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
