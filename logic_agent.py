from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from board import (
    ARENA_LENGTH,
    ARENA_WIDTH,
    BRIDGE_LANE_X,
    RIVER_Y,
    TOWER_LAYOUT,
    Board,
)

from cards import PLAYABLE_CARDS, Troop, cards_dic

ARENA_CENTER_X = 9.0
# Cannon: 5 kafelków od rzeki (P0: y=11 zamiast y=6 w strefach rzutu).
CANNON_RIVER_DISTANCE = 5.0

BUILDING_ATTACKERS = frozenset({"Giant", "Hog_Rider"})
RANGED_UNITS = frozenset({"Musketeer", "Wizard", "Witch"})
TANK_UNITS = frozenset({"Giant"})
FIREBALL_RADIUS = 2.5
FIREBALL_CANNON_ELIXIR = 8.0

ELIXIR_FULL = 10.0
ELIXIR_PUSH = 9.0


@dataclass
class LogicAgent:
    """
    Bot oparty na regułach (priorytet od góry — pierwsza pasująca wygrywa).
    Używa board.set_pending_play() do stawiania kart w dowolnym punkcie strefy rzutu.
    """

    deck: list[str] = field(default_factory=lambda: list(PLAYABLE_CARDS))
    _combo_followup: Optional[tuple[str, float, float]] = None

    def choose_action(self, board: Board, player: int = 0) -> int:
        play = self._decide_play(board, player)
        board.set_pending_play(player, play)
        return 0

    def reset(self) -> None:
        self._combo_followup = None

    # --- decyzja ---

    def _decide_play(
        self, board: Board, player: int
    ) -> Optional[tuple[str, float, float]]:
        if self._combo_followup is not None:
            play = self._combo_followup
            self._combo_followup = None
            if self._can_play(board, player, play[0], offensive=True):
                return self._clamp_play(player, play)
            self._combo_followup = None

        elixir = board.elixir[player]
        mine = _my_troops(board, player)
        enemies = _enemy_troops(board, player)

        # Wrogi Hog — Cannon (wysoki priorytet, koszt 3).
        for enemy in enemies:
            if enemy.name == "Hog_Rider":
                play = _cannon_vs_hog(board, player, enemy)
                if play and self._can_play(board, player, "Cannon", defensive=True):
                    return play

        # Giant na rzece — Cannon.
        for enemy in enemies:
            if enemy.name == "Giant" and _at_river(enemy):
                play = _cannon_vs_building_attacker(board, player, enemy)
                if play and self._can_play(board, player, "Cannon", defensive=True):
                    return play

        # Fireball: Musketeer, skupisko 2+, spokojna gra + wrogi Cannon.
        fireball = _try_fireball_play(board, player, elixir, enemies)
        if fireball and self._can_play(board, player, "Fireball", defensive=True):
            return self._clamp_play(player, fireball)

        # 9. Oszczędzaj eliksir przy braku zagrożenia.
        if elixir < 3.0 and not _enemies_in_threat_zone(board, player):
            return None

        # 5. Tank + ranged support — najpierw Cannon, potem Musketeer (kolejna tura).
        tank_ranged = _tank_with_ranged_support(enemies)
        if tank_ranged is not None:
            tank, ranged = tank_ranged
            if self._can_play(board, player, "Cannon", defensive=True):
                cannon = _cannon_vs_building_attacker(board, player, tank)
                if cannon:
                    self._combo_followup = _musketeer_safe_support(board, player, ranged)
                    return cannon

        # 6. Wroga jednostka bije wieżę przy < 3 eliksiru — Knight na nią.
        tower_threat = _enemy_attacking_my_tower(board, player)
        if tower_threat is not None and elixir < 3.0:
            play = _knight_on_unit(board, player, tower_threat)
            if play and self._can_play(board, player, "Knight", defensive=True):
                return play

        # 4. Wroga jednostka dystansowa właśnie po Twojej stronie rzeki — Knight na nią.
        for enemy in enemies:
            if _ranged_just_crossed(enemy, player):
                play = _knight_on_unit(board, player, enemy)
                if play and self._can_play(board, player, "Knight", defensive=True):
                    return play

        on_my_side = [e for e in enemies if _on_my_side(e, player)]

        # Hog: przeciwnik ma Cannon — rush drugim mostem.
        for enemy in enemies:
            if enemy.name == "Cannon":
                play = _hog_opposite_lane(board, player, enemy)
                if play and self._can_play(board, player, "Hog_Rider", offensive=True):
                    return play

        # 6 (pełniejszy eliksir) — obrona wieży Knightem.
        if tower_threat is not None:
            play = _knight_on_unit(board, player, tower_threat)
            if play and self._can_play(board, player, "Knight", defensive=True):
                return play

        # 7. Własny Giant / Hog przy moście + >= 4 — wsparcie dystansowe.
        for ally in mine:
            if ally.name == "Giant" and _near_bridge(ally, player):
                play = _musketeer_behind_unit(player, ally, dist=3.5)
                if play and self._can_play(board, player, "Musketeer", offensive=True):
                    return play
        # 8. Knight po obronie przechodzi most + >= 4 — Musketeer za Knightem.
        for ally in mine:
            if (
                ally.name == "Knight"
                and ally.max_hp > 0
                and ally.hp / ally.max_hp > 0.5
                and _crossing_bridge_toward_enemy(ally, player)
            ):
                play = _musketeer_behind_unit(player, ally, dist=3.5)
                if play and self._can_play(board, player, "Musketeer", offensive=True):
                    return play

        # Hog rush: >= 4 eliksiru, brak własnego Hoga, mało zagrożenia — słaba alejka.
        if (
            elixir >= 4.0
            and not any(t.name == "Hog_Rider" for t in mine)
            and not _enemies_in_threat_zone(board, player)
        ):
            play = _hog_rush_lane(board, player)
            if play and self._can_play(board, player, "Hog_Rider", offensive=True):
                return play

        # 1. Push: >= 9 eliksiru, brak własnych wojsk — Giant za wieżą w stronę słabszej wieży.
        if elixir >= ELIXIR_PUSH and len(mine) == 0:
            play = _giant_push_lane(board, player)
            if play and self._can_play(board, player, "Giant", offensive=True):
                return play

        # 2. Pełny eliksir, brak wrogów — Musketeer za King Tower.
        if elixir >= ELIXIR_FULL - 0.05 and len(enemies) == 0:
            if self._can_play(board, player, "Musketeer", offensive=True):
                return _musketeer_behind_king(player)

        return None

    def _can_play(
        self,
        board: Board,
        player: int,
        card: str,
        *,
        offensive: bool = False,
        defensive: bool = False,
    ) -> bool:
        if not board.card_in_hand(player, card):
            return False
        cost = cards_dic[card]["elisir"]
        elixir = board.elixir[player]
        if elixir < cost:
            return False
        if defensive:
            return True
        # Nie wydawaj ostatniego eliksiru na atak bez zapasu.
        if offensive and elixir < 3.0:
            return False
        if offensive and elixir - cost < 1.0:
            return False
        return True

    def _clamp_play(
        self, player: int, play: tuple[str, float, float]
    ) -> tuple[str, float, float]:
        card, x, y = play
        x = max(1.0, min(ARENA_WIDTH - 1.0, x))
        if card == "Fireball":
            y = max(2.0, min(ARENA_LENGTH - 2.0, y))
            return card, x, y
        if player == 0:
            y = max(2.0, min(RIVER_Y - 0.5, y))
        else:
            y = max(RIVER_Y + 0.5, min(ARENA_LENGTH - 2.0, y))
        return card, x, y


# --- helpers stanu planszy ---


def _my_troops(board: Board, player: int) -> list[Troop]:
    return [t for t in board.troops if t.alive and t.owner == player]


def _enemy_troops(board: Board, player: int) -> list[Troop]:
    return [t for t in board.troops if t.alive and t.owner != player]


def _my_towers(board: Board, player: int) -> list[Troop]:
    return [t for t in board.towers if t.alive and t.owner == player]


def _enemy_towers(board: Board, player: int) -> list[Troop]:
    return [t for t in board.towers if t.alive and t.owner != player]


def _forward(player: int) -> float:
    return 1.0 if player == 0 else -1.0


def _at_river(unit: Troop, margin: float = 1.2) -> bool:
    return abs(unit.y - RIVER_Y) <= margin


def _near_bridge(unit: Troop, player: int, margin: float = 2.5) -> bool:
    if player == 0:
        return 14.0 <= unit.y <= 18.5
    return 13.5 <= unit.y <= 18.0


def _on_my_side(unit: Troop, player: int) -> bool:
    if player == 0:
        return unit.y < RIVER_Y
    return unit.y > RIVER_Y


def _ranged_unit(unit: Troop) -> bool:
    if unit.name in RANGED_UNITS:
        return True
    return isinstance(unit.range, (int, float)) and unit.range >= 5.0


def _ranged_just_crossed(enemy: Troop, player: int) -> bool:
    """Wroga jednostka dystansowa po Twojej stronie, blisko rzeki (świeży przeskok)."""
    if not _ranged_unit(enemy):
        return False
    if not _on_my_side(enemy, player):
        return False
    if player == 0:
        return RIVER_Y - 4.0 <= enemy.y < RIVER_Y
    return RIVER_Y < enemy.y <= RIVER_Y + 4.0


def _crossing_bridge_toward_enemy(ally: Troop, player: int) -> bool:
    if player == 0:
        return 15.5 <= ally.y <= 19.0
    return 13.0 <= ally.y <= 16.5


def _enemies_in_threat_zone(board: Board, player: int, radius: float = 7.0) -> bool:
    for enemy in _enemy_troops(board, player):
        for tower in _my_towers(board, player):
            if enemy.distance_to(tower) <= radius:
                return True
    return False


def _enemy_attacking_my_tower(board: Board, player: int) -> Optional[Troop]:
    best: Optional[Troop] = None
    best_dist = float("inf")
    for enemy in _enemy_troops(board, player):
        target = board.find_nearest_target(enemy)
        if target is None or not target.is_tower or target.owner != player:
            continue
        d = enemy.distance_to(target)
        if d < best_dist:
            best_dist = d
            best = enemy
    return best


def _tank_with_ranged_support(
    enemies: list[Troop],
) -> Optional[tuple[Troop, Troop]]:
    tanks = [e for e in enemies if e.name in TANK_UNITS]
    ranged = [e for e in enemies if _ranged_unit(e)]
    if not tanks or not ranged:
        return None

    for tank in tanks:
        for r in ranged:
            if _ranged_behind_tank(tank, r):
                return tank, r
    return None


def _ranged_behind_tank(tank: Troop, ranged: Troop) -> bool:
    """Ranged jest po stronie wroga (dalej od nas) względem tanka."""
    if tank.owner == 0:
        return ranged.y > tank.y and ranged.distance_to(tank) <= 6.0
    return ranged.y < tank.y and ranged.distance_to(tank) <= 6.0


def _weakest_enemy_princess(board: Board, player: int) -> Optional[Troop]:
    enemy = 1 - player
    princesses = [
        t for t in _enemy_towers(board, player) if t.name == "Tower" and t.alive
    ]
    if not princesses:
        return None
    return min(princesses, key=lambda t: t.hp)


def _lane_for_tower(tower: Troop) -> str:
    if tower.x < ARENA_WIDTH / 2:
        return "Tower_L"
    return "Tower_R"


def _giant_push_lane(board: Board, player: int) -> Optional[tuple[str, float, float]]:
    target = _weakest_enemy_princess(board, player)
    if target is None:
        lane = "Tower_L"
    else:
        lane = _lane_for_tower(target)

    tx, ty = TOWER_LAYOUT[player][lane]
    # Za własną wieżą, w stronę przeciwnika (push).
    x, y = tx, ty + 3.0 * _forward(player)
    return "Giant", x, y


def _musketeer_behind_king(player: int) -> tuple[str, float, float]:
    kx, ky = TOWER_LAYOUT[player]["King_Tower"]
    x = kx
    y = ky - 2.0 * _forward(player)
    return "Musketeer", x, y


def _musketeer_behind_unit(
    player: int, unit: Troop, dist: float = 3.5
) -> tuple[str, float, float]:
    x = unit.x
    y = unit.y - dist * _forward(player)
    return "Musketeer", x, y


def _musketeer_safe_support(
    board: Board, player: int, ranged: Troop
) -> tuple[str, float, float]:
    """Musketeer blisko własnej wieży po przeciwnej stronie niż akcja."""
    towers = _my_towers(board, player)
    if not towers:
        kx, ky = TOWER_LAYOUT[player]["King_Tower"]
        return "Musketeer", kx, ky - 1.5 * _forward(player)

    if ranged.x < ARENA_WIDTH / 2:
        tower = next((t for t in towers if t.x > ARENA_WIDTH / 2), towers[0])
    else:
        tower = next((t for t in towers if t.x < ARENA_WIDTH / 2), towers[0])

    x = tower.x
    y = tower.y - 2.0 * _forward(player)
    return "Musketeer", x, y


def _cannon_river_y(player: int) -> float:
    if player == 0:
        return RIVER_Y - CANNON_RIVER_DISTANCE
    return RIVER_Y + CANNON_RIVER_DISTANCE


def _cannon_vs_building_attacker(
    board: Board, player: int, enemy: Troop
) -> Optional[tuple[str, float, float]]:
    """Cannon 5 pól od rzeki, w osi akcji wroga."""
    x = max(3.0, min(ARENA_WIDTH - 3.0, enemy.x))
    return "Cannon", x, _cannon_river_y(player)


def _fireball_on_unit(unit: Troop) -> tuple[str, float, float]:
    return "Fireball", unit.x, unit.y


def _fireball_best_cluster(
    enemies: list[Troop],
    *,
    min_count: int = 2,
    radius: float = FIREBALL_RADIUS,
) -> Optional[tuple[str, float, float]]:
    if len(enemies) < min_count:
        return None

    best_play: Optional[tuple[str, float, float]] = None
    best_hits = 0
    for anchor in enemies:
        cluster = [e for e in enemies if e.distance_to(anchor) <= radius]
        if len(cluster) < min_count:
            continue
        cx = sum(e.x for e in cluster) / len(cluster)
        cy = sum(e.y for e in cluster) / len(cluster)
        hits = sum(1 for e in enemies if e.distance_to_xy(cx, cy) <= radius)
        if hits >= min_count and hits > best_hits:
            best_hits = hits
            best_play = ("Fireball", cx, cy)
    return best_play


def _is_quiet_board(board: Board, player: int) -> bool:
    """Brak walki: żadne wojsko nie naciera ani nie broni aktywnie (Cannon wroga nie liczy się)."""
    active_enemies = [
        e
        for e in _enemy_troops(board, player)
        if not (e.is_building and e.name == "Cannon")
    ]
    for enemy in active_enemies:
        if _on_my_side(enemy, player):
            return False
        for tower in _my_towers(board, player):
            if enemy.distance_to(tower) <= 7.0:
                return False
    for ally in _my_troops(board, player):
        if not _on_my_side(ally, player):
            return False
        if _near_bridge(ally, player) or _crossing_bridge_toward_enemy(ally, player):
            return False
    return True


def _try_fireball_play(
    board: Board,
    player: int,
    elixir: float,
    enemies: list[Troop],
) -> Optional[tuple[str, float, float]]:
    if not enemies:
        return None

    # Zawsze Fireball na Musketeer.
    for enemy in enemies:
        if enemy.name == "Musketeer":
            return _fireball_on_unit(enemy)

    # 2+ wrogów w promieniu jednego Fireballa.
    cluster = _fireball_best_cluster(enemies)
    if cluster is not None:
        return cluster

    # Spokój na planszy, >= 8 eliksiru, wrogi Cannon — wybij go.
    if elixir >= FIREBALL_CANNON_ELIXIR and _is_quiet_board(board, player):
        for enemy in enemies:
            if enemy.name == "Cannon":
                return _fireball_on_unit(enemy)

    return None


def _hog_on_bridge(player: int, lane: str) -> tuple[str, float, float]:
    """Hog zawsze na moście (własna strona rzeki, oś lewego lub prawego mostu)."""
    bx = BRIDGE_LANE_X[0] if lane == "Tower_L" else BRIDGE_LANE_X[1]
    if player == 0:
        y = RIVER_Y - 1.0
    else:
        y = RIVER_Y + 1.0
    return "Hog_Rider", bx, y


def _hog_rush_lane(board: Board, player: int) -> Optional[tuple[str, float, float]]:
    target = _weakest_enemy_princess(board, player)
    lane = _lane_for_tower(target) if target else "Tower_L"
    return _hog_on_bridge(player, lane)


def _hog_opposite_lane(
    board: Board, player: int, cannon: Troop
) -> Optional[tuple[str, float, float]]:
    """Cannon w centrum — Hog na drugi most."""
    lane = "Tower_R" if cannon.x < ARENA_CENTER_X else "Tower_L"
    return _hog_on_bridge(player, lane)


def _hog_is_attacking(board: Board, player: int, hog: Troop) -> bool:
    if _at_river(hog) or _on_my_side(hog, player):
        return True
    target = board.find_nearest_target(hog)
    return (
        target is not None
        and target.is_tower
        and target.owner == player
        and target.alive
    )


def _tower_under_hog_attack(board: Board, player: int, hog: Troop) -> Optional[Troop]:
    target = board.find_nearest_target(hog)
    if target and target.is_tower and target.owner == player and target.alive:
        return target
    threatened = _enemy_attacking_my_tower(board, player)
    if threatened == hog:
        for tower in _my_towers(board, player):
            if tower.alive and hog.distance_to(tower) <= 10.0:
                return tower
    return None


def _cannon_vs_hog(
    board: Board, player: int, hog: Troop
) -> tuple[str, float, float]:
    """
    Cannon: x = środek areny ± 1 kafelek w stronę wieży, którą Hog atakuje;
    y = 5 kafelków od rzeki (bliżej mostu niż domyślne strefy rzutu).
    """
    tower = _tower_under_hog_attack(board, player, hog)
    if tower is not None:
        if tower.x < ARENA_CENTER_X:
            x = ARENA_CENTER_X - 1.0
        elif tower.x > ARENA_CENTER_X:
            x = ARENA_CENTER_X + 1.0
        else:
            x = ARENA_CENTER_X
    elif hog.x < ARENA_CENTER_X:
        x = ARENA_CENTER_X - 1.0
    else:
        x = ARENA_CENTER_X + 1.0

    return "Cannon", x, _cannon_river_y(player)


def _knight_on_unit(
    board: Board, player: int, enemy: Troop
) -> Optional[tuple[str, float, float]]:
    """Knight w odległości 0–1 od wroga (w strefie rzutu)."""
    for dist in (0.0, 0.6, 1.0):
        for dx, dy in ((0, dist), (dist, 0), (-dist, 0), (0, -dist)):
            x = enemy.x + dx
            y = enemy.y + dy
            if board._in_deploy_zone(player, x, y):
                return "Knight", x, y

    # Wroga jednostka głęboko na naszej połowie — najbliższy punkt w strefie.
    x = max(1.0, min(ARENA_WIDTH - 1.0, enemy.x))
    if player == 0:
        y = max(2.0, min(RIVER_Y - 0.5, enemy.y))
    else:
        y = max(RIVER_Y + 0.5, min(ARENA_LENGTH - 2.0, enemy.y))
    return "Knight", x, y
