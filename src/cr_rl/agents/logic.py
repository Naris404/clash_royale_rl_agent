"""
Bot regułowy — decyzje przez scoring kandydatów (najwyższy wynik wygrywa).

Struktura:
  BattleContext  — snapshot stanu planszy
  PlayCandidate  — karta + pozycja + wynik + uzasadnienie
  _rules_*       — generatory kandydatów (obrona, czary, kontra, atak, cycle)
  LogicAgent     — kolejka combo + wybór najlepszego zagrania
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

from cr_rl.game.board import (
    ARENA_LENGTH,
    ARENA_WIDTH,
    BRIDGE_LANE_X,
    MATCH_TIME_LIMIT,
    RIVER_Y,
    TOWER_LAYOUT,
    Board,
)

from cr_rl.game.cards import PLAYABLE_CARDS, Troop, cards_dic

# --- stałe mety (progi eliksiru + skala punktów reguł) ---

ARENA_CENTER_X = 9.0
CANNON_RIVER_DISTANCE = 5.0
FIREBALL_DAMAGE = float(cards_dic["Fireball"]["damage"])
FIREBALL_RADIUS = float(cards_dic["Fireball"]["radius"])

BUILDING_ATTACKERS = frozenset({"Giant", "Hog_Rider"})
RANGED_UNITS = frozenset({"Musketeer", "Wizard", "Witch"})
TANK_UNITS = frozenset({"Giant"})

ELIXIR_FULL = 10.0
ELIXIR_PUSH = 9.0
ELIXIR_LEAK = 9.5
FIREBALL_CANNON_ELIXIR = 8.0

SCORE_CRITICAL = 100.0
SCORE_HIGH = 80.0
SCORE_MID = 60.0
SCORE_LOW = 40.0


# --- modele decyzji ---


@dataclass
class PlayCandidate:
    """Jeden możliwy ruch: karta, cel, wynik reguły, ewentualne kolejne tury."""
    card: str
    x: float
    y: float
    score: float
    reason: str
    followups: list[tuple[str, float, float]] = field(default_factory=list)  # combo na przyszłe tury


@dataclass
class BattleContext:
    """Snapshot planszy — reguły czytają tylko to, nie cały Board."""
    board: Board
    player: int
    elixir: float
    enemy_elixir: float
    time: float
    mine: list[Troop]
    enemies: list[Troop]
    on_my_side: list[Troop]  # wrogowie już na naszej połowie
    tower_threat: Optional[Troop]  # wróg celujący w naszą wieżę
    under_threat: bool  # wróg blisko naszych wież
    quiet: bool  # brak aktywnej walki — można pushować / cycle
    late_game: bool
    own_cannon_alive: bool
    open_lane: str  # Tower_L | Tower_R — słabsza / otwarta alejka wroga


RuleFn = Callable[["LogicAgent", BattleContext], list[PlayCandidate]]


@dataclass
class LogicAgent:
    """
    Bot regułowy z scoringiem i kolejką combo (2-3 tury z wyprzedzeniem).
  Używa board.set_pending_play() — dowolna pozycja na własnej połowie.
    """

    deck: list[str] = field(default_factory=lambda: list(PLAYABLE_CARDS))
    _combo_queue: deque[tuple[str, float, float]] = field(default_factory=deque)
    last_decision: str = ""

    def choose_action(self, board: Board, player: int = 0) -> int:
        play = self._decide_play(board, player)  # wybór karty+pozycji, nie indeksu akcji
        board.set_pending_play(player, play)  # gra odczyta to w step()
        return 0  # zawsze noop — rzeczywisty ruch idzie przez pending_play

    def reset(self) -> None:
        self._combo_queue.clear()
        self.last_decision = ""

    def _decide_play(
        self, board: Board, player: int
    ) -> Optional[tuple[str, float, float]]:
        if self._combo_queue:  # najpierw dokończ zaplanowany combo (np. Giant→Musk→Hog)
            card, x, y = self._combo_queue.popleft()
            if self._can_play(board, player, card, offensive=True, defensive=True):
                self.last_decision = f"combo: {card}"
                return self._clamp_play(player, (card, x, y))

        ctx = _build_context(board, player)
        candidates: list[PlayCandidate] = []

        # każda reguła dorzuca kandydatów; kolejność = priorytet przy remisie score
        for rule in (
            _rules_critical_defense,  # Hog/Cannon/Giant — najwyższe score
            _rules_spells,
            _rules_hog_counters,
            _rules_push_combos,
            _rules_tower_defense,
            _rules_lane_pressure,
            _rules_support_allies,
            _rules_elixir_cycle,  # najniższe score — tylko gdy nic ważniejszego
        ):
            candidates.extend(rule(self, ctx))

        if not candidates:
            self.last_decision = "pass"
            return None

        best = max(candidates, key=lambda c: c.score)  # wygrywa najwyższy score, nie pierwsza reguła
        if best.followups:
            self._combo_queue.extend(best.followups)  # zaplanuj następne tury z wyprzedzeniem

        self.last_decision = f"{best.card} ({best.score:.0f}): {best.reason}"
        return self._clamp_play(player, (best.card, best.x, best.y))

    def _can_play(  # filtr: karta w ręce, eliksir, tryb ofensywny vs defensywny
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
        if board.elixir[player] < cost:
            return False
        if defensive and not offensive:
            return True
        if offensive and board.elixir[player] < 3.0 and not defensive:
            return False
        if offensive and not defensive and board.elixir[player] - cost < 1.0:
            return False
        return True

    def _clamp_play(  # przytnij (x,y) do legalnej strefy deploy gracza
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

    def _add_if_legal(
        self,
        board: Board,
        player: int,
        candidates: list[PlayCandidate],
        candidate: PlayCandidate,
        *,
        offensive: bool = False,
        defensive: bool = False,
    ) -> None:
        if self._can_play(
            board, player, candidate.card, offensive=offensive, defensive=defensive
        ):
            candidates.append(candidate)


# --- reguły (scoring) ---


def _rules_critical_defense(  # Knight/Cannon na Hoga i tank+ranged — score ~100
    agent: LogicAgent, ctx: BattleContext
) -> list[PlayCandidate]:
    out: list[PlayCandidate] = []
    board, player = ctx.board, ctx.player

    for enemy in ctx.enemies:
        if enemy.name != "Hog_Rider":
            continue
        if _on_my_side(enemy, player) or _at_river(enemy):
            knight = _knight_on_unit(board, player, enemy)
            if knight:
                out.append(
                    PlayCandidate(
                        *knight,
                        SCORE_CRITICAL + 5,
                        "Knight blokuje Hoga na naszej połowie",
                    )
                )
            if not ctx.own_cannon_alive:
                cannon = _cannon_vs_hog(board, player, enemy)
                out.append(
                    PlayCandidate(
                        *cannon,
                        SCORE_CRITICAL,
                        "Cannon na ścieżkę Hoga",
                    )
                )

    tank_ranged = _tank_with_ranged_support(ctx.enemies)
    if tank_ranged and not ctx.own_cannon_alive:
        tank, ranged = tank_ranged
        cannon = _cannon_vs_building_attacker(board, player, tank)
        musk = _musketeer_safe_support(board, player, ranged)
        out.append(
            PlayCandidate(
                *cannon,
                SCORE_CRITICAL - 5,
                "Cannon na tanka + Musketeer w kolejce",
                followups=[musk],
            )
        )

    for enemy in ctx.enemies:
        if enemy.name == "Giant" and _at_river(enemy) and not ctx.own_cannon_alive:
            play = _cannon_vs_building_attacker(board, player, enemy)
            out.append(PlayCandidate(*play, SCORE_HIGH, "Cannon na Gianta przy rzece"))

    return [c for c in out if agent._can_play(board, player, c.card, defensive=True)]


def _rules_spells(agent: LogicAgent, ctx: BattleContext) -> list[PlayCandidate]:  # Fireball: value trade, clustery, Giant+support
    out: list[PlayCandidate] = []
    board, player = ctx.board, ctx.player

    for enemy in ctx.enemies:
        if enemy.name != "Musketeer":
            continue
        play = _fireball_on_unit(enemy)
        score = SCORE_HIGH + 10 if enemy.hp <= FIREBALL_DAMAGE else SCORE_MID + 5
        out.append(PlayCandidate(*play, score, "Fireball na Musketeer"))

    cluster = _fireball_best_cluster(ctx.enemies)
    if cluster:
        hits = _fireball_hit_count(ctx.enemies, cluster[1], cluster[2])
        out.append(
            PlayCandidate(
                *cluster,
                SCORE_MID + hits * 8,
                f"Fireball na skupisko ({hits} jednostek)",
            )
        )

    if ctx.elixir >= FIREBALL_CANNON_ELIXIR and ctx.quiet:
        for enemy in ctx.enemies:
            if enemy.name == "Cannon":
                out.append(
                    PlayCandidate(
                        *_fireball_on_unit(enemy),
                        SCORE_MID,
                        "Spokój — Fireball na wrogi Cannon",
                    )
                )

    # Giant + support za tankiem
    for enemy in ctx.enemies:
        if enemy.name != "Giant":
            continue
        for support in ctx.enemies:
            if support is enemy or support.owner != enemy.owner:
                continue
            if not _ranged_behind_tank(enemy, support):
                continue
            play = _fireball_on_unit(support)
            out.append(
                PlayCandidate(
                    *play,
                    SCORE_HIGH,
                    "Fireball na ranged za Giantem",
                )
            )

    return [c for c in out if agent._can_play(board, player, c.card, defensive=True)]


def _rules_hog_counters(agent: LogicAgent, ctx: BattleContext) -> list[PlayCandidate]:  # Hog drugą alejką gdy wróg ma Cannon
    out: list[PlayCandidate] = []
    board, player = ctx.board, ctx.player

    for enemy in ctx.enemies:
        if enemy.name != "Cannon":
            continue
        play = _hog_opposite_lane(board, player, enemy)
        out.append(
            PlayCandidate(
                *play,
                SCORE_HIGH + 5,
                "Hog drugą alejką — wrogi Cannon zajęty",
            )
        )

    return [c for c in out if agent._can_play(board, player, c.card, offensive=True)]


def _rules_push_combos(agent: LogicAgent, ctx: BattleContext) -> list[PlayCandidate]:  # Giant push + followupy albo solo Hog rush
    out: list[PlayCandidate] = []
    board, player = ctx.board, ctx.player

    if ctx.elixir >= ELIXIR_PUSH and len(ctx.mine) == 0 and not ctx.under_threat:
        giant = _giant_push_lane(board, player)
        lane = ctx.open_lane
        gx, gy = giant[1], giant[2]
        musk_follow = ("Musketeer", gx, gy - 2.5 * _forward(player))
        hog_follow = _hog_on_bridge(player, lane)
        out.append(
            PlayCandidate(
                *giant,
                SCORE_HIGH if ctx.elixir >= ELIXIR_FULL else SCORE_MID + 10,
                "Giant push — Musketeer + Hog w kolejce",
                followups=[musk_follow, hog_follow],
            )
        )

    if (
        ctx.elixir >= 4.0
        and not any(t.name == "Hog_Rider" for t in ctx.mine)
        and not ctx.under_threat
        and (ctx.quiet or ctx.late_game)
    ):
        play = _hog_rush_lane(board, player)
        out.append(
            PlayCandidate(
                *play,
                SCORE_MID + (10 if ctx.late_game else 0),
                "Hog rush w otwartą alejkę",
            )
        )

    return [c for c in out if agent._can_play(board, player, c.card, offensive=True)]


def _rules_tower_defense(  # Knight na jednostkę bijącą wieżę lub świeżego ranged
    agent: LogicAgent, ctx: BattleContext
) -> list[PlayCandidate]:
    out: list[PlayCandidate] = []
    board, player = ctx.board, ctx.player

    if ctx.tower_threat is None:
        return out

    play = _knight_on_unit(board, player, ctx.tower_threat)
    if play:
        score = SCORE_HIGH + 8 if ctx.elixir < 4.0 else SCORE_MID + 15
        out.append(PlayCandidate(*play, score, "Knight na jednostkę bijącą wieżę"))

    for enemy in ctx.enemies:
        if not _ranged_just_crossed(enemy, player):
            continue
        play = _knight_on_unit(board, player, enemy)
        if play:
            out.append(
                PlayCandidate(
                    *play,
                    SCORE_MID + 12,
                    "Knight na świeżego ranged za rzeką",
                )
            )

    return [c for c in out if agent._can_play(board, player, c.card, defensive=True)]


def _rules_lane_pressure(agent: LogicAgent, ctx: BattleContext) -> list[PlayCandidate]:  # placeholder — na razie pusto
    return []


def _rules_support_allies(  # Musketeer za Giantem/Knightem przy moście
    agent: LogicAgent, ctx: BattleContext
) -> list[PlayCandidate]:
    out: list[PlayCandidate] = []
    board, player = ctx.board, ctx.player

    for ally in ctx.mine:
        if ally.name == "Giant" and _near_bridge(ally, player):
            play = _musketeer_behind_unit(player, ally, dist=3.5)
            out.append(
                PlayCandidate(*play, SCORE_MID + 8, "Musketeer za Giantem przy moście")
            )
        if (
            ally.name == "Knight"
            and ally.max_hp > 0
            and ally.hp / ally.max_hp > 0.5
            and _crossing_bridge_toward_enemy(ally, player)
        ):
            play = _musketeer_behind_unit(player, ally, dist=3.5)
            out.append(
                PlayCandidate(*play, SCORE_MID + 5, "Musketeer za Knightem na moście")
            )

    return [c for c in out if agent._can_play(board, player, c.card, offensive=True)]


def _rules_elixir_cycle(agent: LogicAgent, ctx: BattleContext) -> list[PlayCandidate]:  # leak/full + spokój → tanie karty za Kingiem
    out: list[PlayCandidate] = []
    board, player = ctx.board, ctx.player

    if ctx.elixir < 3.0 and ctx.quiet:
        return out

    leak = ctx.elixir >= ELIXIR_LEAK
    full = ctx.elixir >= ELIXIR_FULL - 0.05

    if (leak or full) and ctx.quiet and len(ctx.enemies) == 0:
        play = _musketeer_behind_king(player)
        out.append(
            PlayCandidate(
                *play,
                SCORE_LOW + (15 if leak else 8),
                "Cycle: Musketeer za Kingiem",
            )
        )
        knight = _knight_behind_king(player)
        out.append(
            PlayCandidate(
                *knight,
                SCORE_LOW + (12 if leak else 5),
                "Cycle: Knight za Kingiem",
            )
        )

    if leak and not ctx.under_threat:
        play = _hog_rush_lane(board, player)
        out.append(
            PlayCandidate(*play, SCORE_LOW + 10, "Leak — wymuszony Hog rush")
        )

    return [
        c
        for c in out
        if agent._can_play(
            board, player, c.card, offensive=not ctx.under_threat, defensive=True
        )
    ]


# --- kontekst planszy ---


def _build_context(board: Board, player: int) -> BattleContext:  # jednorazowy snapshot dla wszystkich reguł
    enemy = 1 - player
    mine = _my_troops(board, player)
    enemies = _enemy_troops(board, player)
    return BattleContext(
        board=board,
        player=player,
        elixir=board.elixir[player],
        enemy_elixir=board.elixir[enemy],
        time=board.time,
        mine=mine,
        enemies=enemies,
        on_my_side=[e for e in enemies if _on_my_side(e, player)],
        tower_threat=_enemy_attacking_my_tower(board, player),
        under_threat=_enemies_in_threat_zone(board, player),
        quiet=_is_quiet_board(board, player),
        late_game=board.time >= MATCH_TIME_LIMIT * 0.55,
        own_cannon_alive=any(
            t.name == "Cannon" and t.alive and t.owner == player for t in mine
        ),
        open_lane=_open_lane(board, player),
    )


def _open_lane(board: Board, player: int) -> str:  # alejka ze słabszą / jedyną żywą princess
    princesses = [
        t for t in _enemy_towers(board, player) if t.name == "Tower" and t.alive
    ]
    if len(princesses) == 1:
        return _lane_for_tower(princesses[0])
    if princesses:
        weakest = min(princesses, key=lambda t: t.hp)
        return _lane_for_tower(weakest)
    return "Tower_L"


def _fireball_hit_count(
    enemies: list[Troop], cx: float, cy: float, radius: float = FIREBALL_RADIUS
) -> int:
    return sum(1 for e in enemies if e.distance_to_xy(cx, cy) <= radius)


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


def _near_bridge(unit: Troop, player: int) -> bool:
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
    if not _ranged_unit(enemy) or not _on_my_side(enemy, player):
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
    if tank.owner == 0:
        return ranged.y > tank.y and ranged.distance_to(tank) <= 6.0
    return ranged.y < tank.y and ranged.distance_to(tank) <= 6.0


def _weakest_enemy_princess(board: Board, player: int) -> Optional[Troop]:
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


def _is_quiet_board(board: Board, player: int) -> bool:  # nikt nie walczy na moście / naszej połowie
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


# --- pozycjonowanie kart ---


def _giant_push_lane(board: Board, player: int) -> tuple[str, float, float]:
    target = _weakest_enemy_princess(board, player)
    lane = _lane_for_tower(target) if target else "Tower_L"
    tx, ty = TOWER_LAYOUT[player][lane]
    return "Giant", tx, ty + 3.0 * _forward(player)


def _musketeer_behind_king(player: int) -> tuple[str, float, float]:
    kx, ky = TOWER_LAYOUT[player]["King_Tower"]
    return "Musketeer", kx, ky - 2.0 * _forward(player)


def _knight_behind_king(player: int) -> tuple[str, float, float]:
    kx, ky = TOWER_LAYOUT[player]["King_Tower"]
    return "Knight", kx, ky - 1.5 * _forward(player)


def _musketeer_behind_unit(
    player: int, unit: Troop, dist: float = 3.5
) -> tuple[str, float, float]:
    return "Musketeer", unit.x, unit.y - dist * _forward(player)


def _musketeer_safe_support(
    board: Board, player: int, ranged: Troop
) -> tuple[str, float, float]:
    towers = _my_towers(board, player)
    if not towers:
        kx, ky = TOWER_LAYOUT[player]["King_Tower"]
        return "Musketeer", kx, ky - 1.5 * _forward(player)
    if ranged.x < ARENA_WIDTH / 2:
        tower = next((t for t in towers if t.x > ARENA_WIDTH / 2), towers[0])
    else:
        tower = next((t for t in towers if t.x < ARENA_WIDTH / 2), towers[0])
    return "Musketeer", tower.x, tower.y - 2.0 * _forward(player)


def _cannon_river_y(player: int) -> float:
    if player == 0:
        return RIVER_Y - CANNON_RIVER_DISTANCE
    return RIVER_Y + CANNON_RIVER_DISTANCE


def _cannon_vs_building_attacker(
    board: Board, player: int, enemy: Troop
) -> tuple[str, float, float]:
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


def _hog_on_bridge(player: int, lane: str) -> tuple[str, float, float]:
    bx = BRIDGE_LANE_X[0] if lane == "Tower_L" else BRIDGE_LANE_X[1]
    y = RIVER_Y - 1.0 if player == 0 else RIVER_Y + 1.0
    return "Hog_Rider", bx, y


def _hog_rush_lane(board: Board, player: int) -> tuple[str, float, float]:
    lane = _open_lane(board, player)
    return _hog_on_bridge(player, lane)


def _hog_opposite_lane(
    board: Board, player: int, cannon: Troop
) -> tuple[str, float, float]:
    lane = "Tower_R" if cannon.x < ARENA_CENTER_X else "Tower_L"
    return _hog_on_bridge(player, lane)


def _tower_under_hog_attack(board: Board, player: int, hog: Troop) -> Optional[Troop]:
    target = board.find_nearest_target(hog)
    if target and target.is_tower and target.owner == player and target.alive:
        return target
    if _enemy_attacking_my_tower(board, player) == hog:
        for tower in _my_towers(board, player):
            if tower.alive and hog.distance_to(tower) <= 10.0:
                return tower
    return None


def _cannon_vs_hog(
    board: Board, player: int, hog: Troop
) -> tuple[str, float, float]:
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
    for dist in (0.0, 0.6, 1.0):
        for dx, dy in ((0, dist), (dist, 0), (-dist, 0), (0, -dist)):
            x = enemy.x + dx
            y = enemy.y + dy
            if board._in_deploy_zone(player, x, y):
                return "Knight", x, y
    x = max(1.0, min(ARENA_WIDTH - 1.0, enemy.x))
    if player == 0:
        y = max(2.0, min(RIVER_Y - 0.5, enemy.y))
    else:
        y = max(RIVER_Y + 0.5, min(ARENA_LENGTH - 2.0, enemy.y))
    return "Knight", x, y
