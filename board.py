from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from cards import (
    CARD_TO_ID,
    PLAYABLE_CARDS,
    Troop,
    Tower,
    cards_dic,
)

# Plansza: 1 jednostka ≈ 1 kafelek (tile) w Clash Royale.
# Oficjalna arena CR: 18 kafelków szerokości × 32 długości (wiki / datamining).
ARENA_WIDTH = 18.0   # kafelki w poziomie (oś X)
ARENA_LENGTH = 32.0  # kafelki w pionie (oś Y); rzeka na Y = 16
TICK_DT = 0.1
MAX_ELIXIR = 10.0
START_ELIXIR = 5.0
ELIXIR_PER_SECOND = 1.0 / 2.8
MATCH_TIME_LIMIT = 180.0
RIVER_Y = ARENA_LENGTH / 2
RIVER_HALF_WIDTH = 1.2
# Dwa mosty (lewy / prawy), jak na arenie CR.
BRIDGE_LANE_X = (4.0, 14.0)
BRIDGES = [(BRIDGE_LANE_X[0], RIVER_Y), (BRIDGE_LANE_X[1], RIVER_Y)]

# Wieże (jak w CR): princess ~6 kafelków od rzeki, król z tyłu.
# P0 = dół (y rośnie w górę), P1 = góra.
TOWER_LAYOUT = {
    0: {
        "Tower_L": (3.5, 10.0),
        "Tower_R": (14.5, 10.0),
        "King_Tower": (9.0, 2.0),
    },
    1: {
        "Tower_L": (3.5, 22.0),
        "Tower_R": (14.5, 22.0),
        "King_Tower": (9.0, 30.0),
    },
}

DEPLOY_ZONES = {
    0: [(4.0, 6.0), (9.0, 6.0), (14.0, 6.0)],
    1: [(4.0, 26.0), (9.0, 26.0), (14.0, 26.0)],
}

HAND_SIZE = 4
NUM_ACTIONS = 1 + HAND_SIZE * len(DEPLOY_ZONES[0])  # noop + slot ręki (0..3) × strefa
MAX_UNITS_OBS = 24
OBS_DIM = 2 + 6 + MAX_UNITS_OBS * 5  # elixir, czas + wieże + jednostki


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict = field(default_factory=dict)


class Board:
    """
    Uproszczone środowisko Clash Royale pod RL.

    Akcja (int 0..12):
      0 — nic nie rób
      1..12 — zagraj kartę (slot ręki 0..3 × strefa 0..2); tylko 4 karty na ręce (cykl z 6)

    Gracz 0 uczy się; gracz 1 domyślnie losowy bot (można podpiąć drugi model).
    """

    def __init__(
        self,
        deck_p0: Optional[list[str]] = None,
        deck_p1: Optional[list[str]] = None,
        *,
        seed: Optional[int] = None,
    ):
        self.deck_p0 = list(deck_p0 or PLAYABLE_CARDS)
        self.deck_p1 = list(deck_p1 or PLAYABLE_CARDS)
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
        self._pending_play = {0: None, 1: None}
        self._init_towers()
        self._init_hands()
        return self.get_observation(player=0)

    def _full_deck(self, player: int) -> list[str]:
        return self.deck_p0 if player == 0 else self.deck_p1

    def _init_hands(self) -> None:
        """6 unikalnych kart w kolejce: 4 na ręce, po rzucie karta wraca na koniec (bez duplikatów)."""
        for player in (0, 1):
            cards = list(dict.fromkeys(self._full_deck(player)))
            self._rng.shuffle(cards)
            queue: deque[str] = deque(cards)
            self.hand[player] = [queue.popleft() for _ in range(HAND_SIZE)]
            self.hand_queue[player] = queue

    def get_hand(self, player: int) -> list[str]:
        return list(self.hand[player])

    def card_in_hand(self, player: int, card_name: str) -> bool:
        return card_name in self.hand[player]

    def _cycle_hand_after_play(self, player: int, card_name: str) -> None:
        hand = self.hand[player]
        if card_name not in hand:
            return
        hand.remove(card_name)
        queue = self.hand_queue[player]
        queue.append(card_name)
        if queue:
            hand.append(queue.popleft())

    def set_pending_play(
        self, player: int, play: Optional[tuple[str, float, float]]
    ) -> None:
        self._pending_play[player] = play

    def _in_deploy_zone(self, player: int, x: float, y: float) -> bool:
        if not (1.0 <= x <= ARENA_WIDTH - 1.0):
            return False
        if player == 0:
            return 2.0 <= y <= RIVER_Y - 0.5
        return RIVER_Y + 0.5 <= y <= ARENA_LENGTH - 2.0

    def play_card_at(self, player: int, card_name: str, x: float, y: float) -> bool:
        if self.done or not self._in_deploy_zone(player, x, y):
            return False
        if not self.card_in_hand(player, card_name):
            return False

        stats = cards_dic.get(card_name)
        if not stats:
            return False

        cost = stats["elisir"]
        if self.elixir[player] < cost:
            return False

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
        zones = len(DEPLOY_ZONES[player])
        idx = action - 1
        slot = idx // zones
        zone_idx = idx % zones
        if slot >= len(hand):
            return None, None
        return hand[slot], zone_idx

    def play_card(self, player: int, card_name: str, zone_idx: int) -> bool:
        if self.done or zone_idx < 0 or zone_idx >= len(DEPLOY_ZONES[player]):
            return False
        if not self.card_in_hand(player, card_name):
            return False

        stats = cards_dic.get(card_name)
        if not stats:
            return False

        cost = stats["elisir"]
        if self.elixir[player] < cost:
            return False

        x, y = DEPLOY_ZONES[player][zone_idx]
        troop = Troop(card_name, player)
        troop.place((x, y))
        self.elixir[player] -= cost
        self.troops.append(troop)
        self._cycle_hand_after_play(player, card_name)
        return True

    def step(
        self,
        action_p0: int,
        action_p1: Optional[int] = None,
    ) -> StepResult:
        if self.done:
            return StepResult(
                self.get_observation(0), 0.0, True, False, {"winner": self.winner}
            )

        reward = 0.0
        towers_before = self._tower_hp_snapshot()

        if action_p1 is None:
            action_p1 = self._random_bot_action(1)

        self._apply_action(0, action_p0)
        self._apply_action(1, action_p1)

        self._regen_elixir()
        dmg_dealt, dmg_taken = self._simulate_combat(TICK_DT)
        reward += 0.001 * dmg_dealt - 0.001 * dmg_taken

        tower_delta = self._tower_hp_delta(towers_before)
        reward += 0.01 * tower_delta[0] - 0.01 * tower_delta[1]

        self._remove_dead()
        self.time += TICK_DT

        terminated, truncated = self._check_end()
        if terminated:
            if self.winner == 0:
                reward += 1.0
            elif self.winner == 1:
                reward -= 1.0

        reward -= 0.001  # mała kara za czas — zachęta do kończenia meczu

        return StepResult(
            observation=self.get_observation(0),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info={
                "winner": self.winner,
                "time": self.time,
                "elixir": self.elixir[0],
                "troops_alive": sum(1 for t in self.troops if t.alive),
            },
        )

    def _apply_action(self, player: int, action: int) -> None:
        pending = self._pending_play[player]
        if pending is not None:
            card, x, y = pending
            if not self.play_card_at(player, card, x, y):
                # Zapas: środek własnej połowy (np. Cannon przy rzece)
                mid_x = ARENA_WIDTH / 2
                mid_y = (RIVER_Y - 3.0) if player == 0 else (RIVER_Y + 3.0)
                self.play_card_at(player, card, mid_x, mid_y)
            self._pending_play[player] = None
            return

        card, zone = self.action_to_card_zone(action, player)
        if card is not None and zone is not None:
            self.play_card(player, card, zone)

    def _random_bot_action(self, player: int) -> int:
        affordable = []
        hand = self.hand[player]
        zones = len(DEPLOY_ZONES[player])
        for i, name in enumerate(hand):
            cost = cards_dic[name]["elisir"]
            if self.elixir[player] >= cost:
                for z in range(zones):
                    affordable.append(1 + i * zones + z)
        if affordable and self._rng.random() < 0.35:
            return self._rng.choice(affordable)
        return 0

    def _regen_elixir(self) -> None:
        gain = ELIXIR_PER_SECOND * TICK_DT
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
        return any(abs(troop.x - bx) <= 1.8 for bx in BRIDGE_LANE_X)

    def _needs_bridge(self, troop: Troop, target: Troop) -> bool:
        if getattr(troop, "jumps_river", False):
            return False
        if troop.is_building and troop.speed <= 0:
            return False
        ty, target_y = troop.y, target.y
        if ty < RIVER_Y and target_y < RIVER_Y:
            return False
        if ty > RIVER_Y and target_y > RIVER_Y:
            return False
        return True

    def get_move_waypoint(self, troop: Troop, target: Troop) -> tuple[float, float]:
        """Ruch tylko przez najbliższy most — najpierw do osi mostu, potem przez rzekę."""
        if not self._needs_bridge(troop, target):
            return target.x, target.y

        bx = self._nearest_bridge_x(troop.x)

        if troop.owner == 0:
            if abs(troop.x - bx) > 1.2:
                return bx, troop.y
            if troop.y < RIVER_Y - 1.5:
                return bx, RIVER_Y - 1.5
            if troop.y < RIVER_Y + RIVER_HALF_WIDTH:
                return bx, RIVER_Y + RIVER_HALF_WIDTH + 0.3
            return target.x, target.y

        if abs(troop.x - bx) > 1.2:
            return bx, troop.y
        if troop.y > RIVER_Y + 1.5:
            return bx, RIVER_Y + 1.5
        if troop.y > RIVER_Y - RIVER_HALF_WIDTH:
            return bx, RIVER_Y - RIVER_HALF_WIDTH - 0.3
        return target.x, target.y

    def _enforce_no_river_cut(self, troop: Troop) -> None:
        """Cofa jednostkę, jeśli weszła w rzekę poza mostem."""
        if getattr(troop, "jumps_river", False):
            return
        if troop.is_building and troop.speed <= 0:
            return
        if not self._in_river_water(troop.y) or self._on_bridge_lane(troop):
            return

        bx = self._nearest_bridge_x(troop.x)
        troop.x = bx
        if troop.owner == 0:
            troop.y = RIVER_Y - RIVER_HALF_WIDTH - 0.2
        else:
            troop.y = RIVER_Y + RIVER_HALF_WIDTH + 0.2

    def find_nearest_target(self, troop: Troop) -> Optional[Troop]:
        best: Optional[Troop] = None
        best_dist = float("inf")

        for other in self._all_combatants():
            if not troop.can_target(other):
                continue
            dist = troop.distance_to(other)
            if dist < best_dist:
                best_dist = dist
                best = other
        return best

    def find_tower_target(self, tower: Troop) -> Optional[Troop]:
        """Wieża strzela do najbliższego wroga w zasięgu (nie do celu poza range)."""
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

    def _simulate_combat(self, dt: float) -> tuple[float, float]:
        """Zwraca (obrażenia zadane przez P0, obrażenia otrzymane przez P0)."""
        dealt_p0 = 0.0
        taken_p0 = 0.0

        for troop in list(self.troops):
            if not troop.alive:
                continue
            if not troop.tick_age(dt):
                continue

            target = self.find_nearest_target(troop)
            if target is None:
                continue

            if troop.is_in_range(target):
                dmg = troop.hit(target, dt)
            elif not troop.is_building or troop.speed > 0:
                wx, wy = self.get_move_waypoint(troop, target)
                troop.move_towards_point(wx, wy, dt)
                self._enforce_no_river_cut(troop)
                troop.update_cooldown(dt)
                dmg = 0.0
            else:
                troop.update_cooldown(dt)
                dmg = 0.0

            if dmg <= 0:
                continue
            if troop.owner == 0:
                dealt_p0 += dmg
            elif target.owner == 0:
                taken_p0 += dmg

        for tower in self.towers:
            if not tower.alive:
                continue
            target = self.find_tower_target(tower)
            if target is None:
                tower.update_cooldown(dt)
                dmg = 0.0
            else:
                dmg = tower.hit(target, dt)
            if dmg <= 0:
                continue
            if tower.owner == 0:
                dealt_p0 += dmg
            elif target.owner == 0:
                taken_p0 += dmg

        return dealt_p0, taken_p0

    def _remove_dead(self) -> None:
        self.troops = [t for t in self.troops if t.alive]

    def _tower_hp_snapshot(self) -> dict[int, float]:
        snap: dict[int, float] = {0: 0.0, 1: 0.0}
        for t in self.towers:
            if t.alive:
                snap[t.owner] += t.hp
        return snap

    def _tower_hp_delta(self, before: dict[int, float]) -> tuple[float, float]:
        after = self._tower_hp_snapshot()
        # (korzyść P0, korzyść P1) w sensie utraty HP przeciwnika
        enemy_loss_p0 = before[1] - after[1]
        enemy_loss_p1 = before[0] - after[0]
        return enemy_loss_p0, enemy_loss_p1

    def _check_end(self) -> tuple[bool, bool]:
        kings = {0: None, 1: None}
        for t in self.towers:
            if t.name == "King_Tower":
                kings[t.owner] = t

        if kings[0] and not kings[0].alive:
            self.winner = 1
            self.done = True
            return True, False
        if kings[1] and not kings[1].alive:
            self.winner = 0
            self.done = True
            return True, False

        if self.time >= MATCH_TIME_LIMIT:
            hp0 = self._tower_hp_snapshot()[0]
            hp1 = self._tower_hp_snapshot()[1]
            if hp0 > hp1:
                self.winner = 0
            elif hp1 > hp0:
                self.winner = 1
            else:
                self.winner = None
            self.done = True
            return False, True

        return False, False

    def get_observation(self, player: int) -> np.ndarray:
        """Wektor stanu z perspektywy `player` (0 = agent RL)."""
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0] = self.elixir[player] / MAX_ELIXIR
        obs[1] = min(1.0, self.time / MATCH_TIME_LIMIT)

        idx = 2
        for t in self.towers:
            obs[idx] = t.hp / t.max_hp if t.max_hp else 0.0
            idx += 1

        unit_base = 8
        slot = 0
        for troop in self.troops:
            if not troop.alive or slot >= MAX_UNITS_OBS:
                continue
            base = unit_base + slot * 5
            obs[base] = 1.0 if troop.owner == player else -1.0
            obs[base + 1] = CARD_TO_ID.get(troop.name, -1) / max(1, len(PLAYABLE_CARDS) - 1)
            obs[base + 2] = troop.x / ARENA_WIDTH
            obs[base + 3] = troop.y / ARENA_LENGTH
            obs[base + 4] = troop.hp / troop.max_hp if troop.max_hp else 0.0
            slot += 1

        return obs

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
    total_reward = 0.0
    for _ in range(steps):
        action = env.np_rng.integers(0, NUM_ACTIONS)
        result = env.step(action)
        total_reward += result.reward
        if result.terminated or result.truncated:
            break
    print(f"Koniec meczu: winner={env.winner}, reward={total_reward:.3f}, steps={env.time:.1f}s")


if __name__ == "__main__":
    run_random_episode()
