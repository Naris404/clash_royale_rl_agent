"""
Wizualizacja pygame — uruchom: python visualize.py

Tempo gry — ustaw REAL_SECONDS_PER_SIM_SECOND (na górze pliku lub --tempo):
  1.0  → 1 sekunda symulacji trwa ~1 sekundę u Ciebie (domyślnie)
  2.0  → wolniej (1 s gry ≈ 2 s rzeczywiste)
  0.25 → szybciej (1 s gry ≈ ¼ s rzeczywiste)

Sterowanie:
  Spacja       — pauza / wznowienie
  +/-          — dodatkowy mnożnik tempa
  [ / ]        — zmniejsz / zwiększ REAL_SECONDS_PER_SIM_SECOND
  ← / → (pauza) — cofnij / do przodu po historii meczu
  ↑ / ↓ (pauza) — przesuń widok areny
  Q / Esc      — wyjście
"""

from __future__ import annotations

import argparse
import sys

try:
    import pygame
except ImportError:
    print("Brak pygame. Zainstaluj: pip install pygame")
    sys.exit(1)

from board import (
    ARENA_LENGTH,
    ARENA_WIDTH,
    BRIDGE_LANE_X,
    BRIDGES,
    DEPLOY_ZONES,
    RIVER_HALF_WIDTH,
    RIVER_Y,
    TICK_DT,
    Board,
    SpellEffect,
)

# Legenda: niebieskie/czerwone kółka = strefy rzutu; żółte = most (Hog)

# --- TEMPO GRY (dostosuj tutaj) ---
# Ile sekund RZECZYWISTEGO czasu ma trwać 1 sekunda w symulacji.
# 1.0   = wierne tempo (1 s licznika gry ≈ 1 s u Ciebie)
# 0.08  = szybki podgląd (~jak dawniej ticks_per_frame=2 przy 60 FPS)
REAL_SECONDS_PER_SIM_SECOND = 0.08
from cards import Troop, cards_dic
from logic_agent import LogicAgent
from rl_agent import RLAgent

# Skala ekranu — zmniejsz TILE_PX / UI_SCALE jeśli nie mieści się na monitorze
UI_SCALE = 0.65
TILE_PX = max(10, int(22 * UI_SCALE))
MARGIN_X = max(12, int(36 * UI_SCALE))
MARGIN_Y = max(20, int(50 * UI_SCALE))
HUD_H = max(36, int(56 * UI_SCALE))
HAND_BAR_H = max(34, int(52 * UI_SCALE))
CARD_W = max(48, int(78 * UI_SCALE))
CARD_H = max(30, int(44 * UI_SCALE))
CARD_GAP = 4
SCROLL_STEP = max(14, int(28 * UI_SCALE))

CARD_LABEL = {
    "Knight": "Knight",
    "Giant": "Giant",
    "Cannon": "Cannon",
    "Musketeer": "Musk.",
    "Hog_Rider": "Hog",
    "Fireball": "Fire",
}

COL_GRASS = (45, 106, 48)
COL_GRASS_DARK = (38, 92, 42)
COL_RIVER = (55, 140, 210)
COL_BRIDGE = (139, 90, 43)
COL_PATH = (160, 120, 70)
COL_P0 = (59, 130, 246)
COL_P1 = (239, 68, 68)
COL_TEXT = (240, 240, 240)
COL_HP_BG = (40, 40, 40)
COL_HP_FG = (74, 222, 128)

CARD_SHORT = {
    "Knight": "K",
    "Giant": "G",
    "Cannon": "C",
    "Musketeer": "M",
    "Hog_Rider": "H",
    "Fireball": "F",
}

MAX_HISTORY = 4000
TEMPO_STEP = 0.15  # zmiana [ / ] dla REAL_SECONDS_PER_SIM_SECOND
MIN_SIM_SEC_DURATION = 0.05
MAX_SIM_SEC_DURATION = 8.0


def apply_ui_scale(scale: float) -> None:
    """Ustaw skalę okna (wywołaj przed run_pygame)."""
    global UI_SCALE, TILE_PX, MARGIN_X, MARGIN_Y, HUD_H, HAND_BAR_H, CARD_W, CARD_H
    UI_SCALE = max(0.35, min(1.0, scale))
    TILE_PX = max(10, int(22 * UI_SCALE))
    MARGIN_X = max(12, int(36 * UI_SCALE))
    MARGIN_Y = max(20, int(50 * UI_SCALE))
    HUD_H = max(36, int(56 * UI_SCALE))
    HAND_BAR_H = max(34, int(52 * UI_SCALE))
    CARD_W = max(48, int(78 * UI_SCALE))
    CARD_H = max(30, int(44 * UI_SCALE))


def sim_ticks_from_real_time(
    dt_real: float,
    real_seconds_per_sim_second: float,
    speed_mult: float,
    pending_sim_time: float,
) -> tuple[int, float]:
    """
    dt_real — sekundy od ostatniej klatki.
    Zwraca (liczba ticków symulacji, niewykorzystany czas symulacji w buforze).
    """
    if real_seconds_per_sim_second <= 0:
        real_seconds_per_sim_second = 1.0
    pending_sim_time += (dt_real / real_seconds_per_sim_second) * speed_mult
    n = int(pending_sim_time / TICK_DT)
    pending_sim_time -= n * TICK_DT
    return n, pending_sim_time


def capture_state(env: Board) -> dict:
    return {
        "time": env.time,
        "elixir": list(env.elixir),
        "done": env.done,
        "winner": env.winner,
        "troops": [
            (
                t.name,
                t.owner,
                t.x,
                t.y,
                t.hp,
                t.max_hp,
                t.alive,
                t.cooldown,
                t.age,
            )
            for t in env.troops
        ],
        "towers": [
            (t.name, t.owner, t.x, t.y, t.hp, t.max_hp, t.alive) for t in env.towers
        ],
        "hand": {0: env.get_hand(0), 1: env.get_hand(1)},
        "hand_queue": {
            0: list(env.hand_queue[0]),
            1: list(env.hand_queue[1]),
        },
        "spell_effects": [
            (
                e.card,
                e.x,
                e.y,
                e.start_x,
                e.start_y,
                e.radius,
                e.owner,
                e.age,
                e.travel_time,
                e.fade_time,
            )
            for e in env.spell_effects
        ],
    }


def restore_state(env: Board, state: dict) -> None:
    env.time = state["time"]
    env.elixir = list(state["elixir"])
    env.done = state["done"]
    env.winner = state["winner"]
    env.troops = []
    for row in state["troops"]:
        t = Troop(row[0], row[1])
        t.place((row[2], row[3]))
        t.hp, t.max_hp, t.alive = row[4], row[5], row[6]
        t.cooldown, t.age = row[7], row[8]
        env.troops.append(t)
    for tower, row in zip(env.towers, state["towers"]):
        tower.hp = row[4]
        tower.max_hp = row[5]
        tower.alive = row[6]
    if "hand" in state:
        from collections import deque

        env.hand = {0: list(state["hand"][0]), 1: list(state["hand"][1])}
        if "hand_queue" in state:
            env.hand_queue = {
                0: deque(state["hand_queue"][0]),
                1: deque(state["hand_queue"][1]),
            }
    if "spell_effects" in state:
        env.spell_effects = [
            SpellEffect(
                card=row[0],
                x=row[1],
                y=row[2],
                start_x=row[3],
                start_y=row[4],
                radius=row[5],
                owner=row[6],
                age=row[7],
                travel_time=row[8],
                fade_time=row[9],
            )
            for row in state["spell_effects"]
        ]
    else:
        env.spell_effects = []


def world_to_screen(x: float, y: float, scroll: tuple[int, int]) -> tuple[int, int]:
    """Y rośnie w dół ekranu (P0 na dole). scroll = przesunięcie widoku w px."""
    sx = MARGIN_X + int(x * TILE_PX) + scroll[0]
    sy = MARGIN_Y + int((ARENA_LENGTH - y) * TILE_PX) + scroll[1]
    return sx, sy


def arena_pixel_size() -> tuple[int, int]:
    return int(ARENA_WIDTH * TILE_PX), int(ARENA_LENGTH * TILE_PX)


def draw_arena(surface: pygame.Surface, scroll: tuple[int, int]) -> None:
    aw, ah = arena_pixel_size()
    ox, oy = MARGIN_X + scroll[0], MARGIN_Y + scroll[1]

    # Szachownica trawy
    for row in range(int(ARENA_LENGTH)):
        for col in range(int(ARENA_WIDTH)):
            c = COL_GRASS if (row + col) % 2 == 0 else COL_GRASS_DARK
            rx = ox + col * TILE_PX
            ry = oy + (int(ARENA_LENGTH) - 1 - row) * TILE_PX
            pygame.draw.rect(surface, c, (rx, ry, TILE_PX, TILE_PX))

    # Ścieżki do mostów (pionowe pasy)
    for bx in BRIDGE_LANE_X:
        sx = ox + int((bx - 0.5) * TILE_PX)
        pygame.draw.rect(
            surface,
            COL_PATH,
            (sx, oy, TILE_PX * 2, ah),
        )

    # Rzeka (poziomy pas)
    river_top = oy + int((ARENA_LENGTH - (RIVER_Y + RIVER_HALF_WIDTH)) * TILE_PX)
    river_h = int(RIVER_HALF_WIDTH * 2 * TILE_PX)
    pygame.draw.rect(surface, COL_RIVER, (ox, river_top, aw, river_h))

    # Mosty (dwa prostokąty na rzece)
    bridge_w = TILE_PX * 3
    bridge_h = int(RIVER_HALF_WIDTH * 2 * TILE_PX) + 4
    for bx in BRIDGE_LANE_X:
        sx = ox + int((bx - 1.0) * TILE_PX)
        sy = river_top - 2
        pygame.draw.rect(surface, COL_BRIDGE, (sx, sy, bridge_w, bridge_h))
        # poręcze
        pygame.draw.rect(surface, (90, 55, 25), (sx, sy, bridge_w, bridge_h), 2)


def hp_ratio(unit) -> float:
    if getattr(unit, "max_hp", 0) <= 0:
        return 0.0
    return max(0.0, min(1.0, unit.hp / unit.max_hp))


def draw_hp_bar(
    surface: pygame.Surface,
    cx: int,
    cy: int,
    ratio: float,
    color: tuple[int, int, int],
    width: int | None = None,
) -> None:
    if width is None:
        width = max(16, int(28 * UI_SCALE))
    ratio = max(0.0, min(1.0, ratio))
    h = max(2, int(4 * UI_SCALE))
    x = cx - width // 2
    y = cy - max(10, int(18 * UI_SCALE))
    pygame.draw.rect(surface, COL_HP_BG, (x, y, width, h))
    if ratio > 0:
        pygame.draw.rect(surface, color, (x, y, int(width * ratio), h))


def draw_tower(surface: pygame.Surface, tower, scroll: tuple[int, int]) -> None:
    if not tower.alive:
        return
    color = COL_P0 if tower.owner == 0 else COL_P1
    sx, sy = world_to_screen(tower.x, tower.y, scroll)
    size = int((26 if tower.name == "King_Tower" else 20) * UI_SCALE)
    rect = pygame.Rect(0, 0, size, size)
    rect.center = (sx, sy)
    pygame.draw.rect(surface, color, rect, border_radius=4)
    pygame.draw.rect(surface, (255, 255, 255), rect, 2, border_radius=4)
    label = "K" if tower.name == "King_Tower" else "P"
    font = pygame.font.SysFont("arial", max(9, int(14 * UI_SCALE)), bold=True)
    text = font.render(label, True, (255, 255, 255))
    surface.blit(text, text.get_rect(center=rect.center))
    draw_hp_bar(surface, sx, sy, hp_ratio(tower), color)


def _draw_alpha_circle(
    surface: pygame.Surface,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int, int],
) -> None:
    if radius <= 0:
        return
    diameter = radius * 2
    overlay = pygame.Surface((diameter, diameter), pygame.SRCALPHA)
    pygame.draw.circle(overlay, color, (radius, radius), radius)
    surface.blit(overlay, (center[0] - radius, center[1] - radius))


def draw_spell_effects(
    surface: pygame.Surface,
    effects: list[SpellEffect],
    scroll: tuple[int, int],
) -> None:
    for effect in effects:
        if effect.card != "Fireball":
            continue

        sx0, sy0 = world_to_screen(effect.start_x, effect.start_y, scroll)
        sx1, sy1 = world_to_screen(effect.x, effect.y, scroll)
        owner_color = COL_P0 if effect.owner == 0 else COL_P1

        travel_t = max(0.001, effect.travel_time)
        if effect.age < travel_t:
            progress = effect.age / travel_t
            cx = int(sx0 + (sx1 - sx0) * progress)
            cy = int(sy0 + (sy1 - sy0) * progress)
            trail_r = max(4, int(10 * UI_SCALE))
            for i, alpha in enumerate((50, 90, 140)):
                offset = int((2 - i) * 5 * progress)
                tx = cx - int((sx1 - sx0) * 0.04 * (i + 1))
                ty = cy - int((sy1 - sy0) * 0.04 * (i + 1))
                _draw_alpha_circle(
                    surface,
                    (tx - offset, ty - offset),
                    trail_r - i * 2,
                    (255, 120 + i * 25, 40, alpha),
                )
            ball_r = max(5, int(12 * UI_SCALE))
            pygame.draw.circle(surface, (255, 90, 20), (cx, cy), ball_r)
            pygame.draw.circle(surface, (255, 210, 80), (cx, cy), max(3, ball_r - 4))
            pygame.draw.circle(surface, (255, 255, 220), (cx, cy), max(2, ball_r - 7))

        impact_age = effect.age - effect.travel_time
        if impact_age >= -0.04:
            fade_t = max(0.001, effect.fade_time)
            explode = max(0.0, min(1.0, impact_age / fade_t))
            radius_px = int(effect.radius * TILE_PX * (0.25 + 0.85 * explode))
            alpha = int(200 * (1.0 - explode))
            _draw_alpha_circle(
                surface,
                (sx1, sy1),
                radius_px,
                (255, 110, 35, max(0, alpha)),
            )
            ring_alpha = int(220 * (1.0 - explode * 0.85))
            if ring_alpha > 0:
                ring = pygame.Surface(
                    (radius_px * 2 + 6, radius_px * 2 + 6), pygame.SRCALPHA
                )
                pygame.draw.circle(
                    ring,
                    (*owner_color, ring_alpha),
                    (radius_px + 3, radius_px + 3),
                    radius_px,
                    max(2, int(3 * UI_SCALE)),
                )
                surface.blit(ring, (sx1 - radius_px - 3, sy1 - radius_px - 3))
            if explode < 0.35:
                flash_r = max(6, int(18 * UI_SCALE * (1.0 - explode / 0.35)))
                _draw_alpha_circle(
                    surface,
                    (sx1, sy1),
                    flash_r,
                    (255, 240, 180, int(180 * (1.0 - explode / 0.35))),
                )


def draw_troop(surface: pygame.Surface, troop, scroll: tuple[int, int]) -> None:
    if not troop.alive:
        return
    color = COL_P0 if troop.owner == 0 else COL_P1
    sx, sy = world_to_screen(troop.x, troop.y, scroll)
    radius = int((14 if troop.is_building else 11) * UI_SCALE)
    pygame.draw.circle(surface, color, (sx, sy), radius)
    pygame.draw.circle(surface, (255, 255, 255), (sx, sy), radius, 2)
    short = CARD_SHORT.get(troop.name, troop.name[0])
    font = pygame.font.SysFont("arial", max(8, int(12 * UI_SCALE)), bold=True)
    text = font.render(short, True, (255, 255, 255))
    surface.blit(text, text.get_rect(center=(sx, sy)))
    draw_hp_bar(surface, sx, sy, hp_ratio(troop), color)


def draw_deploy_zones(surface: pygame.Surface, scroll: tuple[int, int]) -> None:
    """Przerywane kółka = domyślne strefy rzutu (lewo / środek / prawo), nie przy wieżach."""
    for player, col in ((0, COL_P0), (1, COL_P1)):
        for x, y in DEPLOY_ZONES[player]:
            sx, sy = world_to_screen(x, y, scroll)
            pygame.draw.circle(surface, col, (sx, sy), max(6, int(10 * UI_SCALE)), 2)

    # Mosty — miejsce rzutu Hoga (jaśniejsze kółko)
    for bx in BRIDGE_LANE_X:
        for player, col in ((0, COL_P0), (1, COL_P1)):
            y = RIVER_Y - 1.0 if player == 0 else RIVER_Y + 1.0
            sx, sy = world_to_screen(bx, y, scroll)
            pygame.draw.circle(surface, (255, 220, 120), (sx, sy), max(5, int(8 * UI_SCALE)), 2)


def draw_player_hand(
    surface: pygame.Surface,
    env: Board,
    player: int,
    y_top: int,
    font: pygame.font.Font,
) -> None:
    """4 karty w ręce (jak w Clash Royale)."""
    hand = env.get_hand(player)
    color = COL_P0 if player == 0 else COL_P1
    total_w = len(hand) * CARD_W + max(0, len(hand) - 1) * CARD_GAP
    x0 = (surface.get_width() - total_w) // 2
    elixir = env.elixir[player]
    title = font.render(
        f"P{player} reka ({'nieb.' if player == 0 else 'czerw.'})",
        True,
        color,
    )
    surface.blit(title, (x0, y_top - 14))

    for i, card in enumerate(hand):
        cost = cards_dic[card]["elisir"]
        affordable = elixir >= cost
        x = x0 + i * (CARD_W + CARD_GAP)
        y = y_top
        bg = (55, 65, 80) if affordable else (40, 40, 45)
        border = color if affordable else (90, 90, 90)
        pygame.draw.rect(surface, bg, (x, y, CARD_W, CARD_H), border_radius=6)
        pygame.draw.rect(surface, border, (x, y, CARD_W, CARD_H), 2, border_radius=6)
        label = CARD_LABEL.get(card, card[:6])
        text = font.render(label, True, COL_TEXT if affordable else (120, 120, 120))
        surface.blit(text, (x + 6, y + 6))
        cost_t = font.render(f"{cost}", True, (255, 220, 80) if affordable else (100, 100, 100))
        surface.blit(cost_t, (x + CARD_W - 18, y + CARD_H - 18))


def draw_hud(surface: pygame.Surface, env: Board, font: pygame.font.Font, info: str) -> None:
    bar_y = 8
    pygame.draw.rect(surface, (30, 30, 35), (0, 0, surface.get_width(), HUD_H))
    el0 = env.elixir[0]
    el1 = env.elixir[1]
    lines = [
        f"P0 (niebieski) eliksir: {el0:.1f}/10",
        f"P1 (czerwony) eliksir: {el1:.1f}/10",
        f"Czas: {env.time:.1f}s",
        info,
    ]
    x = 12
    for i, line in enumerate(lines):
        color = COL_P0 if i == 0 else COL_P1 if i == 1 else COL_TEXT
        text = font.render(line, True, color)
        surface.blit(text, (x, bar_y + i * 13))


def run_pygame(
    seed: int = 0,
    real_seconds_per_sim_second: float = REAL_SECONDS_PER_SIM_SECOND,
    fps: int = 60,
    model_path: str | None = None,
) -> None:
    pygame.init()
    aw, ah = arena_pixel_size()
    win_w = aw + MARGIN_X * 2
    win_h = ah + MARGIN_Y * 2 + HUD_H + HAND_BAR_H * 2 + 16
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(
        f"CR RL — plansza {int(ARENA_WIDTH)}x{int(ARENA_LENGTH)} kafelkow | P0=RL P1=bot"
    )
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", max(10, int(13 * UI_SCALE)))

    env = Board(seed=seed)
    env.reset(seed=seed)
    agent0 = RLAgent(seed=seed, model_path=model_path)
    agent1 = LogicAgent()
    if agent0.is_trained:
        print(f"P0: wczytano model PPO ({model_path or 'models/ppo_cr_best.zip'})")
    else:
        print("P0: brak modelu PPO — losowy agent (wytrenuj: python train.py)")

    paused = False
    speed = 1.0
    sim_sec_duration = max(
        MIN_SIM_SEC_DURATION,
        min(MAX_SIM_SEC_DURATION, real_seconds_per_sim_second),
    )
    pending_sim_time = 0.0
    done = False
    end_msg = ""
    scroll = [0, 0]
    history: list[dict] = [capture_state(env)]
    hist_idx = 0

    aw, ah = arena_pixel_size()
    max_scroll_x = max(0, aw // 3)
    max_scroll_y = max(0, ah // 3)

    running = True
    while running:
        dt_real = clock.tick(fps) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    if paused and history:
                        restore_state(env, history[hist_idx])
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    speed = min(6.0, speed * 1.25)
                elif event.key == pygame.K_MINUS:
                    speed = max(0.25, speed / 1.25)
                elif event.key == pygame.K_LEFTBRACKET:
                    sim_sec_duration = max(
                        MIN_SIM_SEC_DURATION, sim_sec_duration - TEMPO_STEP
                    )
                elif event.key == pygame.K_RIGHTBRACKET:
                    sim_sec_duration = min(
                        MAX_SIM_SEC_DURATION, sim_sec_duration + TEMPO_STEP
                    )
                elif paused:
                    if event.key == pygame.K_LEFT:
                        hist_idx = max(0, hist_idx - 1)
                        restore_state(env, history[hist_idx])
                    elif event.key == pygame.K_RIGHT:
                        hist_idx = min(len(history) - 1, hist_idx + 1)
                        restore_state(env, history[hist_idx])
                    elif event.key == pygame.K_UP:
                        scroll[1] = min(max_scroll_y, scroll[1] + SCROLL_STEP)
                    elif event.key == pygame.K_DOWN:
                        scroll[1] = max(-max_scroll_y, scroll[1] - SCROLL_STEP)

        if not paused and not done:
            n, pending_sim_time = sim_ticks_from_real_time(
                dt_real, sim_sec_duration, speed, pending_sim_time
            )
            for _ in range(n):
                action_p0 = agent0.choose_action(env, 0)
                action_p1 = agent1.choose_action(env, 1)
                result = env.step(action_p0, action_p1)
                history.append(capture_state(env))
                if len(history) > MAX_HISTORY:
                    history.pop(0)
                hist_idx = len(history) - 1
                if result.terminated or result.truncated:
                    done = True
                    if env.winner == 0:
                        end_msg = "Wygral P0 (niebieski)!"
                    elif env.winner == 1:
                        end_msg = "Wygral P1 (czerwony)!"
                    else:
                        end_msg = "Remis"
                    break

        screen.fill((25, 30, 28))
        scroll_t = (scroll[0], scroll[1])
        arena_oy = HUD_H + HAND_BAR_H + 8
        # Przesunięcie areny — miejsce na rękę P1 u góry i P0 na dole
        arena_scroll = (scroll_t[0], scroll_t[1] + arena_oy - MARGIN_Y)

        draw_player_hand(screen, env, 1, HUD_H + 4, font)
        draw_arena(screen, arena_scroll)
        draw_deploy_zones(screen, arena_scroll)

        for tower in env.towers:
            draw_tower(screen, tower, arena_scroll)
        for troop in env.troops:
            draw_troop(screen, troop, arena_scroll)
        draw_spell_effects(screen, env.spell_effects, arena_scroll)

        draw_player_hand(
            screen, env, 0, screen.get_height() - HAND_BAR_H - 8, font
        )

        status = "PAUZA" if paused else "GRA"
        if done:
            status = f"KONIEC — {end_msg}"
        tempo_label = (
            f"1s gry={sim_sec_duration:.2f}s real x{speed:.2f} | "
            f"kółka=rzut, żółte=most Hog | [ ] tempo"
        )
        if paused:
            info = (
                f"{status} | {tempo_label} | ←→ ({hist_idx + 1}/{len(history)}) | "
                f"↑↓ | Spacja | Q"
            )
        else:
            info = f"{status} | {tempo_label} | Spacja | Q"
        draw_hud(screen, env, font, info)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pygame — wizualizacja CR RL")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="skala UI (np. 0.5 = mniejsze okno); domyslnie UI_SCALE z pliku",
    )
    parser.add_argument(
        "--tempo",
        type=float,
        default=REAL_SECONDS_PER_SIM_SECOND,
        help="ile sekund rzeczywistych = 1 sekunda symulacji (np. 2.0 = wolniej)",
    )
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="sciezka do .zip PPO (domyslnie models/ppo_cr_best.zip)",
    )
    args = parser.parse_args()
    if args.scale is not None:
        apply_ui_scale(args.scale)
    run_pygame(
        seed=args.seed,
        real_seconds_per_sim_second=args.tempo,
        fps=args.fps,
        model_path=args.model,
    )
