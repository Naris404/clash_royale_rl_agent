"""env-v1 playtest: plays real matches in the web client and checks the mechanics.

The script drives a headless browser through the UI (pick a card, click the arena) and
reads every snapshot the server streams over the WebSocket. It reports one PASS/FAIL
line per checklist item (docs/playtest-env-v1.md) and saves screenshots. Run against a
server with the built frontend:

  python scripts/playtest_env_v1.py --url http://localhost:8000 --out playtest
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from math import hypot
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

from cr_rl.game.board import (
    ARENA_LENGTH,
    ARENA_WIDTH,
    BRIDGE_HALF_WIDTH,
    BRIDGE_LANE_X,
    DOUBLE_ELIXIR_TIME,
    ELIXIR_PER_SECOND,
    OVERTIME_TIME,
    REGULATION_TIME,
    RIVER_HALF_WIDTH,
    RIVER_Y,
)
from cr_rl.game.cards import DECK, cards_dic

# Records every server message and exposes the sockets so the script can set the speed.
TAP = """
window.__msgs = [];
window.__sockets = [];
const Native = window.WebSocket;
window.WebSocket = function (...args) {
  const ws = new Native(...args);
  window.__sockets.push(ws);
  ws.addEventListener("message", (e) => window.__msgs.push(JSON.parse(e.data)));
  return ws;
};
Object.assign(window.WebSocket, { OPEN: Native.OPEN, CONNECTING: Native.CONNECTING });
"""

MODES = {"coach": "Training with coach", "vs_rl": "Match vs AI", "vs_bot": "Match vs bot"}
PRINCESS_RANGE = cards_dic["Tower"]["range"]
KING_RANGE = cards_dic["King_Tower"]["range"]


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


class Match:
    """One browser match: plays through the UI and collects the server's messages."""

    def __init__(self, page: Page, url: str, mode: str = "vs_bot", speed: float = 4.0):
        self.page = page
        self.msgs: list[dict] = []
        page.goto(url)
        page.get_by_role("button", name=MODES[mode], exact=False).first.click()
        page.wait_for_selector(".arena-canvas canvas")
        page.wait_for_function("window.__msgs.some(m => m.type === 'snapshot')")
        self.set_speed(speed)

    def pump(self) -> None:
        self.msgs += self.page.evaluate("n => window.__msgs.slice(n)", len(self.msgs))

    @property
    def snaps(self) -> list[dict]:
        return [m for m in self.msgs if m["type"] == "snapshot"]

    @property
    def now(self) -> dict:
        self.pump()
        return self.snaps[-1]

    def set_speed(self, speed: float) -> None:
        self.page.evaluate(
            "v => window.__sockets.at(-1).send(JSON.stringify({type: 'speed', value: v}))", speed
        )

    def wait_game_time(self, seconds: float) -> None:
        target = self.now["time"] + seconds
        while self.now["time"] < target and not self.now["done"]:
            self.page.wait_for_timeout(60)

    def play(self, card: str, x: float, y: float) -> None:
        """Select the card in the hand bar, then click the arena tile (x, y)."""
        self.page.locator(".card", has_text=card.replace("_", " ")).first.click()
        box = self.page.locator(".arena-canvas").bounding_box()
        tile = min(box["width"] / (ARENA_WIDTH + 1), box["height"] / (ARENA_LENGTH + 1))
        px = box["x"] + (box["width"] - ARENA_WIDTH * tile) / 2 + x * tile
        py = box["y"] + (box["height"] - ARENA_LENGTH * tile) / 2 + (ARENA_LENGTH - y) * tile
        self.page.mouse.click(px, py)

    def screenshot(self, out: Path, name: str) -> None:
        self.page.screenshot(path=out / f"{name}.png")


def affordable(snap: dict) -> list[str]:
    return [c for c in snap["hand"] if cards_dic[c]["elisir"] <= snap["elixir"][0]]


def random_play(match: Match, rng: random.Random) -> None:
    """One random legal play: a troop on my half, or a spell on an enemy unit or tower."""
    snap = match.now
    options = affordable(snap)
    if not options:
        return
    card = rng.choice(options)
    if cards_dic[card]["type"] == "spell":
        targets = [(t["x"], t["y"]) for t in snap["troops"] + snap["towers"] if t["owner"] == 1 and t.get("alive", True)]
        x, y = rng.choice(targets)
        x, y = min(max(x, 1.2), ARENA_WIDTH - 1.2), min(max(y, 2.2), ARENA_LENGTH - 2.2)
    else:
        x, y = rng.uniform(1.5, ARENA_WIDTH - 1.5), rng.uniform(3.0, RIVER_Y - 1.5)
    match.play(card, x, y)


def play_match(match: Match, rng: random.Random, gap: float = 4.0, max_game_s: float = REGULATION_TIME + OVERTIME_TIME + 30.0) -> None:
    while not match.now["done"] and match.now["time"] < max_game_s:
        random_play(match, rng)
        match.wait_game_time(gap)


# --- checks over the recorded snapshots ---


def by_id(snap: dict) -> dict[int, dict]:
    return {t["id"]: t for t in snap["troops"]}


def touching(unit: dict, snap: dict) -> bool:
    """Whether another unit or tower overlaps `unit`, so a collision push may move it."""
    def radius(body: dict) -> float:
        return cards_dic[body.get("card", body.get("name"))]["collision_radius"]

    return any(
        other is not unit and hypot(other["x"] - unit["x"], other["y"] - unit["y"]) < radius(unit) + radius(other) + 0.1
        for other in snap["troops"] + snap["towers"]
    )


def check_deploy_time(snaps: list[dict]) -> Check:
    first_seen, moved_early = {}, []
    for prev, cur in zip(snaps, snaps[1:]):
        before = by_id(prev)
        for tid, troop in by_id(cur).items():
            first_seen.setdefault(tid, troop["deploy_remaining"])
            old = before.get(tid)
            moved = old and (old["x"], old["y"]) != (troop["x"], troop["y"])
            if moved and old["deploy_remaining"] > 0 and not touching(old, prev) and not touching(troop, cur):
                moved_early.append((cur["time"], troop["card"]))
    bad_start = {t: d for t, d in first_seen.items() if not 0.8 <= d <= 1.0}
    ok = not moved_early and not bad_start
    return Check("deploy time: 1.0 s, no walking while deploying (collision pushes allowed)", ok,
                 f"{len(first_seen)} units; walked early {moved_early[:3]}; bad start {dict(list(bad_start.items())[:3])}")


def check_river(snaps: list[dict]) -> Check:
    bad = []
    for snap in snaps:
        for t in snap["troops"]:
            flies = t["card"] == "Baby_Dragon" or cards_dic[t["card"]].get("jumps_river")
            wet = RIVER_Y - RIVER_HALF_WIDTH < t["y"] < RIVER_Y + RIVER_HALF_WIDTH
            on_bridge = any(abs(t["x"] - b) <= BRIDGE_HALF_WIDTH for b in BRIDGE_LANE_X)
            if wet and not on_bridge and not flies and cards_dic[t["card"]]["type"] != "building":
                bad.append((snap["time"], t["card"], t["x"], t["y"]))
    return Check("ground units never stand in the river off a bridge", not bad, f"{bad[:3]}")


def check_collisions(snaps: list[dict]) -> Check:
    worst = 0.0
    for snap in snaps:
        units = [t for t in snap["troops"] if t["deploy_remaining"] == 0]
        for i, a in enumerate(units):
            for b in units[i + 1:]:
                if (a["card"] == "Baby_Dragon") != (b["card"] == "Baby_Dragon"):
                    continue
                reach = cards_dic[a["card"]]["collision_radius"] + cards_dic[b["card"]]["collision_radius"]
                worst = max(worst, reach - hypot(a["x"] - b["x"], a["y"] - b["y"]))
    return Check("units do not stack (worst overlap under 0.35 tiles)", worst < 0.35, f"worst overlap {worst:.2f}")


def check_king_activation(snaps: list[dict]) -> Check:
    errors = []
    for snap in snaps:
        for player in (0, 1):
            own = [t for t in snap["towers"] if t["owner"] == player]
            king = next(t for t in own if t["name"] == "King_Tower")
            expected = king["hp"] < king["max_hp"] or any(not t["alive"] for t in own)
            if snap["king_active"][player] != expected:
                errors.append((snap["time"], player, snap["king_active"][player]))
    for prev, cur in zip(snaps, snaps[1:]):
        for player in (0, 1):
            if prev["king_active"][player] and not cur["king_active"][player]:
                errors.append((cur["time"], player, "went back to sleep"))
    woke = [p for p in (0, 1) if snaps[-1]["king_active"][p]]
    return Check("king tower wakes on damage or a fallen princess, then stays awake", not errors,
                 f"{errors[:3]}; awake at the end: {woke}")


def check_tower_reach(snaps: list[dict]) -> Check:
    worst = {"Tower": 0.0, "King_Tower": 0.0}
    sleeping_shots = []
    for snap in snaps:
        pos = {(t["x"], t["y"]): t for t in snap["towers"]}
        for hit in snap["hits"]:
            tower = pos.get((hit["from_x"], hit["from_y"]))
            if tower is None or not hit["ranged"]:
                continue
            worst[tower["name"]] = max(worst[tower["name"]], hypot(hit["x"] - hit["from_x"], hit["y"] - hit["from_y"]))
            if tower["name"] == "King_Tower" and not snap["king_active"][tower["owner"]]:
                sleeping_shots.append(snap["time"])
    # Range reaches the target's edge, so a shot may be as long as range + the largest unit radius.
    edge = max(c.get("collision_radius", 0) for c in cards_dic.values() if c.get("type") in ("ground", "air"))
    ok = worst["Tower"] <= PRINCESS_RANGE + edge and worst["King_Tower"] <= KING_RANGE + edge and not sleeping_shots
    return Check("tower shots stay inside range; sleeping king never shoots", ok,
                 f"farthest princess shot {worst['Tower']:.2f} (range {PRINCESS_RANGE}), "
                 f"king {worst['King_Tower']:.2f} (range {KING_RANGE}); sleeping shots {sleeping_shots[:3]}")


def check_spell_damage(snaps: list[dict]) -> Check:
    """A spell hit comes from the caster's king tower position and is not a ranged shot."""
    towers = {(t["x"], t["y"]) for t in snaps[0]["towers"]}
    full = {c: cards_dic[c]["damage"] for c in DECK if cards_dic[c]["type"] == "spell"}
    reduced = {c: cards_dic[c]["tower_damage"] for c in full}
    seen = {"on tower": set(), "on unit": set()}
    errors = []
    for snap in snaps:
        kings = {(t["x"], t["y"]) for t in snap["towers"] if t["name"] == "King_Tower"}
        for hit in snap["hits"]:
            if hit["ranged"] or (hit["from_x"], hit["from_y"]) not in kings or hit["damage"] not in {*full.values(), *reduced.values()}:
                continue
            on_tower = (hit["x"], hit["y"]) in towers
            expected = reduced if on_tower else full
            if hit["damage"] not in expected.values():
                errors.append((snap["time"], hit["damage"], "tower" if on_tower else "unit"))
            seen["on tower" if on_tower else "on unit"].add(hit["damage"])
    return Check("spells: reduced damage on towers, full damage on units", not errors,
                 f"{errors[:3]}; observed {seen}")


def phase_at(time: float) -> str:
    return "overtime" if time >= REGULATION_TIME else "double_elixir" if time >= DOUBLE_ELIXIR_TIME else "regulation"


def check_phases_and_elixir(snaps: list[dict]) -> Check:
    errors, rates = [], {"regulation": [], "double_elixir": [], "overtime": []}
    for snap in snaps:
        # Ticks of 0.1 s add up to just under 2:00 and 3:00, so a boundary lands one tick late.
        allowed = {phase_at(snap["time"]), phase_at(snap["time"] - 0.15)}
        if snap["phase"] not in allowed:
            errors.append((snap["time"], snap["phase"]))
    for prev, cur in zip(snaps, snaps[1:]):
        dt = cur["time"] - prev["time"]
        for player in (0, 1):
            gain = cur["elixir"][player] - prev["elixir"][player]
            if gain > 0 and cur["elixir"][player] < 9.9 and dt > 0:
                rates[prev["phase"]].append(gain / dt)
    summary = {p: round(sum(r) / len(r), 3) for p, r in rates.items() if r}
    normal = ELIXIR_PER_SECOND
    ok = not errors
    ok &= abs(summary.get("regulation", normal) - normal) < 0.01
    ok &= all(abs(summary.get(p, 2 * normal) - 2 * normal) < 0.02 for p in ("double_elixir", "overtime"))
    return Check("phases at 2:00 and 3:00; elixir 1/2.8 per s, doubled after 2:00", ok,
                 f"{errors[:3]}; mean regen per s {summary} (expected {normal:.3f} / {2 * normal:.3f})")


def check_crowns_and_end(snaps: list[dict]) -> Check:
    errors = []
    for snap in snaps:
        crowns = [0, 0]
        for t in snap["towers"]:
            if not t["alive"]:
                crowns[1 - t["owner"]] += 3 if t["name"] == "King_Tower" else 1
        crowns = [min(c, 3) for c in crowns]
        if crowns != snap["crowns"]:
            errors.append((snap["time"], crowns, snap["crowns"]))
    last = snaps[-1]
    weakest = [min((t["hp"] for t in last["towers"] if t["owner"] == p and t["alive"]), default=0) for p in (0, 1)]
    crowns = last["crowns"]
    if not last["done"]:
        errors.append("match did not finish")
    elif crowns[0] != crowns[1]:
        errors += [] if last["winner"] == (0 if crowns[0] > crowns[1] else 1) else ["winner != more crowns"]
    elif weakest[0] != weakest[1]:
        errors += [] if last["winner"] == (0 if weakest[0] > weakest[1] else 1) else ["tiebreak winner wrong"]
    else:
        errors += [] if last.get("winner") is None else ["draw expected"]
    return Check("crowns per tower; winner by crowns, then weakest tower, then draw", not errors,
                 f"{errors[:3]}; final crowns {crowns}, weakest {weakest}, winner {last.get('winner')}, t={last['time']}")


def check_cycle(snaps: list[dict]) -> Check:
    errors = [s["time"] for s in snaps
              if len(set(s["hand"])) != 4 or s["next_card"] in s["hand"] or not set(s["hand"]) | {s["next_card"]} <= set(DECK)]
    return Check("hand: 4 distinct Deck cards, next card not in hand", not errors, f"{errors[:3]}")


SNAPSHOT_CHECKS = (
    check_deploy_time, check_river, check_collisions, check_king_activation, check_tower_reach,
    check_spell_damage, check_phases_and_elixir, check_crowns_and_end, check_cycle,
)


def run(args: argparse.Namespace) -> list[Check]:
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    checks: list[Check] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for seed in range(args.matches):
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.add_init_script(TAP)
            match = Match(page, args.url, speed=args.speed)
            play_match(match, random.Random(seed))
            match.page.wait_for_timeout(500)
            match.screenshot(out, f"match{seed}_end")
            match.pump()
            print(f"match {seed}: t={match.now['time']:.1f} crowns={match.now['crowns']} winner={match.now.get('winner')}")
            checks += [Check(f"[match {seed}] {c.name}", c.ok, c.detail) for c in (f(match.snaps) for f in SNAPSHOT_CHECKS)]
            page.close()
        browser.close()
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--out", default="playtest")
    parser.add_argument("--matches", type=int, default=3)
    parser.add_argument("--speed", type=float, default=4.0)
    checks = run(parser.parse_args())
    for c in checks:
        print(f"{'PASS' if c.ok else 'FAIL'}  {c.name}\n      {c.detail}")
    raise SystemExit(0 if all(c.ok for c in checks) else 1)


if __name__ == "__main__":
    main()
