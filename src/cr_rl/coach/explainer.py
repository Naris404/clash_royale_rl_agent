"""Coach move explanations — rule-based contexts + PL/EN templates.

The neural network says WHAT to play (action + probability); this module adds
a short WHY based on readable state features (threat, full elixir,
calm board) — the analytics/coaching module from the thesis description.
"""

from __future__ import annotations

from typing import Optional

from cr_rl.game.board import DEPLOY_ZONES, RIVER_Y, Board

# Keyed by DeployZone.name so the labels cannot drift from the zone layout.
ZONE_NAMES = {
    "pl": {
        "back-L": "tył, lewa aleja",
        "back-R": "tył, prawa aleja",
        "mid-L": "środek, lewa aleja",
        "mid-R": "środek, prawa aleja",
        "bridge-L": "lewy most",
        "bridge-R": "prawy most",
        "pull-L": "lewy punkt przyciągania (pull)",
        "pull-R": "prawy punkt przyciągania (pull)",
        "spell-tower-L": "lewa wieża wroga",
        "spell-tower-R": "prawa wieża wroga",
        "spell-king": "król wroga",
    },
    "en": {
        "back-L": "back, left lane",
        "back-R": "back, right lane",
        "mid-L": "mid, left lane",
        "mid-R": "mid, right lane",
        "bridge-L": "left bridge",
        "bridge-R": "right bridge",
        "pull-L": "left pull zone",
        "pull-R": "right pull zone",
        "spell-tower-L": "enemy left tower",
        "spell-tower-R": "enemy right tower",
        "spell-king": "enemy king",
    },
}

DEFENSIVE_CARDS = frozenset({"Knight", "Cannon", "Musketeer"})
OFFENSIVE_CARDS = frozenset({"Hog_Rider", "Giant"})

_REASONS = {
    "defend_hog": {
        "pl": "Obrona — Hog Rider wroga na Twojej połowie",
        "en": "Defense — enemy Hog Rider on your half",
    },
    "defend_push": {
        "pl": "Obrona — jednostki wroga na Twojej połowie",
        "en": "Defense — enemy units on your half",
    },
    "spell_value": {
        "pl": "Fireball — korzystna wymiana obrażeniami obszarowymi",
        "en": "Fireball — favorable area-damage trade",
    },
    "elixir_leak": {
        "pl": "Pełny eliksir — nie marnuj, zagraj kartę",
        "en": "Elixir full — don't waste it, play a card",
    },
    "pressure": {
        "pl": "Spokojna plansza — wywieraj presję na aleję",
        "en": "Quiet board — pressure a lane",
    },
    "wait": {
        "pl": "Czekaj — zbierz eliksir na lepszy moment",
        "en": "Wait — save elixir for a better moment",
    },
    "generic": {
        "pl": "Propozycja polityki RL",
        "en": "RL policy suggestion",
    },
}


def zone_name(zone_idx: int, lang: str = "en") -> str:
    if not 0 <= zone_idx < len(DEPLOY_ZONES[0]):
        return str(zone_idx)
    return ZONE_NAMES.get(lang, ZONE_NAMES["en"])[DEPLOY_ZONES[0][zone_idx].name]


def build_reason(
    board: Board,
    player: int,
    card: Optional[str],
    *,
    lang: str = "en",
) -> str:
    """Short context for a suggested move (or noop when card=None)."""
    lang = lang if lang in ("pl", "en") else "en"

    if card is None:
        return _REASONS["wait"][lang]

    enemies = [t for t in board.troops if t.alive and t.owner != player]
    on_my_side = [
        e for e in enemies
        if (e.y < RIVER_Y if player == 0 else e.y > RIVER_Y) and not e.is_building
    ]

    if card == "Fireball" and enemies:
        return _REASONS["spell_value"][lang]
    if on_my_side and card in DEFENSIVE_CARDS:
        if any(e.name == "Hog_Rider" for e in on_my_side):
            return _REASONS["defend_hog"][lang]
        return _REASONS["defend_push"][lang]
    if board.elixir[player] >= 9.0:
        return _REASONS["elixir_leak"][lang]
    if not on_my_side and card in OFFENSIVE_CARDS:
        return _REASONS["pressure"][lang]
    return _REASONS["generic"][lang]


def format_hint(
    card: Optional[str],
    zone_idx: Optional[int],
    prob: Optional[float],
    *,
    lang: str = "en",
) -> str:
    """Full hint text, e.g. 'Play Knight — left lane (confidence 62%)'."""
    if card is None:
        return _REASONS["wait"][lang if lang in ("pl", "en") else "en"]
    zone = zone_name(zone_idx, lang) if zone_idx is not None else ""
    confidence = f" ({'pewność' if lang == 'pl' else 'confidence'} {prob:.0%})" if prob is not None else ""
    if lang == "en":
        return f"Play {card} — {zone}{confidence}"
    return f"Zagraj {card} — {zone}{confidence}"
