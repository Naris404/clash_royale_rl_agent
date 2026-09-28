"""Objaśnienia ruchów trenera — regułowe konteksty + szablony PL/EN.

Sieć neuronowa mówi CO zagrać (akcja + prawdopodobieństwo); ten moduł dokłada
krótkie DLACZEGO na podstawie czytelnych cech stanu (zagrożenie, pełny eliksir,
spokojna plansza) — moduł analityczno-trenerski z opisu pracy dyplomowej.
"""

from __future__ import annotations

from typing import Optional

from cr_rl.game.board import RIVER_Y, Board

ZONE_NAMES = {
    "pl": ("lewa aleja", "środek", "prawa aleja"),
    "en": ("left lane", "center", "right lane"),
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


def zone_name(zone_idx: int, lang: str = "pl") -> str:
    names = ZONE_NAMES.get(lang, ZONE_NAMES["pl"])
    if 0 <= zone_idx < len(names):
        return names[zone_idx]
    return str(zone_idx)


def build_reason(
    board: Board,
    player: int,
    card: Optional[str],
    *,
    lang: str = "pl",
) -> str:
    """Krótki kontekst dla sugerowanego ruchu (lub noop, gdy card=None)."""
    lang = lang if lang in ("pl", "en") else "pl"

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
    lang: str = "pl",
) -> str:
    """Pełny tekst podpowiedzi, np. 'Zagraj Knight — lewa aleja (pewność 62%)'."""
    if card is None:
        return _REASONS["wait"][lang if lang in ("pl", "en") else "pl"]
    zone = zone_name(zone_idx, lang) if zone_idx is not None else ""
    confidence = f" ({'pewność' if lang == 'pl' else 'confidence'} {prob:.0%})" if prob is not None else ""
    if lang == "en":
        return f"Play {card} — {zone}{confidence}"
    return f"Zagraj {card} — {zone}{confidence}"
