"""Card combat stats come from one committed data file that records its source and level."""

from __future__ import annotations

import json

from cr_rl.game.cards import CARD_STATS_PATH, DECK, cards_dic

TOURNAMENT_STANDARD_LEVEL = 11


def _load() -> dict:
    return json.loads(CARD_STATS_PATH.read_text(encoding="utf-8"))


def test_data_file_records_source_and_tournament_standard_level():
    data = _load()

    assert data["card_level"] == TOURNAMENT_STANDARD_LEVEL
    assert data["source"].startswith("https://")


def test_cards_dic_is_the_data_file():
    assert cards_dic == _load()["cards"]


def test_every_playable_card_and_tower_has_stats():
    for name in (*DECK, "Tower", "King_Tower"):
        assert name in cards_dic


def test_stats_are_numeric_so_nothing_is_resolved_in_code():
    for name, stats in cards_dic.items():
        for key in ("range", "speed"):
            assert isinstance(stats.get(key, 0), (int, float)), f"{name}.{key}"


def test_stats_are_at_tournament_standard():
    assert cards_dic["Fireball"]["damage"] == 688
    assert cards_dic["Musketeer"]["hp"] == 721
    assert cards_dic["Tower"]["hp"] == 3052
    assert cards_dic["King_Tower"]["hp"] == 4824


def test_every_troop_and_building_has_a_deploy_time_and_spells_have_none():
    for name in DECK:
        if cards_dic[name]["type"] == "spell":
            assert "deploy_time" not in cards_dic[name], name
        else:
            assert cards_dic[name]["deploy_time"] > 0, name


def test_every_body_has_a_collision_radius_and_every_mobile_unit_a_mass():
    for name, stats in cards_dic.items():
        if stats["type"] == "spell":
            continue
        assert stats["collision_radius"] > 0, name
        assert ("mass" in stats) == (stats["speed"] > 0), name


def test_new_cards_are_air_splash_and_area_spell():
    dragon = cards_dic["Baby_Dragon"]
    assert dragon["type"] == "air"
    assert set(dragon["targets"]) == {"ground", "air"}
    assert dragon["splash_radius"] > 0
    arrows = cards_dic["Arrows"]
    assert arrows["type"] == "spell" and arrows["radius"] > 0 and arrows["damage"] > 0
