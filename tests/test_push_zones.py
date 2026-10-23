"""Spell-push-L / spell-push-R (zones 11 and 12): spells aimed at the best enemy cluster in a lane."""

from __future__ import annotations

import numpy as np
import pytest

from cr_rl.coach.evaluator import CoachEngine, decode_action, nearest_zone
from cr_rl.coach.inference import PolicyEstimate
from cr_rl.coach.explainer import build_reason, format_hint
from cr_rl.game.board import DEPLOY_ZONES, HAND_SIZE, NUM_ACTIONS, NUM_ZONES
from cr_rl.game.cards import cards_dic
from scenario import new_board, place_unit, set_hand

PUSH_L = 11
PUSH_R = 12
FIREBALL = cards_dic["Fireball"]
ARROWS = cards_dic["Arrows"]


def _caster(*hand: str, player: int = 0):
    board = new_board()
    set_hand(board, player, list(hand))
    board.elixir[player] = 10.0
    return board


def test_layout_has_13_zones_and_53_actions():
    assert NUM_ZONES == 13
    assert NUM_ACTIONS == 53 == 1 + HAND_SIZE * NUM_ZONES
    assert [z.name for z in DEPLOY_ZONES[0][PUSH_L:]] == ["spell-push-L", "spell-push-R"]
    assert [z.name for z in DEPLOY_ZONES[1][PUSH_L:]] == ["spell-push-L", "spell-push-R"]


def test_actions_map_to_slot_and_zone_by_divmod_13():
    board = _caster("Knight", "Giant", "Cannon", "Fireball")
    assert board.action_to_card_zone(1, 0) == ("Knight", 0)
    assert board.action_to_card_zone(14, 0) == ("Giant", 0)
    assert board.action_to_card_zone(52, 0) == ("Fireball", PUSH_R)
    assert board.action_to_card_zone(53, 0) == (None, None)


class TestAim:
    def test_aims_at_the_cluster_that_takes_the_most_damage(self):
        board = _caster("Fireball")
        place_unit(board, "Giant", 1, 3.0, 20.0)  # one unit: 688
        place_unit(board, "Knight", 1, 4.0, 12.0)  # two units: 2 x 688
        place_unit(board, "Knight", 1, 4.5, 12.0)
        hp = {id(t): t.hp for t in board.troops}

        assert board.play_card(0, "Fireball", PUSH_L)

        hit = [t for t in board.troops if t.hp < hp[id(t)]]
        assert [t.name for t in hit] == ["Knight", "Knight"]
        assert board.spell_effects[0].x == 4.0

    def test_damage_is_capped_at_the_hp_a_unit_has_left(self):
        board = _caster("Fireball")
        for _ in range(3):  # a crowd of nearly dead units is worth 3 x 10, less than one healthy Knight
            place_unit(board, "Knight", 1, 3.0, 20.0, hp=10.0)
        healthy = place_unit(board, "Knight", 1, 4.0, 12.0)

        assert board.play_card(0, "Fireball", PUSH_L)

        assert healthy.hp == healthy.max_hp - FIREBALL["damage"]

    def test_an_enemy_princess_tower_in_the_radius_adds_crown_tower_damage(self):
        board = _caster("Fireball")
        place_unit(board, "Knight", 1, 3.0, 14.0)  # 688 on its own
        near_tower = place_unit(board, "Knight", 1, 3.5, 22.5)  # 688 + 159 next to the left princess tower
        tower = next(t for t in board.towers if t.owner == 1 and t.name == "Tower" and t.x < 9)

        assert board.play_card(0, "Fireball", PUSH_L)

        assert near_tower.hp < near_tower.max_hp
        assert tower.hp == tower.max_hp - FIREBALL["tower_damage"]

    def test_tower_damage_is_capped_at_the_tower_hp(self):
        board = _caster("Fireball")
        place_unit(board, "Knight", 1, 3.0, 14.0)  # 688 on its own
        place_unit(board, "Knight", 1, 3.5, 22.5, hp=100.0)  # 100 + min(159, 120) = 220: worse
        tower = next(t for t in board.towers if t.owner == 1 and t.name == "Tower" and t.x < 9)
        tower.hp = 120.0
        far_unit = board.troops[0]

        assert board.play_card(0, "Fireball", PUSH_L)

        assert far_unit.hp == far_unit.max_hp - FIREBALL["damage"]
        assert tower.hp == 120.0

    @pytest.mark.parametrize("first_y", [12.0, 19.0])
    def test_a_tie_goes_to_the_earliest_placed_unit(self, first_y):
        board = _caster("Arrows")
        first = place_unit(board, "Knight", 1, 3.0, first_y)  # two clusters of one, out of each other's reach
        second = place_unit(board, "Knight", 1, 3.0, 31.0 - first_y)

        assert board.play_card(0, "Arrows", PUSH_L)

        assert first.hp == first.max_hp - ARROWS["damage"]
        assert second.hp == second.max_hp

    def test_a_tie_follows_placement_order_not_position_order(self):
        board = _caster("Arrows")
        far = place_unit(board, "Knight", 1, 5.0, 19.0)  # placed first, yet further right and further back
        near = place_unit(board, "Knight", 1, 2.0, 12.0)

        assert board.play_card(0, "Arrows", PUSH_L)

        assert far.hp == far.max_hp - ARROWS["damage"]
        assert near.hp == near.max_hp

    def test_a_building_is_a_valid_target(self):
        board = _caster("Fireball")
        cannon = place_unit(board, "Cannon", 1, 14.0, 20.0)

        assert board.play_card(0, "Fireball", PUSH_R)

        assert cannon.hp == cannon.max_hp - FIREBALL["damage"]

    def test_lane_is_decided_by_x_against_the_centre_line(self):
        board = _caster("Fireball")
        place_unit(board, "Knight", 1, 8.9, 20.0)
        place_unit(board, "Knight", 1, 9.1, 20.0)
        mask = board.valid_action_mask(0)
        assert mask[1 + PUSH_L] and mask[1 + PUSH_R]

        board = _caster("Fireball")
        place_unit(board, "Knight", 1, 8.9, 20.0)
        mask = board.valid_action_mask(0)
        assert mask[1 + PUSH_L] and not mask[1 + PUSH_R]

    def test_the_aim_stays_inside_the_spell_area(self):
        board = _caster("Fireball")
        place_unit(board, "Knight", 1, 0.4, 20.0)  # hugging the side wall

        assert board.play_card(0, "Fireball", PUSH_L)

    def test_player1_aims_in_the_same_lane_on_its_own_side(self):
        board = _caster("Fireball", player=1)
        target = place_unit(board, "Knight", 0, 3.0, 12.0)
        place_unit(board, "Knight", 0, 15.0, 12.0)

        assert board.play_card(1, "Fireball", PUSH_L)

        assert target.hp == target.max_hp - FIREBALL["damage"]

    def test_own_units_are_not_targets(self):
        board = _caster("Fireball")
        place_unit(board, "Knight", 0, 3.0, 8.0)
        assert not board.valid_action_mask(0)[1 + PUSH_L]


class TestMasking:
    def test_push_zones_are_masked_while_the_lane_holds_no_enemy_unit_or_building(self):
        board = _caster("Fireball")  # towers stand in both lanes but never unmask a push zone
        mask = board.valid_action_mask(0)
        assert not mask[1 + PUSH_L] and not mask[1 + PUSH_R]
        assert mask[1 + 8] and mask[1 + 9] and mask[1 + 10]  # finishing a tower stays on zones 8-10
        assert not board.play_card(0, "Fireball", PUSH_L)
        assert board.elixir[0] == 10.0

    def test_a_push_zone_opens_with_an_enemy_in_its_lane_only(self):
        board = _caster("Fireball")
        place_unit(board, "Knight", 1, 3.0, 20.0)
        mask = board.valid_action_mask(0)
        assert mask[1 + PUSH_L] and not mask[1 + PUSH_R]

    def test_push_zones_accept_spells_only(self):
        board = _caster("Knight", "Giant", "Cannon", "Fireball")
        place_unit(board, "Knight", 1, 3.0, 20.0)
        place_unit(board, "Knight", 1, 15.0, 20.0)
        mask = board.valid_action_mask(0)
        for slot, card in enumerate(board.get_hand(0)):
            for zone in (PUSH_L, PUSH_R):
                assert mask[1 + slot * NUM_ZONES + zone] == (card == "Fireball"), (card, zone)

    def test_spells_are_masked_off_the_troop_zones(self):
        board = _caster("Fireball")
        place_unit(board, "Knight", 1, 3.0, 20.0)
        mask = board.valid_action_mask(0)
        assert not mask[1 : 1 + 8].any()


class TestCoach:
    def test_a_human_spell_maps_to_a_push_zone_by_its_aim_point_while_it_is_legal(self):
        board = _caster("Fireball")
        place_unit(board, "Knight", 1, 3.0, 12.0)

        assert nearest_zone(board, 0, "Fireball", 3.0, 12.5) == PUSH_L
        assert nearest_zone(board, 0, "Fireball", 14.0, 12.5) != PUSH_R  # that lane is empty: zone illegal

    def test_without_enemies_a_spell_maps_to_the_static_spell_zones(self):
        board = _caster("Fireball")
        assert nearest_zone(board, 0, "Fireball", 3.0, 12.0) == 8  # spell-tower-L

    def test_a_spell_on_a_tower_stays_on_the_tower_zone(self):
        board = _caster("Fireball")
        place_unit(board, "Knight", 1, 3.0, 12.0)
        assert nearest_zone(board, 0, "Fireball", 3.5, 24.0) == 8

    def test_troops_never_map_to_push_zones(self):
        board = _caster("Knight")
        place_unit(board, "Knight", 1, 3.0, 12.0)
        assert nearest_zone(board, 0, "Knight", 3.0, 12.0) < 8

    def test_decoding_a_push_action_names_the_zone(self):
        board = _caster("Knight", "Giant", "Cannon", "Fireball")
        action = 1 + 3 * NUM_ZONES + PUSH_R
        assert decode_action(board, 0, action) == ("Fireball", 3, PUSH_R)

    @pytest.mark.parametrize(
        ("lang", "zone", "words"),
        [
            ("en", PUSH_L, ("push", "left")),
            ("en", PUSH_R, ("push", "right")),
            ("pl", PUSH_L, ("push", "lewa")),
            ("pl", PUSH_R, ("push", "prawa")),
        ],
    )
    def test_hints_name_the_enemy_push_and_its_lane(self, lang, zone, words):
        hint = format_hint("Fireball", zone, 0.5, lang=lang)
        assert all(word in hint for word in words), hint

    @pytest.mark.parametrize(
        ("lang", "words"),
        [("en", ("left", "Knight ×2")), ("pl", ("lewa", "Knight ×2"))],
    )
    def test_the_reason_names_the_lane_and_the_target_cluster(self, lang, words):
        board = _caster("Fireball")
        place_unit(board, "Knight", 1, 4.0, 12.0)
        place_unit(board, "Knight", 1, 4.5, 12.0)
        place_unit(board, "Giant", 1, 3.0, 20.0)

        reason = build_reason(board, 0, "Fireball", PUSH_L, lang=lang)

        assert all(word in reason for word in words), reason
        assert "Giant" not in reason

    def test_a_push_suggestion_carries_the_aim_point_for_the_arena_marker(self):
        class StubInspector:
            available = True

            def estimate(self, board, player):
                action = 1 + PUSH_L  # slot 0 (Fireball) at spell-push-L
                probs = np.zeros(NUM_ACTIONS)
                probs[action] = 1.0
                return PolicyEstimate(action=action, probs=probs, value=0.0, top_k=[(action, 1.0)])

        board = _caster("Fireball")
        place_unit(board, "Knight", 1, 4.0, 12.0)

        suggestion = CoachEngine(StubInspector()).suggest(board)

        assert (suggestion.zone, suggestion.aim) == (PUSH_L, (4.0, 12.0))
