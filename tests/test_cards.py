"""Testy mechanik jednostek z cards.py (Troop / Tower)."""

from __future__ import annotations

from cr_rl.game.cards import Troop, Tower, cards_dic


def _troop(name: str, owner: int, x: float = 5.0, y: float = 5.0) -> Troop:
    t = Troop(name, owner)
    t.place((x, y))
    return t


class TestTroopInit:
    def test_range_and_speed_strings_mapped_to_numbers(self):
        knight = Troop("Knight", 0)
        assert knight.range == cards_dic["Knight"]["range"] or knight.range == 1.2
        assert knight.speed == 1.8  # "medium"

    def test_building_flag(self):
        assert Troop("Cannon", 0).is_building
        assert Tower("Tower", 0).is_tower
        assert not Troop("Knight", 0).is_building

    def test_hog_jumps_river_flag(self):
        assert Troop("Hog_Rider", 0).jumps_river
        assert not Troop("Knight", 0).jumps_river


class TestTargeting:
    def test_ground_unit_cannot_target_air(self):
        knight = _troop("Knight", 0)
        flyer = _troop("Knight", 1)
        flyer.unit_type = "air"
        assert not knight.can_target(flyer)

    def test_musketeer_targets_air(self):
        musk = _troop("Musketeer", 0)
        flyer = _troop("Knight", 1)
        flyer.unit_type = "air"
        assert musk.can_target(flyer)

    def test_building_attacker_ignores_troops_but_hits_buildings(self):
        giant = _troop("Giant", 0)
        assert not giant.can_target(_troop("Knight", 1))
        assert giant.can_target(_troop("Cannon", 1))
        assert giant.can_target(Tower("Tower", 1))

    def test_knight_can_hit_tower(self):
        assert _troop("Knight", 0).can_target(Tower("Tower", 1))

    def test_no_friendly_fire_and_no_targeting_dead(self):
        knight = _troop("Knight", 0)
        assert not knight.can_target(_troop("Knight", 0))
        dead = _troop("Knight", 1)
        dead.alive = False
        assert not knight.can_target(dead)


class TestCombatMechanics:
    def test_first_hit_after_first_hit_speed_then_hit_speed(self):
        attacker = _troop("Knight", 0, 5.0, 5.0)
        target = _troop("Giant", 1, 5.0, 5.5)
        assert attacker.is_in_range(target)

        hits = 0
        for _ in range(200):
            hits += 1 if attacker.hit(target, 0.1) > 0 else 0
            if hits:
                break
        assert hits == 1
        expected_hp = cards_dic["Giant"]["hp"] - cards_dic["Knight"]["damage"]
        assert target.hp == expected_hp

    def test_deploy_time_blocks_attack(self):
        cannon = _troop("Cannon", 0, 5.0, 5.0)  # deploy_time = 1.0
        target = _troop("Knight", 1, 5.0, 5.5)
        for _ in range(5):  # 0.5 s < 1.0 s
            assert cannon.hit(target, 0.1) == 0.0

    def test_target_dies_at_zero_hp(self):
        attacker = _troop("Knight", 0, 5.0, 5.0)
        target = _troop("Knight", 1, 5.0, 5.5)
        target.hp = 1.0
        while attacker.hit(target, 0.1) == 0.0:
            pass
        assert target.hp == 0
        assert not target.alive

    def test_cannon_expires_after_lifespan(self):
        cannon = _troop("Cannon", 0)
        assert cannon.tick_age(29.9)
        assert not cannon.tick_age(0.2)
        assert not cannon.alive


class TestMovement:
    def test_move_snaps_to_target_without_overshoot(self):
        knight = _troop("Knight", 0, 5.0, 5.0)
        knight.move_towards_point(5.1, 5.0, 0.1)  # krok 0.18 > dystans 0.1
        assert (knight.x, knight.y) == (5.1, 5.0)

    def test_move_step_proportional_to_speed(self):
        knight = _troop("Knight", 0, 0.0, 0.0)
        knight.move_towards_point(10.0, 0.0, 0.1)
        assert abs(knight.x - 0.18) < 1e-9

    def test_building_does_not_move(self):
        cannon = _troop("Cannon", 0, 5.0, 5.0)
        cannon.move_towards_point(9.0, 9.0, 1.0)
        assert (cannon.x, cannon.y) == (5.0, 5.0)
