"""Board tests — game mechanics, observations, rewards, determinism."""

from __future__ import annotations

import numpy as np
import pytest

from cr_rl.game.board import (
    ARENA_LENGTH,
    ARENA_WIDTH,
    BRIDGE_HALF_WIDTH,
    BRIDGE_LANE_X,
    DEPLOY_ZONES,
    ELIXIR_PER_SECOND,
    HAND_SIZE,
    MATCH_TIME_LIMIT,
    MAX_ELIXIR,
    NUM_ACTIONS,
    NUM_PLAYABLE_CARDS,
    NUM_ZONES,
    OBS_DIM,
    OBS_GLOBAL_FEATURES,
    OBS_HAND_BASE,
    OBS_UNIT_BASE,
    RIVER_HALF_WIDTH,
    RIVER_Y,
    TICK_DT,
    Board,
)
from cr_rl.game.cards import CARD_TO_ID, PLAYABLE_CARDS, Troop, cards_dic
from scenario import new_board, place_unit, set_hand

ZONE_NAMES = [z.name for z in DEPLOY_ZONES[0]]


def _step_seconds(board: Board, seconds: float) -> None:
    for _ in range(round(seconds / TICK_DT)):
        board.step(0, action_p1=0)


def _hits_from(board: Board, unit: Troop) -> int:
    return sum(1 for e in board.hit_events if (e.from_x, e.from_y) == (unit.x, unit.y))


class TestHands:
    def test_initial_hand_is_4_distinct_deck_cards(self):
        board = new_board()
        for player in (0, 1):
            hand = board.get_hand(player)
            assert len(hand) == HAND_SIZE
            assert len(set(hand)) == HAND_SIZE
            assert set(hand) <= set(PLAYABLE_CARDS)

    def test_played_card_returns_after_the_queued_cards(self):
        board = new_board()
        set_hand(board, 0, ["Knight", "Giant", "Cannon", "Musketeer"])  # queue: Hog_Rider, Fireball

        def play(card: str) -> list[str]:
            board.elixir[0] = 10.0
            assert board.play_card(0, card, 0)
            return board.get_hand(0)

        assert play("Knight") == ["Giant", "Cannon", "Musketeer", "Hog_Rider"]
        assert play("Giant") == ["Cannon", "Musketeer", "Hog_Rider", "Fireball"]
        assert play("Cannon") == ["Musketeer", "Hog_Rider", "Fireball", "Knight"]


class TestElixir:
    def test_regen_per_tick(self):
        board = new_board()
        before = board.elixir[0]
        board.step(0, action_p1=0)
        assert board.elixir[0] == pytest.approx(before + ELIXIR_PER_SECOND * TICK_DT)

    def test_capped_at_max(self):
        board = new_board()
        board.elixir[0] = MAX_ELIXIR
        board.step(0, action_p1=0)
        assert board.elixir[0] == MAX_ELIXIR

    def test_play_deducts_cost(self):
        board = new_board()
        board.elixir[0] = 10.0
        set_hand(board, 0, ["Knight", "Giant", "Cannon", "Musketeer"])
        assert board.play_card(0, "Knight", 0)
        assert board.elixir[0] == pytest.approx(7.0)


class TestPlayLegality:
    def test_rejects_card_not_in_hand(self):
        board = new_board()
        board.elixir[0] = 10.0
        set_hand(board, 0, ["Knight", "Giant", "Cannon", "Musketeer"])
        assert not board.play_card(0, "Fireball", 0)

    def test_rejects_when_not_enough_elixir(self):
        board = new_board()
        board.elixir[0] = 2.0
        set_hand(board, 0, ["Knight", "Giant", "Cannon", "Musketeer"])
        assert not board.play_card(0, "Giant", 0)  # costs 5

    def test_rejects_position_outside_deploy_zone(self):
        board = new_board()
        board.elixir[0] = 10.0
        set_hand(board, 0, ["Knight", "Giant", "Cannon", "Musketeer"])
        assert not board.play_card_at(0, "Knight", 9.0, RIVER_Y + 5.0)  # enemy side
        assert not board.play_card_at(0, "Knight", 0.0, 5.0)  # outside the arena

    def test_action_to_card_zone_mapping(self):
        board = new_board()
        set_hand(board, 0, ["Knight", "Giant", "Cannon", "Musketeer"])
        assert board.action_to_card_zone(0, 0) == (None, None)
        assert board.action_to_card_zone(1, 0) == ("Knight", 0)
        assert board.action_to_card_zone(2, 0) == ("Knight", 1)
        assert board.action_to_card_zone(12, 0) == ("Giant", 0)
        assert board.action_to_card_zone(44, 0) == ("Musketeer", 10)
        assert board.action_to_card_zone(45, 0) == (None, None)  # out of range


TROOP_ZONES_P0 = {
    "back-L": (3.0, 4.5),
    "back-R": (14.0, 4.5),
    "mid-L": (3.0, 9.0),
    "mid-R": (14.0, 9.0),
    "bridge-L": (3.0, 14.0),
    "bridge-R": (14.0, 14.0),
    "pull-L": (5.5, 9.5),
    "pull-R": (12.5, 9.5),
}
SPELL_ZONES_P0 = {
    "spell-tower-L": (3.5, 24.0),
    "spell-tower-R": (14.5, 24.0),
    "spell-king": (9.0, 30.0),
}


class TestDeployZones:
    def test_eleven_zones_per_side_with_pinned_coordinates(self):
        assert NUM_ZONES == 11
        zones = {z.name: z for z in DEPLOY_ZONES[0]}
        assert len(DEPLOY_ZONES[0]) == len(DEPLOY_ZONES[1]) == 11
        for name, (x, y) in {**TROOP_ZONES_P0, **SPELL_ZONES_P0}.items():
            assert (zones[name].x, zones[name].y) == (x, y)

    def test_player1_zones_mirror_player0_along_y(self):
        for z0, z1 in zip(DEPLOY_ZONES[0], DEPLOY_ZONES[1]):
            assert z1.name == z0.name
            assert (z1.x, z1.y) == (z0.x, ARENA_LENGTH - z0.y)
            assert z1.card_types == z0.card_types

    def test_zones_carry_allowed_card_types(self):
        for z in DEPLOY_ZONES[0]:
            if z.name in SPELL_ZONES_P0:
                assert z.card_types == frozenset({"spell"})
            else:
                assert z.card_types == frozenset({"ground", "building"})

    def test_num_actions_is_noop_plus_slots_times_zones(self):
        assert NUM_ACTIONS == 45 == 1 + HAND_SIZE * NUM_ZONES

    @pytest.mark.parametrize("player", [0, 1])
    def test_troop_zones_are_legal_placements(self, player):
        board = new_board()
        board.elixir[player] = 10.0
        set_hand(board, player, ["Knight", "Giant", "Cannon", "Musketeer"])
        for z in DEPLOY_ZONES[player]:
            if z.name in TROOP_ZONES_P0:
                assert board.can_play_card_at(player, "Knight", z.x, z.y), z.name

    def test_play_card_uses_absolute_zone_coordinates(self):
        board = new_board()
        board.elixir[0] = 10.0
        set_hand(board, 0, ["Cannon", "Giant", "Knight", "Musketeer"])
        assert board.play_card(0, "Cannon", ZONE_NAMES.index("pull-R"))
        assert (board.troops[0].x, board.troops[0].y) == (12.5, 9.5)

    def test_spell_cast_at_spell_zone_and_rejected_on_troop_zone(self):
        board = new_board()
        board.elixir[0] = 10.0
        set_hand(board, 0, ["Fireball", "Giant", "Cannon", "Musketeer"])
        assert not board.play_card(0, "Fireball", ZONE_NAMES.index("mid-L"))
        assert board.play_card(0, "Fireball", ZONE_NAMES.index("spell-tower-L"))
        effect = board.spell_effects[0]
        assert (effect.x, effect.y) == (3.5, 24.0)

    def test_troop_rejected_on_spell_zone(self):
        board = new_board()
        board.elixir[0] = 10.0
        set_hand(board, 0, ["Knight", "Giant", "Cannon", "Musketeer"])
        assert not board.play_card(0, "Knight", ZONE_NAMES.index("spell-king"))


class TestSpells:
    def test_fireball_damages_enemies_in_radius(self):
        board = new_board()
        board.elixir[0] = 10.0
        set_hand(board, 0, ["Fireball", "Giant", "Cannon", "Musketeer"])
        enemy = place_unit(board, "Giant", 1, 9.0, 20.0)
        ally = place_unit(board, "Knight", 0, 9.0, 20.0)
        far_enemy = place_unit(board, "Knight", 1, 2.0, 25.0)

        assert board.play_card_at(0, "Fireball", 9.0, 20.0)
        assert enemy.hp == cards_dic["Giant"]["hp"] - cards_dic["Fireball"]["damage"]
        assert ally.hp == ally.max_hp  # no friendly fire
        assert far_enemy.hp == far_enemy.max_hp  # outside the radius

    def test_fireball_kills_low_hp_unit(self):
        board = new_board()
        board.elixir[0] = 10.0
        set_hand(board, 0, ["Fireball", "Giant", "Cannon", "Musketeer"])
        enemy = place_unit(board, "Knight", 1, 9.0, 20.0, hp=100.0)
        board.play_card_at(0, "Fireball", 9.0, 20.0)
        assert not enemy.alive

    def test_fireball_cycles_hand_and_costs_elixir(self):
        board = new_board()
        board.elixir[0] = 10.0
        set_hand(board, 0, ["Fireball", "Giant", "Cannon", "Musketeer"])
        assert board.play_card_at(0, "Fireball", 9.0, 20.0)
        assert board.elixir[0] == pytest.approx(6.0)
        assert "Fireball" not in board.get_hand(0)


class TestRiverAndBridges:
    @staticmethod
    def _walk_across(board: Board, troop: Troop, max_steps: int = 600) -> list[tuple[float, float]]:
        """Steps the simulation until the unit crosses the river; returns the path."""
        direction = 1.0 if troop.owner == 0 else -1.0
        far_bank = RIVER_Y + direction * RIVER_HALF_WIDTH
        path = [(troop.x, troop.y)]
        for _ in range(max_steps):
            board.step(0, action_p1=0)
            path.append((troop.x, troop.y))
            if direction * (troop.y - far_bank) > 0:
                return path
        pytest.fail(f"{troop.name} did not cross the river: {path[-1]}")

    def test_diagonal_path_shorter_than_l_path(self):
        board = new_board()
        knight = place_unit(board, "Knight", 0, 8.0, 6.0)
        path = self._walk_across(board, knight)

        length = sum(np.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(path, path[1:]))
        bx = BRIDGE_LANE_X[0]
        l_path = abs(8.0 - bx) + (path[-1][1] - 6.0)
        assert length < l_path - 2.0

        (x0, y0), (x1, y1) = path[0], path[1]
        assert x1 < x0 and y1 > y0  # diagonal from the first step, not sideways first

    @pytest.mark.parametrize("owner", [0, 1])
    @pytest.mark.parametrize("start", [(8.0, 6.0), (1.5, 10.0), (16.5, 12.0), (9.0, 14.0), (5.0, 13.0)])
    def test_river_crossed_only_on_bridge_without_teleports(self, owner, start):
        board = new_board()
        x, y = start if owner == 0 else (ARENA_WIDTH - start[0], ARENA_LENGTH - start[1])
        knight = place_unit(board, "Knight", owner, x, y)
        path = self._walk_across(board, knight)

        max_step = knight.speed * TICK_DT + 1e-6
        for (x0, y0), (x1, y1) in zip(path, path[1:]):
            assert np.hypot(x1 - x0, y1 - y0) <= max_step  # no "going back" across the river
            if (RIVER_Y - RIVER_HALF_WIDTH) < y1 < (RIVER_Y + RIVER_HALF_WIDTH):
                assert any(abs(x1 - bx) <= BRIDGE_HALF_WIDTH for bx in BRIDGE_LANE_X)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "A ground unit within 0.05 tiles short of the far bank still routes to that "
            "bank, and moves under 0.05 tiles are skipped, so it freezes there."
        ),
    )
    def test_unit_just_past_the_far_bank_keeps_walking(self):
        board = new_board()
        knight = place_unit(board, "Knight", 0, 3.0, RIVER_Y + 1.48)
        _step_seconds(board, 1.0)
        assert knight.y > RIVER_Y + 2.0

    def test_hog_rider_jumps_river_in_straight_line(self):
        board = new_board()
        hog = place_unit(board, "Hog_Rider", 0, 9.0, 12.0)
        path = self._walk_across(board, hog)
        assert any(
            (RIVER_Y - RIVER_HALF_WIDTH) < y < (RIVER_Y + RIVER_HALF_WIDTH) and abs(x - 9.0) < 1.0
            for x, y in path
        )


class TestTargetingAndCombat:
    def test_sight_range_limits_aggro(self):
        board = new_board()
        knight = place_unit(board, "Knight", 0, 1.0, 4.0)
        place_unit(board, "Knight", 1, 17.0, 28.0)  # far outside sight range
        assert board.find_nearest_target(knight) is None
        near = place_unit(board, "Knight", 1, knight.x + 3.0, knight.y)
        assert board.find_nearest_target(knight) is near

    def test_tower_shoots_only_in_range(self):
        board = new_board()
        tower = next(
            t for t in board.towers
            if t.owner == 0 and t.name == "Tower" and t.x < ARENA_WIDTH / 2
        )
        enemy = place_unit(board, "Knight", 1, tower.x, tower.y + 12.0)
        assert board.find_tower_target(tower) is None
        enemy.place((tower.x, tower.y + 5.0))
        assert board.find_tower_target(tower) is enemy

    def test_giant_walks_past_troops_to_hit_a_building(self):
        board = new_board()
        giant = place_unit(board, "Giant", 0, 3.0, 19.0)
        knight = place_unit(board, "Knight", 1, 3.0, 20.0)  # right in the Giant's way
        enemy_tower = next(t for t in board.towers if t.owner == 1 and t.name == "Tower" and t.x < 9.0)

        for _ in range(200):
            board.step(0, action_p1=0)
            giant_hits = [e for e in board.hit_events if (e.from_x, e.from_y) == (giant.x, giant.y)]
            if giant_hits:
                break
        assert knight.hp == knight.max_hp
        assert [(e.x, e.y) for e in giant_hits] == [(enemy_tower.x, enemy_tower.y)]

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Troops that target ground ignore buildings (only towers count), so a Knight "
            "walks past an enemy Cannon; in the real game ground targeting includes buildings."
        ),
    )
    def test_knight_attacks_enemy_cannon_in_its_way(self):
        board = new_board()
        cannon = place_unit(board, "Cannon", 1, 9.0, 20.0)
        knight = place_unit(board, "Knight", 0, 9.0, 18.0)
        assert board.find_nearest_target(knight) is cannon

    def test_cannon_holds_fire_during_deploy_time_and_stays_put(self):
        board = new_board()
        board.elixir[0] = 10.0
        set_hand(board, 0, ["Cannon", "Giant", "Knight", "Musketeer"])
        assert board.play_card(0, "Cannon", ZONE_NAMES.index("pull-L"))
        cannon = board.troops[0]
        place_unit(board, "Giant", 1, cannon.x, cannon.y + 2.0)

        shots = 0
        for _ in range(10):  # first 1.0 s
            board.step(0, action_p1=0)
            shots += _hits_from(board, cannon)
        assert shots == 0
        for _ in range(20):
            board.step(0, action_p1=0)
            shots += _hits_from(board, cannon)
        assert shots > 0
        assert (cannon.x, cannon.y) == (5.5, 9.5)

    def test_cannon_expires_after_30_seconds(self):
        board = new_board()
        board.elixir[0] = 10.0
        set_hand(board, 0, ["Cannon", "Giant", "Knight", "Musketeer"])
        assert board.play_card(0, "Cannon", ZONE_NAMES.index("back-L"))
        _step_seconds(board, 29.5)
        assert [t.name for t in board.troops] == ["Cannon"]
        _step_seconds(board, 1.0)
        assert board.troops == []

    def test_king_dead_ends_match(self):
        board = new_board()
        king = next(t for t in board.towers if t.owner == 1 and t.name == "King_Tower")
        king.hp = 0.0
        king.alive = False
        result = board.step(0, action_p1=0)
        assert result.terminated
        assert board.winner == 0

    def test_time_limit_truncates_and_picks_hp_leader(self):
        board = new_board()
        board.time = MATCH_TIME_LIMIT - TICK_DT
        enemy_tower = next(t for t in board.towers if t.owner == 1 and t.name == "Tower")
        enemy_tower.hp -= 500.0  # P1 has less total HP → P0 wins
        result = board.step(0, action_p1=0)
        assert result.truncated
        assert board.winner == 0


class TestHitEvents:
    @staticmethod
    def _first_hit_by(board: Board, attacker: Troop, max_steps: int = 40):
        for _ in range(max_steps):
            board.step(0, action_p1=0)
            for event in board.hit_events:
                if (event.from_x, event.from_y) == (attacker.x, attacker.y):
                    return event
        pytest.fail(f"no hit recorded for {attacker.name}")

    def test_ranged_hit_recorded_with_origin_and_damage(self):
        board = new_board()
        musketeer = place_unit(board, "Musketeer", 0, 9.0, 8.0)
        giant = place_unit(board, "Giant", 1, 9.0, 13.0)
        event = self._first_hit_by(board, musketeer)
        assert event.ranged and event.owner == 0
        assert giant.distance_to_xy(event.x, event.y) <= giant.speed * TICK_DT + 1e-6
        assert event.damage == cards_dic["Musketeer"]["damage"]

    def test_melee_hit_is_not_ranged(self):
        board = new_board()
        knight = place_unit(board, "Knight", 0, 9.0, 8.0)
        place_unit(board, "Giant", 1, 9.0, 9.0)
        assert not self._first_hit_by(board, knight).ranged

    def test_spell_hits_recorded_and_cleared_next_step(self):
        board = new_board()
        board.elixir[0] = 10.0
        set_hand(board, 0, ["Fireball", "Giant", "Cannon", "Musketeer"])
        place_unit(board, "Giant", 1, 9.0, 20.0)
        board.set_pending_play(0, ("Fireball", 9.0, 20.0))
        board.step(0, action_p1=0)
        spell_hits = [e for e in board.hit_events if e.damage == cards_dic["Fireball"]["damage"]]
        assert [(e.x, e.y) for e in spell_hits] == [(9.0, 20.0)]
        board.step(0, action_p1=0)
        assert all(e.damage != cards_dic["Fireball"]["damage"] for e in board.hit_events)


class TestRewardsAndObservations:
    def test_reward_positive_when_enemy_tower_damaged(self):
        board = new_board()
        tower = next(t for t in board.towers if t.owner == 1 and t.name == "Tower")
        place_unit(board, "Hog_Rider", 0, tower.x, tower.y - 1.0)
        # Hog has deploy_time=1.0 and first_hit_speed=0.6 — first damage after ~1.6 s.
        rewards = [board.step(0, action_p1=0).reward for _ in range(20)]
        assert sum(rewards) > 0.0

    def test_reward_zero_without_combat(self):
        board = new_board()
        result = board.step(0, action_p1=0)
        assert result.reward == 0.0

    def test_observation_shape_and_normalization(self):
        board = new_board()
        obs = board.get_observation(0)
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32
        assert -1.0 <= obs.min() and obs.max() <= 1.0
        assert obs[0] == pytest.approx(board.elixir[0] / MAX_ELIXIR)
        assert obs[2] == pytest.approx(board.time / MATCH_TIME_LIMIT)

    def test_observation_tower_hp_and_hand_onehot(self):
        board = new_board()
        obs = board.get_observation(0)
        np.testing.assert_array_equal(obs[OBS_GLOBAL_FEATURES : OBS_GLOBAL_FEATURES + 6], 1.0)
        for slot, card in enumerate(board.get_hand(0)):
            base = OBS_HAND_BASE + slot * NUM_PLAYABLE_CARDS
            assert obs[base + CARD_TO_ID[card]] == 1.0
            assert obs[base : base + NUM_PLAYABLE_CARDS].sum() == 1.0

    def test_action_mask_legality(self):
        board = new_board()
        set_hand(board, 0, ["Knight", "Giant", "Cannon", "Musketeer"])
        board.elixir[0] = 3.0  # can only afford Knight (3) and Cannon (3)
        mask = board.valid_action_mask(0)
        assert mask.shape == (NUM_ACTIONS,)
        assert mask.dtype == np.bool_
        assert mask[0]
        assert mask[1:9].all() and not mask[9:12].any()  # Knight: 8 troop zones, 3 spell zones off
        assert not mask[12:23].any()  # Giant (5) — too expensive
        assert mask[23:31].all() and not mask[31:34].any()  # Cannon (3)
        assert not mask[34:45].any()  # Musketeer (4) — too expensive

    def test_action_mask_spell_only_on_spell_zones(self):
        board = new_board()
        set_hand(board, 0, ["Fireball", "Giant", "Cannon", "Musketeer"])
        board.elixir[0] = 4.0
        mask = board.valid_action_mask(0)
        assert not mask[1:9].any() and mask[9:12].all()  # Fireball only on the 3 spell zones
        assert mask.sum() == 1 + 3 + 0 + 8 + 8  # noop + Fireball + Giant + Cannon + Musketeer


class TestMirroredObservation:
    def test_player1_view_is_mirrored(self):
        board = new_board()
        place_unit(board, "Knight", 0, 4.0, 10.0)
        obs0 = board.get_observation(0)
        obs1 = board.get_observation(1)

        # elixir swapped
        assert obs1[0] == obs0[1] and obs1[1] == obs0[0]
        assert obs0[2] == obs1[2]  # time is shared

        # towers: own first — P1 block in obs1 == P0 block in obs0
        seg = OBS_GLOBAL_FEATURES
        np.testing.assert_array_equal(obs1[seg : seg + 3], obs0[seg + 3 : seg + 6])
        np.testing.assert_array_equal(obs1[seg + 3 : seg + 6], obs0[seg : seg + 3])

        # unit: owner sign flipped, coordinates rotated by 180°
        base0 = OBS_UNIT_BASE
        assert obs0[base0] == 1.0 and obs1[base0] == -1.0  # own vs enemy
        assert obs0[base0 + 2] == pytest.approx(4.0 / 18.0)
        assert obs0[base0 + 3] == pytest.approx(10.0 / 32.0)
        assert obs1[base0 + 2] == pytest.approx((18.0 - 4.0) / 18.0)
        assert obs1[base0 + 3] == pytest.approx((32.0 - 10.0) / 32.0)


class TestDeterminism:
    def test_same_seed_same_match(self):
        actions = [0, 1, 5, 0, 12, 3, 0, 0, 7, 2] * 5
        results = []
        for _ in range(2):
            board = new_board(seed=7)
            run = []
            for a0, a1 in zip(actions, actions[::-1]):
                r = board.step(a0, action_p1=a1)
                run.append((r.observation.copy(), r.reward, r.terminated, r.truncated))
            results.append(run)
        for (obs1, rew1, t1, tr1), (obs2, rew2, t2, tr2) in zip(*results):
            np.testing.assert_array_equal(obs1, obs2)
            assert (rew1, t1, tr1) == (rew2, t2, tr2)

    def test_different_seeds_differ(self):
        obs_a = new_board(seed=1).get_observation(0)
        obs_b = new_board(seed=2).get_observation(0)
        assert not np.array_equal(obs_a, obs_b)  # different hands after shuffling


class TestGymEnvIntegration:
    def test_gym_api_and_masks(self):
        from cr_rl.env.gym_env import ClashRoyaleEnv

        env = ClashRoyaleEnv(seed=0)
        obs, info = env.reset(seed=0)
        assert env.observation_space.contains(obs)
        mask = env.action_masks()
        assert mask[0]
        action = int(np.flatnonzero(mask)[0])
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
