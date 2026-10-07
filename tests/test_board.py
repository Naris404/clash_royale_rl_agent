"""Testy środowiska Board — mechanika gry, obserwacje, nagrody, determinizm."""

from __future__ import annotations

from collections import deque

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
    TOWER_LAYOUT,
    Board,
)
from cr_rl.game.cards import CARD_TO_ID, Troop, cards_dic


def _board(seed: int = 42) -> Board:
    env = Board(seed=seed)
    env.reset(seed=seed)
    return env


def _force_hand(board: Board, player: int, hand: list[str], queue: list[str]) -> None:
    board.hand[player] = list(hand)
    board.hand_queue[player] = deque(queue)


def _add_troop(board: Board, name: str, owner: int, x: float, y: float) -> Troop:
    troop = Troop(name, owner)
    troop.place((x, y))
    board.troops.append(troop)
    return troop


class TestHands:
    def test_initial_hand_4_unique_cards_from_6(self):
        board = _board()
        for player in (0, 1):
            hand = board.get_hand(player)
            assert len(hand) == HAND_SIZE
            assert len(set(hand)) == HAND_SIZE
            remaining = list(board.hand_queue[player])
            assert len(remaining) == 2
            assert set(hand) | set(remaining) == set(board._full_deck(player))

    def test_cycle_after_play_puts_card_at_queue_back(self):
        board = _board()
        board.elixir[0] = 10.0
        _force_hand(board, 0, ["Knight", "Giant", "Cannon", "Musketeer"], ["Hog_Rider", "Fireball"])
        hand = board.get_hand(0)
        played = hand[0]
        expected_next = board.hand_queue[0][0]
        assert board.play_card(0, played, 0)
        new_hand = board.get_hand(0)
        assert played not in new_hand
        assert new_hand[-1] == expected_next
        assert board.hand_queue[0][-1] == played
        assert len(new_hand) == HAND_SIZE


class TestElixir:
    def test_regen_per_tick(self):
        board = _board()
        before = board.elixir[0]
        board.step(0, action_p1=0)
        assert board.elixir[0] == pytest.approx(before + ELIXIR_PER_SECOND * TICK_DT)

    def test_capped_at_max(self):
        board = _board()
        board.elixir[0] = MAX_ELIXIR
        board.step(0, action_p1=0)
        assert board.elixir[0] == MAX_ELIXIR

    def test_play_deducts_cost(self):
        board = _board()
        board.elixir[0] = 10.0
        _force_hand(board, 0, ["Knight", "Giant", "Cannon", "Musketeer"], ["Hog_Rider", "Fireball"])
        assert board.play_card(0, "Knight", 0)
        assert board.elixir[0] == pytest.approx(10.0 - cards_dic["Knight"]["elisir"])


class TestPlayLegality:
    def test_rejects_card_not_in_hand(self):
        board = _board()
        board.elixir[0] = 10.0
        _force_hand(board, 0, ["Knight", "Giant", "Cannon", "Musketeer"], ["Hog_Rider", "Fireball"])
        assert not board.play_card(0, "Fireball", 0)

    def test_rejects_when_not_enough_elixir(self):
        board = _board()
        board.elixir[0] = 2.0
        _force_hand(board, 0, ["Knight", "Giant", "Cannon", "Musketeer"], ["Hog_Rider", "Fireball"])
        assert not board.play_card(0, "Giant", 0)  # koszt 5

    def test_rejects_position_outside_deploy_zone(self):
        board = _board()
        board.elixir[0] = 10.0
        _force_hand(board, 0, ["Knight", "Giant", "Cannon", "Musketeer"], ["Hog_Rider", "Fireball"])
        assert not board.play_card_at(0, "Knight", 9.0, RIVER_Y + 5.0)  # strona wroga
        assert not board.play_card_at(0, "Knight", 0.0, 5.0)  # poza areną

    def test_action_to_card_zone_mapping(self):
        board = _board()
        _force_hand(board, 0, ["Knight", "Giant", "Cannon", "Musketeer"], ["Hog_Rider", "Fireball"])
        assert board.action_to_card_zone(0, 0) == (None, None)
        assert board.action_to_card_zone(1, 0) == ("Knight", 0)
        assert board.action_to_card_zone(2, 0) == ("Knight", 1)
        assert board.action_to_card_zone(12, 0) == ("Giant", 0)
        assert board.action_to_card_zone(44, 0) == ("Musketeer", 10)
        assert board.action_to_card_zone(45, 0) == (None, None)  # poza zakresem


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
        board = _board()
        for z in DEPLOY_ZONES[player]:
            if z.name in TROOP_ZONES_P0:
                assert board._in_deploy_zone(player, z.x, z.y), z.name

    def test_play_card_uses_absolute_zone_coordinates(self):
        board = _board()
        board.elixir[0] = 10.0
        _force_hand(board, 0, ["Cannon", "Giant", "Knight", "Musketeer"], ["Hog_Rider", "Fireball"])
        pull_r = next(i for i, z in enumerate(DEPLOY_ZONES[0]) if z.name == "pull-R")
        assert board.play_card(0, "Cannon", pull_r)
        assert (board.troops[0].x, board.troops[0].y) == (12.5, 9.5)

    def test_spell_cast_at_spell_zone_and_rejected_on_troop_zone(self):
        board = _board()
        board.elixir[0] = 10.0
        _force_hand(board, 0, ["Fireball", "Giant", "Cannon", "Musketeer"], ["Hog_Rider", "Knight"])
        names = [z.name for z in DEPLOY_ZONES[0]]
        assert not board.play_card(0, "Fireball", names.index("mid-L"))
        assert board.play_card(0, "Fireball", names.index("spell-tower-L"))
        effect = board.spell_effects[0]
        assert (effect.x, effect.y) == (3.5, 24.0)

    def test_troop_rejected_on_spell_zone(self):
        board = _board()
        board.elixir[0] = 10.0
        _force_hand(board, 0, ["Knight", "Giant", "Cannon", "Musketeer"], ["Hog_Rider", "Fireball"])
        assert not board.play_card(0, "Knight", [z.name for z in DEPLOY_ZONES[0]].index("spell-king"))


class TestSpells:
    def test_fireball_damages_enemies_in_radius(self):
        board = _board()
        board.elixir[0] = 10.0
        _force_hand(board, 0, ["Fireball", "Giant", "Cannon", "Musketeer"], ["Hog_Rider", "Knight"])
        enemy = _add_troop(board, "Giant", 1, 9.0, 20.0)
        ally = _add_troop(board, "Knight", 0, 9.0, 20.0)
        far_enemy = _add_troop(board, "Knight", 1, 2.0, 25.0)

        assert board.play_card_at(0, "Fireball", 9.0, 20.0)
        assert enemy.hp == cards_dic["Giant"]["hp"] - cards_dic["Fireball"]["damage"]
        assert ally.hp == ally.max_hp  # bez friendly fire
        assert far_enemy.hp == far_enemy.max_hp  # poza promieniem

    def test_fireball_kills_low_hp_unit(self):
        board = _board()
        board.elixir[0] = 10.0
        _force_hand(board, 0, ["Fireball", "Giant", "Cannon", "Musketeer"], ["Hog_Rider", "Knight"])
        enemy = _add_troop(board, "Knight", 1, 9.0, 20.0)
        enemy.hp = 100.0
        board.play_card_at(0, "Fireball", 9.0, 20.0)
        assert not enemy.alive

    def test_fireball_cycles_hand_and_costs_elixir(self):
        board = _board()
        board.elixir[0] = 10.0
        _force_hand(board, 0, ["Fireball", "Giant", "Cannon", "Musketeer"], ["Hog_Rider", "Knight"])
        assert board.play_card_at(0, "Fireball", 9.0, 20.0)
        assert board.elixir[0] == pytest.approx(10.0 - cards_dic["Fireball"]["elisir"])
        assert "Fireball" not in board.get_hand(0)


class TestRiverAndBridges:
    def test_off_bridge_unit_pushed_back_to_own_side(self):
        board = _board()
        knight = _add_troop(board, "Knight", 0, 9.0, RIVER_Y)  # środek rzeki, poza mostem
        board._enforce_no_river_cut(knight)
        assert knight.x == pytest.approx(BRIDGE_LANE_X[1])  # bliższy most (x=14)
        assert knight.y < RIVER_Y - RIVER_HALF_WIDTH  # cofnięty na swoją stronę

    def test_hog_not_pushed_back(self):
        board = _board()
        hog = _add_troop(board, "Hog_Rider", 0, 9.0, RIVER_Y)
        board._enforce_no_river_cut(hog)
        assert (hog.x, hog.y) == (9.0, RIVER_Y)  # Hog skacze przez rzekę

    def test_ground_unit_crosses_river_via_bridge_lane(self):
        board = _board()
        _add_troop(board, "Knight", 0, 9.0, 12.0)
        crossed = False
        for _ in range(3000):
            board.step(0, action_p1=0)
            knight = board.troops[0] if board.troops else None
            if knight is None:
                break
            if (RIVER_Y - RIVER_HALF_WIDTH) < knight.y < (RIVER_Y + RIVER_HALF_WIDTH):
                assert any(abs(knight.x - bx) <= 1.8 for bx in BRIDGE_LANE_X)
            if knight.y > RIVER_Y + RIVER_HALF_WIDTH:
                crossed = True
                break
        assert crossed

    @staticmethod
    def _walk_across(board: Board, troop: Troop, max_steps: int = 600) -> list[tuple[float, float]]:
        """Kroki symulacji aż jednostka przejdzie przez rzekę; zwraca trasę."""
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
        board = _board()
        knight = _add_troop(board, "Knight", 0, 8.0, 6.0)
        path = self._walk_across(board, knight)

        length = sum(np.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(path, path[1:]))
        bx = BRIDGE_LANE_X[0]
        l_path = abs(8.0 - bx) + (path[-1][1] - 6.0)
        assert length < l_path - 2.0

        (x0, y0), (x1, y1) = path[0], path[1]
        assert x1 < x0 and y1 > y0  # od pierwszego kroku w skos, nie najpierw w bok

    @pytest.mark.parametrize("owner", [0, 1])
    @pytest.mark.parametrize("start", [(8.0, 6.0), (1.5, 10.0), (16.5, 12.0), (9.0, 14.0), (5.0, 13.0)])
    def test_river_crossed_only_on_bridge_without_teleports(self, owner, start):
        board = _board()
        x, y = start if owner == 0 else (ARENA_WIDTH - start[0], 32.0 - start[1])
        knight = _add_troop(board, "Knight", owner, x, y)
        path = self._walk_across(board, knight)

        max_step = knight.speed * TICK_DT + 1e-6
        for (x0, y0), (x1, y1) in zip(path, path[1:]):
            assert np.hypot(x1 - x0, y1 - y0) <= max_step  # bez „cofania” przez rzekę
            if (RIVER_Y - RIVER_HALF_WIDTH) < y1 < (RIVER_Y + RIVER_HALF_WIDTH):
                assert any(abs(x1 - bx) <= BRIDGE_HALF_WIDTH for bx in BRIDGE_LANE_X)

    @pytest.mark.parametrize("lane", ["L", "R"])
    def test_cannon_in_pull_zone_draws_hog_off_tower_lane(self, lane):
        names = [z.name for z in DEPLOY_ZONES[0]]

        def run(with_cannon: bool) -> tuple[Board, Troop]:
            board = _board()
            board.elixir = [10.0, 10.0]
            _force_hand(board, 0, ["Cannon", "Knight", "Giant", "Musketeer"], ["Fireball", "Hog_Rider"])
            _force_hand(board, 1, ["Hog_Rider", "Knight", "Giant", "Musketeer"], ["Fireball", "Cannon"])
            if with_cannon:
                assert board.play_card(0, "Cannon", names.index(f"pull-{lane}"))
            assert board.play_card(1, "Hog_Rider", names.index(f"bridge-{lane}"))
            hog = board.troops[-1]
            for _ in range(80):
                board.step(0, action_p1=0)
            return board, hog

        princess = lambda board: next(  # noqa: E731
            t for t in board.towers
            if t.owner == 0 and t.name == "Tower" and (t.x < ARENA_WIDTH / 2) == (lane == "L")
        )

        without, _ = run(with_cannon=False)
        assert princess(without).hp < princess(without).max_hp  # bez Cannona Hog bije wieżę

        board, hog = run(with_cannon=True)
        cannon = next(t for t in board.troops if t.name == "Cannon")
        assert princess(board).hp == princess(board).max_hp  # wieża nietknięta
        assert cannon.hp < cannon.max_hp  # Hog zajęty Cannonem

    def test_hog_rider_jumps_river_in_straight_line(self):
        board = _board()
        hog = _add_troop(board, "Hog_Rider", 0, 9.0, 12.0)
        path = self._walk_across(board, hog)
        assert any(
            (RIVER_Y - RIVER_HALF_WIDTH) < y < (RIVER_Y + RIVER_HALF_WIDTH) and abs(x - 9.0) < 1.0
            for x, y in path
        )


class TestTargetingAndCombat:
    def test_sight_range_limits_aggro(self):
        board = _board()
        knight = _add_troop(board, "Knight", 0, 1.0, 4.0)
        far = _add_troop(board, "Knight", 1, 17.0, 28.0)  # daleko poza sight range
        assert board.find_nearest_target(knight) is None
        near = _add_troop(board, "Knight", 1, knight.x + 3.0, knight.y)
        assert board.find_nearest_target(knight) is near
        far.alive = False

    def test_tower_shoots_only_in_range(self):
        board = _board()
        tower = next(
            t for t in board.towers
            if t.owner == 0 and t.name == "Tower" and t.x < ARENA_WIDTH / 2
        )
        out_of_range = _add_troop(board, "Knight", 1, tower.x, tower.y + 12.0)
        assert board.find_tower_target(tower) is None
        out_of_range.place((tower.x, tower.y + 5.0))
        assert board.find_tower_target(tower) is out_of_range

    def test_king_dead_ends_match(self):
        board = _board()
        king = next(t for t in board.towers if t.owner == 1 and t.name == "King_Tower")
        king.hp = 0.0
        king.alive = False
        result = board.step(0, action_p1=0)
        assert result.terminated
        assert board.winner == 0

    def test_time_limit_truncates_and_picks_hp_leader(self):
        board = _board()
        board.time = MATCH_TIME_LIMIT - TICK_DT
        enemy_tower = next(t for t in board.towers if t.owner == 1 and t.name == "Tower")
        enemy_tower.hp -= 500.0  # P1 ma mniej łącznego HP → wygrywa P0
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
        board = _board()
        musketeer = _add_troop(board, "Musketeer", 0, 9.0, 8.0)
        giant = _add_troop(board, "Giant", 1, 9.0, 13.0)
        event = self._first_hit_by(board, musketeer)
        assert event.ranged and event.owner == 0
        assert giant.distance_to_xy(event.x, event.y) <= giant.speed * TICK_DT + 1e-6
        assert event.damage == cards_dic["Musketeer"]["damage"]

    def test_melee_hit_is_not_ranged(self):
        board = _board()
        knight = _add_troop(board, "Knight", 0, 9.0, 8.0)
        _add_troop(board, "Giant", 1, 9.0, 9.0)
        assert not self._first_hit_by(board, knight).ranged

    def test_spell_hits_recorded_and_cleared_next_step(self):
        board = _board()
        board.elixir[0] = 10.0
        _force_hand(board, 0, ["Fireball", "Giant", "Cannon", "Musketeer"], ["Hog_Rider", "Knight"])
        _add_troop(board, "Giant", 1, 9.0, 20.0)
        board.set_pending_play(0, ("Fireball", 9.0, 20.0))
        board.step(0, action_p1=0)
        spell_hits = [e for e in board.hit_events if e.damage == cards_dic["Fireball"]["damage"]]
        assert [(e.x, e.y) for e in spell_hits] == [(9.0, 20.0)]
        board.step(0, action_p1=0)
        assert all(e.damage != cards_dic["Fireball"]["damage"] for e in board.hit_events)


class TestRewardsAndObservations:
    def test_reward_positive_when_enemy_tower_damaged(self):
        board = _board()
        tower = next(t for t in board.towers if t.owner == 1 and t.name == "Tower")
        _add_troop(board, "Hog_Rider", 0, tower.x, tower.y - 1.0)
        # Hog ma deploy_time=1.0 i first_hit_speed=0.6 — pierwsze obrażenia po ~1.6 s.
        rewards = [board.step(0, action_p1=0).reward for _ in range(20)]
        assert sum(rewards) > 0.0

    def test_reward_zero_without_combat(self):
        board = _board()
        result = board.step(0, action_p1=0)
        assert result.reward == 0.0

    def test_observation_shape_and_normalization(self):
        board = _board()
        obs = board.get_observation(0)
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32
        assert -1.0 <= obs.min() and obs.max() <= 1.0
        assert obs[0] == pytest.approx(board.elixir[0] / MAX_ELIXIR)
        assert obs[2] == pytest.approx(board.time / MATCH_TIME_LIMIT)

    def test_observation_tower_hp_and_hand_onehot(self):
        board = _board()
        obs = board.get_observation(0)
        towers = board.towers
        for i, tower in enumerate(towers):
            assert obs[OBS_GLOBAL_FEATURES + i] == pytest.approx(tower.hp / tower.max_hp)
        hand = board.get_hand(0)
        for slot, card in enumerate(hand):
            base = OBS_HAND_BASE + slot * NUM_PLAYABLE_CARDS
            assert obs[base + CARD_TO_ID[card]] == 1.0
            assert obs[base : base + NUM_PLAYABLE_CARDS].sum() == 1.0

    def test_action_mask_legality(self):
        board = _board()
        _force_hand(board, 0, ["Knight", "Giant", "Cannon", "Musketeer"], ["Hog_Rider", "Fireball"])
        board.elixir[0] = 3.0  # stać tylko na Knighta (3) i Cannona (3)
        mask = board.valid_action_mask(0)
        assert mask.shape == (NUM_ACTIONS,)
        assert mask.dtype == np.bool_
        assert mask[0]
        assert mask[1:9].all() and not mask[9:12].any()  # Knight: 8 stref wojsk, 3 czarowe off
        assert not mask[12:23].any()  # Giant (5) — za drogi
        assert mask[23:31].all() and not mask[31:34].any()  # Cannon (3)
        assert not mask[34:45].any()  # Musketeer (4) — za drogi

    def test_action_mask_spell_only_on_spell_zones(self):
        board = _board()
        _force_hand(board, 0, ["Fireball", "Giant", "Cannon", "Musketeer"], ["Hog_Rider", "Knight"])
        board.elixir[0] = 4.0
        mask = board.valid_action_mask(0)
        assert not mask[1:9].any() and mask[9:12].all()  # Fireball tylko na 3 strefach czarów
        assert mask.sum() == 1 + 3 + 0 + 8 + 8  # noop + Fireball + Giant + Cannon + Musketeer


class TestMirroredObservation:
    def test_player1_view_is_mirrored(self):
        board = _board()
        _add_troop(board, "Knight", 0, 4.0, 10.0)
        obs0 = board.get_observation(0)
        obs1 = board.get_observation(1)

        # eliksir zamieniony miejscami
        assert obs1[0] == obs0[1] and obs1[1] == obs0[0]
        assert obs0[2] == obs1[2]  # czas wspólny

        # wieże: własne najpierw — blok P1 w obs1 == blok P0 w obs0
        seg = OBS_GLOBAL_FEATURES
        np.testing.assert_array_equal(obs1[seg : seg + 3], obs0[seg + 3 : seg + 6])
        np.testing.assert_array_equal(obs1[seg + 3 : seg + 6], obs0[seg : seg + 3])

        # jednostka: znak właściciela odwrócony, współrzędne obrócone o 180°
        base0 = OBS_UNIT_BASE
        assert obs0[base0] == 1.0 and obs1[base0] == -1.0  # własna vs wroga
        assert obs0[base0 + 2] == pytest.approx(4.0 / 18.0)
        assert obs0[base0 + 3] == pytest.approx(10.0 / 32.0)
        assert obs1[base0 + 2] == pytest.approx((18.0 - 4.0) / 18.0)
        assert obs1[base0 + 3] == pytest.approx((32.0 - 10.0) / 32.0)


class TestDeterminism:
    def test_same_seed_same_match(self):
        actions = [0, 1, 5, 0, 12, 3, 0, 0, 7, 2] * 5
        results = []
        for _ in range(2):
            board = _board(seed=7)
            run = []
            for a0, a1 in zip(actions, actions[::-1]):
                r = board.step(a0, action_p1=a1)
                run.append((r.observation.copy(), r.reward, r.terminated, r.truncated))
            results.append(run)
        for (obs1, rew1, t1, tr1), (obs2, rew2, t2, tr2) in zip(*results):
            np.testing.assert_array_equal(obs1, obs2)
            assert (rew1, t1, tr1) == (rew2, t2, tr2)

    def test_different_seeds_differ(self):
        obs_a = _board(seed=1).get_observation(0)
        obs_b = _board(seed=2).get_observation(0)
        assert not np.array_equal(obs_a, obs_b)  # różne ręce po tasowaniu


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
