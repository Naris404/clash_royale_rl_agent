"""Decision points: the env asks the agent only when a card play is legal and no noop hold runs."""

from __future__ import annotations

from cr_rl.env.gym_env import ClashRoyaleEnv
from cr_rl.game.board import TICK_DT


def _ticks_between(env: ClashRoyaleEnv, action: int) -> int:
    before = env.board.time
    env.step(action)
    return round((env.board.time - before) / TICK_DT)


class TestNoopHold:
    def test_noop_holds_three_ticks_by_default(self):
        env = ClashRoyaleEnv(seed=0, opponent="random")
        env.reset(seed=0)
        assert _ticks_between(env, 0) == 3

    def test_noop_only_state_is_never_returned(self):
        env = ClashRoyaleEnv(seed=3, opponent="logic")
        env.reset(seed=3)
        done = False
        while not done:
            mask = env.action_masks()
            assert mask[1:].any()
            _, _, terminated, truncated, _ = env.step(0)  # always wait: elixir hits the cap
            done = terminated or truncated

    def test_reset_clears_a_hold_left_over_from_the_previous_match(self):
        used, fresh = ClashRoyaleEnv(seed=0), ClashRoyaleEnv(seed=0)
        used.reset(seed=0)
        done = False
        while not done:
            _, _, terminated, truncated, _ = used.step(0)
            done = terminated or truncated
        used.reset(seed=0)
        fresh.reset(seed=0)
        play = int(fresh.action_masks()[1:].argmax()) + 1
        used.step(play)
        fresh.step(play)
        assert used.board.time == fresh.board.time

    def test_hold_is_configurable_and_one_gives_per_tick_queries(self):
        env = ClashRoyaleEnv(seed=0, opponent="random", hold_ticks=1)
        env.reset(seed=0)
        assert _ticks_between(env, 0) == 1
        env = ClashRoyaleEnv(seed=0, opponent="random", hold_ticks=5)
        env.reset(seed=0)
        assert _ticks_between(env, 0) == 5
