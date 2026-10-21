"""
Gymnasium wrapper for Board — PPO training against LogicAgent (default).

`opponent`: "logic" | "random" | an object with choose_action(board, player) and reset()
— enables a curriculum (random first) and self-play (an old checkpoint).

The agent is asked only at Decision points (see `cr_rl.game.decisions`); the ticks in
between are simulated inside `step()`.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cr_rl.game.board import MAX_ELIXIR, NUM_ACTIONS, OBS_DIM, Board
from cr_rl.game.decisions import NOOP_HOLD_TICKS, DecisionSchedule
from cr_rl.agents.logic import LogicAgent
from cr_rl.agents.rl import RandomAgent


class ClashRoyaleEnv(gym.Env):
    """Player 0 (RL) vs player 1 (configurable opponent)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        seed: Optional[int] = None,
        opponent: Union[str, Any] = "logic",
        hold_ticks: int = NOOP_HOLD_TICKS,
    ):
        super().__init__()
        self._seed = seed
        self.schedule = DecisionSchedule(hold_ticks)
        self.board = Board(seed=seed)
        if opponent == "logic":
            self.opponent: Any = LogicAgent()
        elif opponent == "random":
            self.opponent = RandomAgent(seed=seed)
        else:
            self.opponent = opponent
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBS_DIM,),
            dtype=np.float32,
        )

    def action_masks(self) -> np.ndarray:
        return self.board.valid_action_mask(player=0)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            episode_seed = seed
        elif self._seed is not None:
            episode_seed = self._seed
        else:
            episode_seed = int(self.np_random.integers(0, 2**31 - 1))

        self.board.reset(seed=episode_seed)
        self.opponent.reset()
        self.board.set_pending_play(0, None)
        self.board.set_pending_play(1, None)
        self.schedule.reset()
        while not self.schedule.is_decision_point(self.board, 0):
            self._tick(0)
        return self.board.get_observation(0), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply `action`, then advance to the next Decision point or the end of the match.

        The reward of every simulated tick is summed into the returned reward.
        """
        self.schedule.record(int(action))
        total_reward = 0.0
        result = self._tick(int(action))
        total_reward += result.reward
        while not (
            result.terminated
            or result.truncated
            or self.schedule.is_decision_point(self.board, 0)
        ):
            result = self._tick(0)
            total_reward += result.reward
        info = dict(result.info)
        info["winner"] = self.board.winner
        return result.observation, float(total_reward), result.terminated, result.truncated, info

    def _tick(self, action: int):
        self.board.set_pending_play(0, None)
        # LogicAgent plays through set_pending_play and returns 0; RandomAgent/RLAgent
        # return a discrete action that is passed on to the Board. The opponent acts every tick.
        opponent_action = self.opponent.choose_action(self.board, player=1)
        result = self.board.step(action, action_p1=int(opponent_action or 0))
        self.schedule.advance()
        return result

    def render(self) -> None:
        self.board.render()


def make_env(seed: Optional[int] = None) -> ClashRoyaleEnv:
    return ClashRoyaleEnv(seed=seed)
