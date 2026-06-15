"""
Wrapper Gymnasium dla Board — trening PPO przeciwko LogicAgent.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from board import MAX_ELIXIR, NUM_ACTIONS, OBS_DIM, Board
from logic_agent import LogicAgent


class ClashRoyaleEnv(gym.Env):
    """Gracz 0 (RL) vs gracz 1 (LogicAgent)."""

    metadata = {"render_modes": []}

    def __init__(self, *, seed: Optional[int] = None):
        super().__init__()
        self._seed = seed
        self.board = Board(seed=seed)
        self.opponent = LogicAgent()
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
        obs = self.board.get_observation(0)
        return obs, {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.board.set_pending_play(0, None)
        self.opponent.choose_action(self.board, player=1)
        result = self.board.step(int(action), action_p1=0)
        obs = result.observation
        terminated = result.terminated
        truncated = result.truncated
        info = dict(result.info)
        info["winner"] = self.board.winner
        return obs, float(result.reward), terminated, truncated, info

    def render(self) -> None:
        self.board.render()


def make_env(seed: Optional[int] = None) -> ClashRoyaleEnv:
    return ClashRoyaleEnv(seed=seed)
