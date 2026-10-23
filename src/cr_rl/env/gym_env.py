"""
Gymnasium wrapper for Board — PPO training against LogicAgent (default).

`opponent`: "logic" | "logic-eps" | "random" | a frozen MaskablePPO policy | an object with
choose_action(board, player) and reset(). Policies play as player 1 through the fixed P1 view
at the same Decision-point schedule they were trained with.

The agent is asked only at Decision points (see `cr_rl.game.decisions`); the ticks in
between are simulated inside `step()`. The reward is computed once per decision step
(see `cr_rl.game.rewards`).
"""

from __future__ import annotations

from typing import Any, Optional, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cr_rl.game.board import MAX_ELIXIR, NUM_ACTIONS, OBS_DIM, Board
from cr_rl.game.decisions import NOOP_HOLD_TICKS, DecisionSchedule, ScheduledAgent
from cr_rl.game.rewards import DEFAULT_GAMMA, RewardWeights, potential, step_reward
from cr_rl.agents.logic import EpsilonLogicAgent, LogicAgent
from cr_rl.agents.rl import RandomAgent, RLAgent


def derive_episode_seed(run_seed: int, env_index: int, episode: int) -> int:
    """Seed of one training episode, derived from the run seed, the env index and the episode counter."""
    return int(np.random.SeedSequence([run_seed, env_index, episode]).generate_state(1)[0] >> 1)


class ClashRoyaleEnv(gym.Env):
    """Player 0 (RL) vs player 1 (configurable opponent).

    A seeded env gives every episode its own seed (see `derive_episode_seed`); `reset(seed=...)`
    overrides it, e.g. for evaluation on fixed seeds.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        seed: Optional[int] = None,
        env_index: int = 0,
        opponent: Union[str, Any] = "logic",
        hold_ticks: int = NOOP_HOLD_TICKS,
        reward_weights: RewardWeights = RewardWeights(),
        gamma: float = DEFAULT_GAMMA,
    ):
        super().__init__()
        self.reward_weights = reward_weights
        self.gamma = gamma
        self._potential = 0.0
        self._seed = seed
        self._env_index = env_index
        self._episodes = 0
        self.schedule = DecisionSchedule(hold_ticks)
        self.board = Board(seed=seed)
        self.opponent = _build_opponent(opponent, seed, hold_ticks)
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
            episode_seed = derive_episode_seed(self._seed, self._env_index, self._episodes)
        else:
            episode_seed = int(self.np_random.integers(0, 2**31 - 1))
        self._episodes += 1

        self.board.reset(seed=episode_seed)
        self.opponent.reset()
        if isinstance(self.opponent, RandomAgent):
            self.opponent.reset(seed=episode_seed)
        self.board.set_pending_play(0, None)
        self.board.set_pending_play(1, None)
        self.schedule.reset()
        while not self.schedule.is_decision_point(self.board, 0):
            self._tick(0)
        self._potential = potential(self.board, 0, self.reward_weights)
        return self.board.get_observation(0), {"seed": episode_seed}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply `action`, then advance to the next Decision point or the end of the match.

        The reward is the outcome plus the shaping over the whole step, not summed per tick.
        """
        self.schedule.record(int(action))
        result = self._tick(int(action))
        while not (result.terminated or self.schedule.is_decision_point(self.board, 0)):
            result = self._tick(0)
        reward = step_reward(self.board, 0, self.reward_weights, self.gamma, self._potential)
        self._potential = potential(self.board, 0, self.reward_weights)
        info = dict(result.info)
        info["winner"] = self.board.winner
        return result.observation, float(reward), result.terminated, result.truncated, info

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


def _build_opponent(spec: Union[str, Any], seed: Optional[int], hold_ticks: int) -> Any:
    if spec == "logic":
        return LogicAgent()
    if spec == "logic-eps":
        return EpsilonLogicAgent()
    if spec == "random":
        return RandomAgent(seed=seed)
    if isinstance(spec, str):
        raise ValueError(f"Unknown opponent {spec!r}")
    if hasattr(spec, "predict"):  # a frozen policy, asked at Decision points like in training
        return ScheduledAgent(RLAgent(model=spec), hold_ticks)
    return spec


def make_env(seed: Optional[int] = None) -> ClashRoyaleEnv:
    return ClashRoyaleEnv(seed=seed)
