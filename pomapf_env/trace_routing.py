"""Stored Direct tie rankings, shared by ARPE rollout and inference.

These are routing metadata, not neural features. Saving them in observations
ensures PPO recomputes the same route instead of drawing new randomness.
"""

import gymnasium as gym
import numpy as np


TIE_KEY = "bonus_tie_ranks"


def bonus_rng(seed):
    return np.random.default_rng(np.random.SeedSequence([int(seed or 0), 130913, 1]))


def draw_tie_ranks(rng, count):
    ranks = []
    # Keep two separate draws: this is also Direct's random-number order.
    for _ in range(2):
        values = rng.random((count, 5))
        ranks.append(np.argsort(np.argsort(values, axis=1, kind="stable"),
                                axis=1, kind="stable").astype(np.float32))
    return np.stack(ranks, axis=1)


class BonusRoutingObservation(gym.Wrapper):
    """Attach reproducible tie ranks once per environment observation."""

    def __init__(self, env, routing_seed=0):
        super().__init__(env)
        spaces = dict(env.observation_space.spaces)
        spaces[TIE_KEY] = gym.spaces.Box(0, 4, shape=(2, 5), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(spaces)
        self.num_agents = env.num_agents
        self.is_multiagent = env.is_multiagent
        self.routing_seed = int(routing_seed or 0)
        self._routing_grid = None
        self._episode_index = -1
        self._bonus_rng = None

    def _augment(self, observations):
        grid = self.env.unwrapped.grid
        if grid is not self._routing_grid:
            self._routing_grid = grid
            self._episode_index += 1
            self._bonus_rng = bonus_rng(self.routing_seed + 65537 * self._episode_index)
        ranks = draw_tie_ranks(self._bonus_rng, len(observations))
        for obs, rank in zip(observations, ranks):
            obs[TIE_KEY] = rank
        return observations

    def reset(self, **kwargs):
        observations, info = self.env.reset(**kwargs)
        return self._augment(observations), info

    def step(self, actions):
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        return self._augment(observations), rewards, terminated, truncated, infos
