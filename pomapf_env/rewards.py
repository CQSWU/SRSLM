import gymnasium as gym
import numpy as np


def _current_grid(env):
    current = env
    while current is not None:
        grid = getattr(current, "grid", None)
        if grid is not None:
            return grid
        current = getattr(current, "env", None)
    raise RuntimeError("Cannot identify the current POGEMA grid")


class FailedMoveCredit(gym.Wrapper):
    extra_failed_nonstay_penalty = 0.0098

    def __init__(self, env):
        super().__init__(env)
        self.num_agents = env.num_agents
        self.is_multiagent = True

    def step(self, actions):
        grid = _current_grid(self.env)
        before = np.asarray(grid.positions_xy, dtype=np.int64).copy()
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        next_grid = _current_grid(self.env)

        if next_grid is grid:
            after = np.asarray(next_grid.positions_xy, dtype=np.int64)
            attempted = np.asarray(actions, dtype=np.int64) != 0
            failed = attempted & np.all(after == before, axis=1)
            shaped = np.asarray(rewards, dtype=np.float32)
            shaped -= failed.astype(np.float32) * self.extra_failed_nonstay_penalty
            rewards = shaped.tolist()
        return observations, rewards, terminated, truncated, infos
