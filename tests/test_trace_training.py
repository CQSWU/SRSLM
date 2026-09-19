"""Regression tests for the selected ARPE checkpoint's training reward."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import yaml

from pathlib import Path

from pomapf_env.trace_training import FailedMoveCredit


class _StepEnv(gym.Env):
    def __init__(self, *, auto_reset=False):
        self.num_agents = 3
        self.action_space = gym.spaces.Discrete(5)
        self.observation_space = gym.spaces.Box(0, 1, shape=(1,), dtype=np.float32)
        self.grid = SimpleNamespace(positions_xy=np.array([[1, 1], [2, 2], [3, 3]]))
        self.auto_reset = auto_reset
        self.observations = [{"agent": i} for i in range(self.num_agents)]
        self.infos = [{"is_active": True} for _ in range(self.num_agents)]
        self.last_actions = None

    def step(self, actions):
        self.last_actions = actions
        if self.auto_reset:
            self.grid = SimpleNamespace(positions_xy=self.grid.positions_xy.copy())
        else:
            self.grid.positions_xy[2, 0] += 1
        return (
            self.observations,
            [-0.0003, -0.0001, 0.9999],
            [False] * 3,
            [self.auto_reset] * 3,
            self.infos,
        )


def test_failed_move_credit_only_penalizes_attempted_movements_that_did_not_move():
    inner = _StepEnv()
    env = FailedMoveCredit(inner)
    actions = [4, 0, 2]
    observations, rewards, terminated, truncated, infos = env.step(actions)
    np.testing.assert_allclose(rewards, [-0.0101, -0.0001, 0.9999], atol=1e-7)
    assert observations is inner.observations
    assert infos is inner.infos
    assert inner.last_actions is actions
    assert terminated == [False] * 3
    assert truncated == [False] * 3
    assert env.num_agents == 3
    assert env.is_multiagent is True


def test_failed_move_credit_does_not_compare_positions_across_autoreset():
    inner = _StepEnv(auto_reset=True)
    env = FailedMoveCredit(inner)
    old_grid = inner.grid
    _, rewards, _, truncated, _ = env.step([4, 0, 2])
    assert old_grid is not inner.grid
    assert rewards == [-0.0003, -0.0001, 0.9999]
    assert truncated == [True] * 3


def test_failed_move_credit_finds_grid_through_wrapper_stack():
    env = FailedMoveCredit(_StepEnv())
    env.env = gym.Wrapper(env.env)
    _, rewards, _, _, _ = env.step([4, 0, 2])
    np.testing.assert_allclose(rewards, [-0.0101, -0.0001, 0.9999], atol=1e-7)


def test_both_arpe_training_recipes_keep_the_original_learning_rate():
    root = Path(__file__).resolve().parents[1]
    for name in ("train_arpe.yaml", "train_arpe_zero_trace.yaml"):
        recipe = yaml.safe_load((root / "learning" / name).read_text())
        assert recipe["experiment_settings"]["learning_rate"] == 0.00005
