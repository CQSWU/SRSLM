"""Check window recording against real POGEMA, not paper-model performance."""
import inspect
from pathlib import Path

import numpy as np
import pytest

import run_experiments as runner


@pytest.mark.parametrize("collision_system", ["block_both", "soft"])
def test_real_lifelong_environment_records_goals_and_short_tail(collision_system):
    public_root = Path(__file__).resolve().parents[1]
    assert Path(runner.__file__).resolve() == public_root / "run_experiments.py"
    assert Path(inspect.getfile(runner.make_pomapf)).resolve().is_relative_to(public_root)

    class TestPolicy:
        """A small goal-seeking test driver; not any reported MAPF method."""

        def __init__(self):
            self.resets = 0
            self.events = []

        def after_reset(self):
            self.resets += 1

        def set_env(self, env):
            self.env = env

        def act(self, observations, rewards, dones, infos):
            grid = self.env.unwrapped.grid
            positions = np.asarray(grid.positions_xy)
            targets = np.asarray(grid.finishes_xy)
            moves = np.asarray(self.env.grid_config.MOVES)
            distances = np.abs(positions[:, None, :] + moves[None, :, :] - targets[:, None, :]).sum(axis=-1)
            return distances.argmin(axis=1).tolist()

        def after_step(self, dones):
            self.events.append(int(sum(self.env.unwrapped.was_on_goal)))

    policy = TestPolicy()
    result = runner.run_algorithm(
        policy, map_name="window_recording_test", max_episode_steps=600,
        seed=42, num_agents=4, obs_radius=5, animate=False,
        on_target="restart", collision_system=collision_system,
        map_text="\n".join(["." * 12] * 12),
    )
    assert policy.resets == 1
    assert len(policy.events) == result["policy_decision_calls"] == 600
    assert result["environment_step_count_observed"] == 600
    assert sum(policy.events) == result["completed_targets_observed"] > 0
    assert result["avg_throughput"] == pytest.approx(sum(policy.events) / 600)
    assert result["policy_decision_seconds"] > 0
    assert result["policy_decision_ms_per_joint_action"] == pytest.approx(
        result["policy_decision_seconds"] * 1000 / 600
    )
    segments = result["throughput_segments"]
    assert [segment["step_count"] for segment in segments] == [512, 88]
    for segment in segments:
        start, end = segment["start_step"] - 1, segment["end_step"]
        assert segment["completed_targets"] == sum(policy.events[start:end])
        assert segment["throughput"] == pytest.approx(sum(policy.events[start:end]) / (end - start))
        assert segment["cumulative_throughput"] == pytest.approx(sum(policy.events[:end]) / end)
