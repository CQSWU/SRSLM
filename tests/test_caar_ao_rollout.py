import ast
import hashlib
import inspect
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import policy_estimation.caar_ao_rollout as rollout_module
from policy_estimation.caar_ao_rollout import (
    AO_SAFE_LANE,
    CAAR_LANE,
    EpisodeSpec,
    FixedBehaviorLane,
    collect_episode,
    collection_implementation_identity,
    derive_episode_sample_seed,
    initial_instance_sha256,
    iter_collected_jobs,
    static_map_sha256,
    validate_paired_episode_samples,
)


@dataclass(frozen=True)
class _RawPlanBatch:
    actions: tuple
    planned_mask: tuple
    reverse_mask: tuple


class _FakeCAAR:
    def __init__(self, action_batches):
        self.action_batches = iter(action_batches)
        self.reset_count = 0
        self.after_step_calls = []
        self.grid_configs = []
        self.envs = []

    def after_reset(self):
        self.reset_count += 1

    def set_grid_config(self, grid_config):
        self.grid_configs.append(grid_config)

    def set_env(self, env):
        self.envs.append(env)

    def act(self, observations, rewards=None, dones=None, infos=None):
        actions = next(self.action_batches)
        if len(actions) != len(observations):
            raise AssertionError("fixture action count mismatch")
        return np.asarray(actions, dtype=np.int64)

    def after_step(self, dones):
        self.after_step_calls.append(tuple(bool(value) for value in dones))


class _FakePlanner:
    def __init__(self, batches):
        self.batches = iter(batches)
        self.use_best_move = True
        self.max_steps = 10_000
        self.commits = []
        self.reset_count = 0
        self._pending = None

    @property
    def pending(self):
        return self._pending

    def reset(self):
        self.reset_count += 1
        self._pending = None

    def propose(self, observations):
        self._pending = next(self.batches)
        if len(self._pending.actions) != len(observations):
            raise AssertionError("fixture proposal count mismatch")
        return self._pending

    def commit(self, mask):
        if self._pending is None:
            raise AssertionError("commit without proposal")
        self.commits.append(tuple(bool(value) for value in mask))
        self._pending = None


class _Grid:
    def __init__(self, count):
        self.obstacles = np.zeros((8, 8), dtype=np.bool_)
        self.positions_xy = np.asarray(
            [(0, index) for index in range(count)], dtype=np.int64
        )
        self.finishes_xy = np.asarray(
            [(9, index) for index in range(count)], dtype=np.int64
        )


class _FakeEnv:
    def __init__(self, rewards_by_step, map_name="random-s8-test"):
        self.rewards_by_step = [list(values) for values in rewards_by_step]
        self.count = len(self.rewards_by_step[0])
        self.grid = _Grid(self.count)
        self.grid_config = SimpleNamespace(
            num_agents=self.count,
            max_episode_steps=len(self.rewards_by_step),
            seed=7,
            map_name=map_name,
        )
        self.step_index = 0
        self.action_history = []

    def _observations(self):
        return [
            {
                "obstacles": np.zeros((5, 5), dtype=np.uint8),
                "agents": np.zeros((5, 5), dtype=np.uint8),
                "xy": position.copy(),
                "target_xy": target.copy(),
            }
            for position, target in zip(
                self.grid.positions_xy,
                self.grid.finishes_xy,
            )
        ]

    def reset(self):
        self.step_index = 0
        self.grid = _Grid(self.count)
        return self._observations(), {}

    def step(self, actions):
        self.action_history.append(tuple(int(value) for value in actions))
        rewards = self.rewards_by_step[self.step_index]
        self.step_index += 1
        self.grid.positions_xy[:, 0] += 1
        terminal = self.step_index == len(self.rewards_by_step)
        return (
            self._observations(),
            rewards,
            [terminal] * self.count,
            [False] * self.count,
            [{} for _ in range(self.count)],
        )


class _PartiallyInactiveEnv(_FakeEnv):
    def step(self, actions):
        self.action_history.append(tuple(int(value) for value in actions))
        rewards = self.rewards_by_step[self.step_index]
        self.step_index += 1
        self.grid.positions_xy[:, 0] += 1
        final_step = self.step_index == len(self.rewards_by_step)
        return (
            self._observations(),
            rewards,
            [True, final_step],
            [False, False],
            [{"is_active": False}, {"is_active": not final_step}],
        )


def _batch(actions, reverse=None):
    actions = tuple(actions)
    if reverse is None:
        reverse = (False,) * len(actions)
    return _RawPlanBatch(
        actions=actions,
        planned_mask=tuple(action is not None for action in actions),
        reverse_mask=tuple(reverse),
    )


def _matrix_converter(observations):
    result = []
    for observation in observations:
        obstacles = np.asarray(observation["obstacles"], dtype=np.float32)
        agents = np.asarray(observation["agents"], dtype=np.float32)
        target = np.zeros_like(obstacles)
        target[target.shape[0] // 2, target.shape[1] // 2] = 1.0
        result.append(
            {
                "obs": np.stack((obstacles, agents, target), axis=0),
                "xy": np.asarray(observation["xy"], dtype=np.float32),
                "target_xy": np.asarray(
                    observation["target_xy"], dtype=np.float32
                ),
            }
        )
    return result


class FixedBehaviorLaneTests(unittest.TestCase):
    def test_caar_lane_never_selects_plan_but_commits_agreement(self):
        caar = _FakeCAAR([[1, 0]])
        planner = _FakePlanner([_batch([1, 4])])
        lane = FixedBehaviorLane(CAAR_LANE, caar, planner, action_count=5)
        lane.after_reset()

        decision = lane.decide([{}, {}])

        self.assertEqual(decision.actions, (1, 0))
        self.assertEqual(decision.plan_selected_mask, (False, False))
        self.assertEqual(decision.planner_commit_mask, (True, False))
        self.assertEqual(planner.commits, [(True, False)])

    def test_ao_safe_uses_multiple_plans_and_reverse_falls_back_to_caar(self):
        caar = _FakeCAAR([[1, 2, 3, 4, 0]])
        planner = _FakePlanner(
            [
                _RawPlanBatch(
                    actions=(2, None, 4, 0, 9),
                    planned_mask=(True, False, True, True, True),
                    reverse_mask=(True, False, False, False, False),
                )
            ]
        )
        lane = FixedBehaviorLane(AO_SAFE_LANE, caar, planner, action_count=5)
        lane.after_reset()

        decision = lane.decide([{}, {}, {}, {}, {}])

        self.assertEqual(decision.actions, (1, 2, 4, 0, 0))
        self.assertEqual(
            decision.plan_selected_mask,
            (False, False, True, True, False),
        )
        self.assertEqual(
            decision.planner_commit_mask,
            (False, False, True, True, False),
        )
        self.assertEqual(planner.commits[-1], (False, False, True, True, False))

    def test_rollout_module_has_no_probe_import(self):
        tree = ast.parse(inspect.getsource(rollout_module))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
        self.assertFalse(
            [name for name in imported if "probe" in name.lower()],
            imported,
        )


class EpisodeCollectionTests(unittest.TestCase):
    def _episode(self, count=1, steps=3):
        return EpisodeSpec(
            "paired-random-7",
            {
                "num_agents": count,
                "max_episode_steps": steps,
                "seed": 7,
                "map_name": "maps/train.yaml",
            },
        )

    def test_pre_action_observation_and_mc_return_are_aligned(self):
        env = _FakeEnv([[1.0], [2.0], [3.0]])
        caar = _FakeCAAR([[0], [0], [0]])
        planner = _FakePlanner([_batch([0]), _batch([0]), _batch([0])])
        lane = FixedBehaviorLane(CAAR_LANE, caar, planner, action_count=5)

        samples = collect_episode(
            env,
            lane,
            episode=self._episode(),
            gamma=0.5,
            sample_fraction=1.0,
            sample_seed=91,
            matrix_converter=_matrix_converter,
        )

        np.testing.assert_array_equal(samples.timestep, [0, 1, 2])
        # These are o_t positions. Recording post-step observations would
        # incorrectly produce [1, 2, 3].
        np.testing.assert_array_equal(samples.xy[:, 0], [0, 1, 2])
        np.testing.assert_allclose(samples.mc_return, [2.75, 3.5, 3.0])
        np.testing.assert_array_equal(samples.terminated, [False, False, True])
        np.testing.assert_array_equal(samples.truncated, [False, False, False])
        self.assertEqual(samples.metadata["actual_map_name"], "random-s8-test")
        self.assertEqual(samples.metadata["map_family"], "random")
        self.assertEqual(len(samples.metadata["static_map_sha256"]), 64)
        self.assertEqual(len(samples.metadata["initial_instance_sha256"]), 64)
        self.assertEqual(samples.metadata["obs_shape"], [3, 5, 5])
        self.assertEqual(samples.metadata["horizon"], 3)
        self.assertEqual(
            samples.metadata["behavior_contract"]["schema_version"],
            "caar_ao_behavior_contract_v1",
        )
        self.assertEqual(
            len(samples.metadata["behavior_contract_sha256"]), 64
        )

    def test_compact_capacity_excludes_inactive_agent_rows(self):
        buffers = rollout_module._allocate_transition_buffers(6, (3, 5, 5))
        self.assertEqual(buffers["obs"].shape, (6, 3, 5, 5))
        self.assertEqual(buffers["obs"].dtype, np.uint8)
        self.assertEqual(buffers["xy"].shape, (6, 2))
        self.assertEqual(buffers["xy"].dtype, np.int32)
        self.assertEqual(buffers["reward"].shape, (6,))
        self.assertEqual(buffers["reward"].dtype, np.float32)
        self.assertEqual(buffers["caar_action"].dtype, np.int16)
        self.assertEqual(buffers["plan_valid"].dtype, np.bool_)

        env = _PartiallyInactiveEnv(
            [[1.0, 10.0], [999.0, 20.0], [999.0, 30.0]]
        )
        lane = FixedBehaviorLane(
            CAAR_LANE,
            _FakeCAAR([[0, 0], [0, 0], [0, 0]]),
            _FakePlanner(
                [_batch([0, 0]), _batch([0, 0]), _batch([0, 0])]
            ),
        )
        allocator = rollout_module._allocate_transition_buffers
        with mock.patch.object(
            rollout_module,
            "_allocate_transition_buffers",
            wraps=allocator,
        ) as allocate:
            samples = collect_episode(
                env,
                lane,
                episode=self._episode(count=2, steps=3),
                gamma=0.5,
                sample_fraction=1.0,
                sample_seed=3,
                matrix_converter=_matrix_converter,
            )
        allocate.assert_called_once_with(6, (3, 5, 5))
        self.assertEqual(samples.metadata["full_row_count"], 4)
        np.testing.assert_array_equal(samples.agent_id, [0, 1, 1, 1])
        np.testing.assert_array_equal(samples.timestep, [0, 0, 1, 2])
        np.testing.assert_allclose(samples.mc_return, [1.0, 27.5, 35.0, 30.0])

    def test_collection_implementation_hash_is_stable_and_complete(self):
        first = collection_implementation_identity(required=True)
        second = collection_implementation_identity(required=True)
        self.assertEqual(first, second)
        files = first["collection_implementation_files_sha256"]
        self.assertIn("agents/caar.py", files)
        self.assertIn("planning/planner.cpp", files)
        self.assertIn("policy_estimation/caar_ao_rollout.py", files)
        self.assertTrue(
            any(
                path.startswith("planning/planner")
                and path.endswith((".so", ".pyd"))
                for path in files
            )
        )
        self.assertEqual(
            len(first["collection_implementation_sha256"]), 64
        )

    def test_explicit_collection_identity_is_required_and_recorded(self):
        files = {"fake/collector.py": "a" * 64}
        canonical = json.dumps(
            files,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        identity = {
            "collection_implementation_files_sha256": files,
            "collection_implementation_sha256": hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest(),
        }
        with self.assertRaisesRegex(RuntimeError, "identity is required"):
            collect_episode(
                _FakeEnv([[0.0], [0.0], [0.0]]),
                FixedBehaviorLane(
                    CAAR_LANE,
                    _FakeCAAR([[0], [0], [0]]),
                    _FakePlanner(
                        [_batch([0]), _batch([0]), _batch([0])]
                    ),
                ),
                episode=self._episode(),
                sample_fraction=1.0,
                sample_seed=1,
                require_collection_identity=True,
                matrix_converter=_matrix_converter,
            )
        samples = collect_episode(
            _FakeEnv([[0.0], [0.0], [0.0]]),
            FixedBehaviorLane(
                CAAR_LANE,
                _FakeCAAR([[0], [0], [0]]),
                _FakePlanner([_batch([0]), _batch([0]), _batch([0])]),
            ),
            episode=self._episode(),
            sample_fraction=1.0,
            sample_seed=1,
            collection_identity=identity,
            require_collection_identity=True,
            matrix_converter=_matrix_converter,
        )
        self.assertEqual(
            samples.metadata["collection_implementation_sha256"],
            identity["collection_implementation_sha256"],
        )
        self.assertEqual(
            samples.metadata["behavior_contract"][
                "collection_implementation_sha256"
            ],
            identity["collection_implementation_sha256"],
        )

    def test_fixed_initial_instance_hash_matches_across_branches(self):
        episode = self._episode()
        caar_env = _FakeEnv([[0.0], [0.0], [0.0]])
        ao_env = _FakeEnv([[0.0], [0.0], [0.0]])
        caar_lane = FixedBehaviorLane(
            CAAR_LANE,
            _FakeCAAR([[0], [0], [0]]),
            _FakePlanner([_batch([1]), _batch([1]), _batch([1])]),
        )
        ao_lane = FixedBehaviorLane(
            AO_SAFE_LANE,
            _FakeCAAR([[0], [0], [0]]),
            _FakePlanner([_batch([1]), _batch([1]), _batch([1])]),
        )
        caar = collect_episode(
            caar_env,
            caar_lane,
            episode=episode,
            sample_fraction=1.0,
            sample_seed=1,
            matrix_converter=_matrix_converter,
        )
        ao = collect_episode(
            ao_env,
            ao_lane,
            episode=episode,
            sample_fraction=1.0,
            sample_seed=1,
            matrix_converter=_matrix_converter,
        )
        self.assertEqual(caar.metadata["branch"], CAAR_LANE)
        self.assertEqual(ao.metadata["branch"], AO_SAFE_LANE)
        self.assertEqual(
            caar.metadata["initial_instance_sha256"],
            ao.metadata["initial_instance_sha256"],
        )
        self.assertEqual(
            caar.metadata["static_map_sha256"],
            ao.metadata["static_map_sha256"],
        )

    def test_static_map_hash_does_not_depend_on_starts_or_targets(self):
        first = _FakeEnv([[0.0], [0.0], [0.0]])
        second = _FakeEnv([[0.0], [0.0], [0.0]])
        first_observations, _ = first.reset()
        second_observations, _ = second.reset()
        second.grid.positions_xy[:] = (4, 4)
        second.grid.finishes_xy[:] = (7, 7)
        second_observations = second._observations()

        self.assertEqual(static_map_sha256(first), static_map_sha256(second))
        self.assertNotEqual(
            initial_instance_sha256(first, first_observations),
            initial_instance_sha256(second, second_observations),
        )

    def test_paired_sampling_seed_is_branch_independent(self):
        self.assertEqual(
            derive_episode_sample_seed(17, "scenario-a", CAAR_LANE),
            derive_episode_sample_seed(17, "scenario-a", AO_SAFE_LANE),
        )

    def test_production_identity_is_required_and_recorded(self):
        episode = self._episode()
        missing_lane = FixedBehaviorLane(
            CAAR_LANE,
            _FakeCAAR([[0], [0], [0]]),
            _FakePlanner([_batch([0]), _batch([0]), _batch([0])]),
        )
        with self.assertRaisesRegex(RuntimeError, "artifact identity"):
            collect_episode(
                _FakeEnv([[0.0], [0.0], [0.0]]),
                missing_lane,
                episode=episode,
                sample_fraction=1.0,
                sample_seed=1,
                require_caar_artifact_identity=True,
                matrix_converter=_matrix_converter,
            )

        caar = _FakeCAAR([[0], [0], [0]])
        caar.checkpoint_sha256 = "1" * 64
        caar.config_sha256 = "2" * 64
        caar.checkpoint_path = "weights/caar/checkpoint.pth"
        caar.config_path = "weights/caar/config.json"
        samples = collect_episode(
            _FakeEnv([[0.0], [0.0], [0.0]]),
            FixedBehaviorLane(
                CAAR_LANE,
                caar,
                _FakePlanner([_batch([0]), _batch([0]), _batch([0])]),
            ),
            episode=episode,
            sample_fraction=1.0,
            sample_seed=1,
            require_caar_artifact_identity=True,
            matrix_converter=_matrix_converter,
        )
        self.assertEqual(samples.metadata["caar_checkpoint_sha256"], "1" * 64)
        self.assertEqual(samples.metadata["caar_config_sha256"], "2" * 64)
        self.assertEqual(
            samples.metadata["caar_checkpoint_path"],
            "weights/caar/checkpoint.pth",
        )

    def test_collector_rejects_false_pair_without_completed_manifest(self):
        from scripts.collect_caar_ao_returns import collect

        episode = self._episode()
        caar = collect_episode(
            _FakeEnv([[0.0], [0.0], [0.0]]),
            FixedBehaviorLane(
                CAAR_LANE,
                _FakeCAAR([[0], [0], [0]]),
                _FakePlanner([_batch([0]), _batch([0]), _batch([0])]),
            ),
            episode=episode,
            sample_fraction=1.0,
            sample_seed=1,
            matrix_converter=_matrix_converter,
        )
        ao = collect_episode(
            _FakeEnv([[0.0], [0.0], [0.0]]),
            FixedBehaviorLane(
                AO_SAFE_LANE,
                _FakeCAAR([[0], [0], [0]]),
                _FakePlanner([_batch([1]), _batch([1]), _batch([1])]),
            ),
            episode=episode,
            sample_fraction=1.0,
            sample_seed=1,
            matrix_converter=_matrix_converter,
        )
        ao.metadata["collection_implementation_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            validate_paired_episode_samples(caar, ao)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scenario_path = root / "scenarios.json"
            scenario_path.write_text(
                json.dumps(
                    [
                        {
                            "scenario_id": episode.scenario_id,
                            "grid_config": dict(episode.grid_config),
                        }
                    ]
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                scenario_manifest=str(scenario_path),
                output=str(root / "dataset"),
                branches=[CAAR_LANE, AO_SAFE_LANE],
                gamma=0.99,
                sample_fraction=0.2,
                sampling_seed=9,
                workers=2,
                shard_rows=100,
                caar_weights="weights/CAAR/CAAR",
                caar_checkpoint_kind="auto",
                caar_device="cpu",
                plan_use_best_move=True,
                plan_max_steps=10_000,
                torch_num_threads=1,
            )

            def fake_iterator(_jobs, *, max_workers):
                self.assertEqual(max_workers, 2)
                yield caar
                yield ao

            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                collect(args, collected_jobs_iterator=fake_iterator)
            self.assertFalse(
                (root / "dataset" / CAAR_LANE / "manifest.json").exists()
            )
            self.assertFalse(
                (root / "dataset" / AO_SAFE_LANE / "manifest.json").exists()
            )

    def test_parallel_collection_accepts_injected_worker_and_executor(self):
        values = list(
            iter_collected_jobs(
                [1, 2, 3],
                max_workers=2,
                worker=lambda value: value * 10,
                executor_factory=ThreadPoolExecutor,
            )
        )
        self.assertEqual(values, [10, 20, 30])


if __name__ == "__main__":
    unittest.main()
