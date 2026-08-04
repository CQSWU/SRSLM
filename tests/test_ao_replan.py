import unittest
from types import SimpleNamespace

import numpy as np

from agents.ao_replan import AORePlan, AORePlanConfig
from planning.ao_replan_algo import (
    AORePlanBase,
    AORePlanWrapper,
    FixNonesWrapper,
    IgnoreAgentsProbe,
    NoPathSoRandomOrStayWrapper,
    StaticMapMemoryProbe,
)


class _ActionSequence:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.rnd = np.random.default_rng(0)

    def act(self, obs, skip_agents=None):
        return [next(self.actions)]


class _FixedProbe:
    def __init__(self, action):
        self.action = action

    def get_action(self, obs, agent_index=0):
        return self.action


class _FallbackRng:
    def __init__(self, values):
        self.values = iter(values)

    def random(self):
        return next(self.values)

    def shuffle(self, actions):
        return None


def _obs(position=(5, 5), target=(5, 5)):
    return [{"xy": position, "target_xy": target}]


def _planning_obs(position=(5, 5), target=(5, 8)):
    return [{
        "obstacles": np.zeros((11, 11), dtype=np.int8),
        "agents": np.zeros((11, 11), dtype=np.int8),
        "xy": position,
        "target_xy": target,
    }]


class AORePlanTests(unittest.TestCase):
    def test_stateless_local_probe_exposes_complete_cardinal_path(self):
        algorithm = AORePlan(AORePlanConfig(max_planning_steps=100))
        algorithm.after_reset()
        observation = _planning_obs()[0]
        original_obstacles = observation["obstacles"].copy()

        path = algorithm.probe_path(observation, agent_index=0)

        self.assertIsNotNone(path)
        self.assertEqual(path[0], (5, 5))
        self.assertEqual(path[-1], (5, 8))
        self.assertTrue(
            all(
                abs(first[0] - second[0])
                + abs(first[1] - second[1])
                == 1
                for first, second in zip(path, path[1:])
            )
        )
        np.testing.assert_array_equal(
            observation["obstacles"],
            original_obstacles,
        )

    def test_probe_path_validator_rejects_partial_and_diagonal_paths(self):
        self.assertIsNone(
            IgnoreAgentsProbe._complete_path(
                [(0, 0), (0, 1)],
                (0, 0),
                (0, 2),
            )
        )
        self.assertIsNone(
            IgnoreAgentsProbe._complete_path(
                [(0, 0), (1, 1), (1, 2)],
                (0, 0),
                (1, 2),
            )
        )

    def test_cancelled_selective_proposal_is_not_learned_as_failure(self):
        base = AORePlanBase(use_best_move=False, max_steps=100, seed=0)
        observation = _planning_obs()
        first = base.act(observation, skip_agents=[False])
        self.assertNotEqual(first, [None])

        base.commit_proposals([False])
        self.assertEqual(
            base.act(observation, skip_agents=[False]),
            first,
        )

    def test_committed_failure_is_consumed_during_skipped_step(self):
        base = AORePlanBase(use_best_move=False, max_steps=100, seed=0)
        observation = _planning_obs()
        first = base.act(observation, skip_agents=[False])
        self.assertNotEqual(first, [None])
        base.commit_proposals([True])

        self.assertEqual(
            base.act(observation, skip_agents=[True]),
            [None],
        )
        second = base.act(observation, skip_agents=[False])
        self.assertNotEqual(second, first)
        self.assertNotEqual(second, [None])

    def test_static_probe_remembers_obstacles_per_agent(self):
        probe = StaticMapMemoryProbe()
        obstacles = np.zeros((11, 11), dtype=np.int8)
        obstacles[5, 6] = 1
        blocked_view = {
            "obstacles": obstacles,
            "xy": (5, 5),
            "target_xy": (5, 8),
        }
        clear_view = {
            "obstacles": np.zeros((11, 11), dtype=np.int8),
            "xy": (5, 5),
            "target_xy": (5, 8),
        }

        first_action = probe.get_action(blocked_view, agent_index=0)
        self.assertNotEqual(first_action, 4)
        self.assertEqual(probe.get_action(clear_view, agent_index=0), first_action)
        self.assertEqual(probe.get_action(clear_view, agent_index=1), 4)

    def test_teammate_caused_reverse_executes_probe_action(self):
        wrapper = AORePlanWrapper(_ActionSequence([1, 2]))
        wrapper.probe = _FixedProbe(3)

        self.assertEqual(wrapper.act(_obs(position=(5, 5))), [1])
        self.assertEqual(wrapper.last_planned_mask, [True])
        self.assertEqual(wrapper.last_probe_replacement_mask, [False])
        self.assertEqual(wrapper.act(_obs(position=(4, 5))), [3])
        self.assertEqual(wrapper.last_teammate_blocked_mask, [False])
        self.assertEqual(wrapper.last_planned_mask, [True])
        self.assertEqual(wrapper.last_probe_replacement_mask, [True])

    def test_handoff_preserves_raw_dynamic_and_static_diagnostics(self):
        wrapper = AORePlanWrapper(
            _ActionSequence([1, 2]),
            handoff_on_reverse=True,
        )
        wrapper.probe = _FixedProbe(3)

        self.assertEqual(wrapper.act(_obs(position=(5, 5))), [1])
        self.assertEqual(wrapper.act(_obs(position=(4, 5))), [None])
        self.assertEqual(wrapper.last_raw_dynamic_actions, [2])
        self.assertEqual(wrapper.last_static_probe_actions, [3])
        self.assertEqual(wrapper.last_teammate_blocked_mask, [True])
        self.assertEqual(wrapper.last_forward_clear_mask, [False])

    def test_forward_clear_requires_nonzero_agreement_without_return(self):
        wrapper = AORePlanWrapper(
            _ActionSequence([4, 0, 4]),
            handoff_on_reverse=True,
        )
        wrapper.probe = _FixedProbe(4)

        self.assertEqual(wrapper.act(_obs()), [4])
        self.assertEqual(wrapper.last_forward_clear_mask, [True])
        self.assertEqual(wrapper.last_static_probe_actions, [4])

        self.assertEqual(wrapper.act(_obs()), [0])
        self.assertEqual(wrapper.last_forward_clear_mask, [False])
        self.assertEqual(wrapper.last_static_probe_actions, [None])

        self.assertEqual(
            wrapper.act(_obs(), skip_agents=[True]),
            [4],
        )
        self.assertEqual(wrapper.last_forward_clear_mask, [False])
        self.assertEqual(wrapper.last_static_probe_actions, [None])

    def test_real_planner_action_is_marked_planned(self):
        wrapper = AORePlanWrapper(_ActionSequence([4]))

        self.assertEqual(wrapper.act(_obs()), [4])
        self.assertEqual(wrapper.last_planned_mask, [True])

    def test_no_path_wait_and_random_fallbacks_are_not_marked_planned(self):
        action_source = _ActionSequence([None, None])
        action_source.rnd = _FallbackRng([0.25, 0.75])
        ao_wrapper = AORePlanWrapper(action_source)
        outer_wrapper = NoPathSoRandomOrStayWrapper(ao_wrapper)

        self.assertEqual(outer_wrapper.act(_planning_obs()), [0])
        self.assertEqual(ao_wrapper.last_planned_mask, [False])
        self.assertEqual(outer_wrapper.act(_planning_obs()), [1])
        self.assertEqual(ao_wrapper.last_planned_mask, [False])

    def test_no_path_wrapper_preserves_none_for_skipped_agent(self):
        action_source = _ActionSequence([None])
        action_source.rnd = _FallbackRng([0.25])
        ao_wrapper = AORePlanWrapper(action_source)
        outer_wrapper = NoPathSoRandomOrStayWrapper(ao_wrapper)

        self.assertEqual(
            outer_wrapper.act(_planning_obs(), skip_agents=[True]),
            [None],
        )
        self.assertEqual(ao_wrapper.last_planned_mask, [False])

    def test_fix_nones_wrapper_preserves_none_for_skipped_agent(self):
        ao_wrapper = AORePlanWrapper(_ActionSequence([None]))
        outer_wrapper = FixNonesWrapper(ao_wrapper)

        self.assertEqual(
            outer_wrapper.act(_planning_obs(), skip_agents=[True]),
            [None],
        )
        self.assertEqual(ao_wrapper.last_planned_mask, [False])

    def test_last_planned_mask_is_read_only_and_reset(self):
        algorithm = AORePlan(AORePlanConfig())

        self.assertIsNone(algorithm.last_planned_mask)
        self.assertIsNone(algorithm.teammate_blocked_mask)
        self.assertIsNone(algorithm.raw_dynamic_actions)
        self.assertIsNone(algorithm.static_probe_actions)
        self.assertIsNone(algorithm.forward_clear_mask)
        algorithm.after_reset()
        self.assertIsNone(algorithm.last_planned_mask)

        algorithm._ao_wrapper.last_planned_mask = [True, False]
        algorithm._ao_wrapper.last_teammate_blocked_mask = [True, False]
        algorithm._ao_wrapper.last_raw_dynamic_actions = [2, None]
        algorithm._ao_wrapper.last_static_probe_actions = [3, None]
        algorithm._ao_wrapper.last_forward_clear_mask = [False, True]
        self.assertEqual(algorithm.last_planned_mask, (True, False))
        self.assertEqual(algorithm.teammate_blocked_mask, (True, False))
        self.assertEqual(algorithm.raw_dynamic_actions, (2, None))
        self.assertEqual(algorithm.static_probe_actions, (3, None))
        self.assertEqual(algorithm.forward_clear_mask, (False, True))
        blocked_snapshot = algorithm.teammate_blocked_mask
        algorithm._ao_wrapper.last_teammate_blocked_mask[0] = False
        self.assertEqual(blocked_snapshot, (True, False))
        with self.assertRaises(AttributeError):
            algorithm.last_planned_mask = (False, False)
        with self.assertRaises(AttributeError):
            algorithm.teammate_blocked_mask = (False, False)
        with self.assertRaises(AttributeError):
            algorithm.raw_dynamic_actions = (None, None)
        with self.assertRaises(AttributeError):
            algorithm.static_probe_actions = (None, None)
        with self.assertRaises(AttributeError):
            algorithm.forward_clear_mask = (False, False)

        algorithm.after_reset()
        self.assertIsNone(algorithm.last_planned_mask)
        self.assertIsNone(algorithm.teammate_blocked_mask)
        self.assertIsNone(algorithm.raw_dynamic_actions)
        self.assertIsNone(algorithm.static_probe_actions)
        self.assertIsNone(algorithm.forward_clear_mask)

    def test_base_does_not_retain_unbounded_unused_position_history(self):
        base = AORePlanBase(use_best_move=False, max_steps=100, seed=0)
        observation = _planning_obs()

        for _ in range(5):
            base.act(observation, skip_agents=[True])

        self.assertFalse(hasattr(base, "previous_positions"))

    def test_commit_cancels_replaced_base_proposal(self):
        algorithm = AORePlan(AORePlanConfig())
        sink = SimpleNamespace(committed=None)

        def commit(mask):
            sink.committed = list(mask)

        algorithm._base = SimpleNamespace(commit_proposals=commit)
        algorithm._ao_wrapper = SimpleNamespace(
            last_planned_mask=[True, True],
            last_probe_replacement_mask=[False, True],
        )
        algorithm.commit_proposals([True, True])
        self.assertEqual(sink.committed, [True, False])

    def test_commit_rejects_unplanned_fallback(self):
        algorithm = AORePlan(AORePlanConfig())
        algorithm._base = SimpleNamespace(commit_proposals=lambda mask: None)
        algorithm._ao_wrapper = SimpleNamespace(
            last_planned_mask=[False],
            last_probe_replacement_mask=[False],
        )
        with self.assertRaisesRegex(ValueError, "Only planned"):
            algorithm.commit_proposals([True])

    def test_map_caused_reverse_keeps_original_action(self):
        wrapper = AORePlanWrapper(_ActionSequence([1, 2]))
        wrapper.probe = _FixedProbe(2)

        self.assertEqual(wrapper.act(_obs(position=(5, 5))), [1])
        self.assertEqual(wrapper.act(_obs(position=(4, 5))), [2])
        self.assertEqual(wrapper.last_teammate_blocked_mask, [False])

    def test_blocked_action_is_not_mistaken_for_executed_reverse(self):
        wrapper = AORePlanWrapper(_ActionSequence([1, 2]))
        self.assertEqual(wrapper.act(_obs(position=(5, 5))), [1])
        self.assertEqual(wrapper.act(_obs(position=(5, 5))), [2])

    def test_target_change_clears_position_history(self):
        wrapper = AORePlanWrapper(_ActionSequence([1, 2]))
        self.assertEqual(
            wrapper.act(_obs(position=(5, 5), target=(1, 1))),
            [1],
        )
        self.assertEqual(
            wrapper.act(_obs(position=(4, 5), target=(8, 8))),
            [2],
        )

    def test_standalone_ao_replan_does_not_handoff_none_actions(self):
        self.assertFalse(AORePlanConfig().handoff_on_reverse)


if __name__ == "__main__":
    unittest.main()
