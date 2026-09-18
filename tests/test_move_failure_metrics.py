import unittest

from run_experiments import _MoveFailureTracker, run_algorithm


class _SharedDestinationPolicy:
    def after_reset(self):
        pass

    def act(self, observations, rewards=None, dones=None, infos=None):
        del observations, rewards, dones, infos
        return [4, 3]

    def after_step(self, dones):
        del dones


class MoveFailureTrackerTests(unittest.TestCase):
    MOVES = ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0))

    @staticmethod
    def _observations(*positions):
        return [{"xy": list(position)} for position in positions]

    def test_counts_failed_submitted_move_as_congestion(self):
        tracker = _MoveFailureTracker(self.MOVES)
        before = self._observations((0, 0), (0, 1), (2, 2))
        pending = tracker.capture(
            actions=[1, 0, 1],
            observations=before,
            dones=[False, False, False],
            infos=[{"is_active": True}] * 3,
        )

        tracker.commit(
            pending,
            self._observations((0, 0), (0, 1), (2, 3)),
        )
        metrics = tracker.metrics()

        self.assertEqual(metrics["move_attempt_count"], 2)
        self.assertEqual(metrics["successful_move_count"], 1)
        self.assertEqual(metrics["conflict_count"], 1)
        self.assertEqual(metrics["move_failure_count"], 1)
        self.assertEqual(metrics["agent_conflict_count"], 1)
        self.assertEqual(metrics["other_or_unattributed_conflict_count"], 0)
        self.assertEqual(metrics["wait_action_count"], 1)
        self.assertEqual(metrics["conflict_step_count"], 1)
        self.assertEqual(metrics["congestion_rate"], 0.5)
        self.assertEqual(metrics["conflict_agent_step_rate"], 1 / 3)
        self.assertEqual(metrics["contention_participant_count"], 2)
        self.assertEqual(metrics["contention_participation_rate"], 2 / 3)

    def test_separates_non_agent_blockage_and_ignores_inactive_agents(self):
        tracker = _MoveFailureTracker(self.MOVES)
        before = self._observations((0, 0), (5, 5))
        pending = tracker.capture(
            actions=[1, 1],
            observations=before,
            dones=[False, True],
            infos=[{"is_active": True}, {"is_active": False}],
        )

        tracker.commit(pending, before)
        metrics = tracker.metrics()

        self.assertEqual(metrics["active_agent_step_count"], 1)
        self.assertEqual(metrics["move_attempt_count"], 1)
        self.assertEqual(metrics["conflict_count"], 1)
        self.assertEqual(metrics["agent_conflict_count"], 0)
        self.assertEqual(metrics["other_or_unattributed_conflict_count"], 1)
        self.assertEqual(metrics["congestion_rate"], 1.0)
        self.assertEqual(metrics["contention_participant_count"], 0)

    def test_counts_all_agents_competing_for_the_same_destination(self):
        tracker = _MoveFailureTracker(self.MOVES)
        before = self._observations((0, 0), (0, 2))
        pending = tracker.capture(
            actions=[1, 2],
            observations=before,
            dones=[False, False],
            infos=[{"is_active": True}] * 2,
        )

        tracker.commit(
            pending,
            self._observations((0, 1), (0, 2)),
        )
        metrics = tracker.metrics()

        self.assertEqual(metrics["move_failure_count"], 1)
        self.assertEqual(metrics["contention_participant_count"], 2)
        self.assertEqual(metrics["contention_participation_rate"], 1.0)
        self.assertEqual(metrics["vertex_flow_pair_count"], 1)
        self.assertEqual(metrics["vertex_flow_move_denominator"], 2)
        self.assertEqual(metrics["vertex_flow_pair_cost_per_move"], 0.5)
        self.assertEqual(metrics["vertex_flow_contested_destination_count"], 1)
        self.assertEqual(metrics["vertex_flow_max_inflow"], 2)

    def test_vertex_flow_cost_counts_pairs_not_only_participants(self):
        tracker = _MoveFailureTracker(self.MOVES)
        before = self._observations((0, 0), (0, 2), (1, 1), (3, 0), (3, 2))
        pending = tracker.capture(
            actions=[1, 2, 4, 1, 2],
            observations=before,
            dones=[False] * 5,
            infos=[{"is_active": True}] * 5,
        )

        tracker.commit(pending, before)
        metrics = tracker.metrics()

        # Three proposals enter (0, 1), contributing C(3, 2) = 3 pairs.
        # Two proposals enter (3, 1), contributing C(2, 2) = 1 pair.
        self.assertEqual(metrics["vertex_flow_pair_count"], 4)
        self.assertEqual(metrics["vertex_flow_move_denominator"], 5)
        self.assertEqual(metrics["vertex_flow_pair_cost_per_move"], 0.8)
        self.assertEqual(metrics["vertex_flow_contested_destination_count"], 2)
        self.assertEqual(metrics["vertex_flow_step_count"], 1)
        self.assertEqual(metrics["vertex_flow_max_inflow"], 3)

    def test_vertex_flow_excludes_obstacles_and_out_of_bounds_targets(self):
        obstacle_mask = [
            [False, True, False],
            [False, False, False],
            [False, False, False],
        ]
        tracker = _MoveFailureTracker(
            self.MOVES,
            obstacle_mask=obstacle_mask,
        )
        before = self._observations((0, 0), (0, 2), (2, 0))
        pending = tracker.capture(
            actions=[1, 2, 3],
            observations=before,
            dones=[False] * 3,
            infos=[{"is_active": True}] * 3,
        )

        tracker.commit(pending, before)
        metrics = tracker.metrics()

        # All three remain ordinary movement attempts, but the first two target
        # an obstacle and the third leaves the map, so none is a graph-edge flow.
        self.assertEqual(metrics["move_attempt_count"], 3)
        self.assertEqual(metrics["vertex_flow_move_denominator"], 0)
        self.assertEqual(metrics["vertex_flow_pair_count"], 0)
        self.assertEqual(metrics["vertex_flow_pair_cost_per_move"], 0.0)

    def test_counts_both_agents_in_an_edge_swap(self):
        tracker = _MoveFailureTracker(self.MOVES)
        before = self._observations((0, 0), (0, 1))
        pending = tracker.capture(
            actions=[1, 2],
            observations=before,
            dones=[False, False],
            infos=[{"is_active": True}] * 2,
        )

        tracker.commit(pending, before)
        metrics = tracker.metrics()

        self.assertEqual(metrics["move_failure_count"], 2)
        self.assertEqual(metrics["contention_participant_count"], 2)
        self.assertEqual(metrics["contention_step_count"], 1)

    def test_does_not_count_a_successful_following_move_as_contention(self):
        tracker = _MoveFailureTracker(self.MOVES)
        before = self._observations((0, 0), (0, 1))
        pending = tracker.capture(
            actions=[1, 1],
            observations=before,
            dones=[False, False],
            infos=[{"is_active": True}] * 2,
        )

        tracker.commit(
            pending,
            self._observations((0, 1), (0, 2)),
        )
        metrics = tracker.metrics()

        self.assertEqual(metrics["move_failure_count"], 0)
        self.assertEqual(metrics["contention_participant_count"], 0)

    def test_uses_global_positions_instead_of_egocentric_observation_xy(self):
        tracker = _MoveFailureTracker(self.MOVES)
        egocentric = self._observations((0, 0), (0, 0))
        pending = tracker.capture(
            actions=[1, 2],
            observations=egocentric,
            dones=[False, False],
            infos=[{"is_active": True}] * 2,
            global_positions=[(4, 4), (4, 6)],
        )

        tracker.commit(
            pending,
            egocentric,
            global_positions=[(4, 5), (4, 6)],
        )
        metrics = tracker.metrics()

        self.assertEqual(metrics["contention_participant_count"], 2)
        self.assertEqual(metrics["contention_participation_rate"], 1.0)

    def test_single_episode_runner_keeps_final_step_for_tracker_commit(self):
        result = run_algorithm(
            _SharedDestinationPolicy(),
            map_name="final-step-diagnostic",
            max_episode_steps=1,
            seed=0,
            num_agents=2,
            obs_radius=1,
            animate=False,
            on_target="finish",
            collision_system="block_both",
            map_text="\n".join(["......."] * 5),
            agents_xy=[[2, 1], [2, 3]],
            targets_xy=[[2, 5], [2, 0]],
        )

        self.assertEqual(result["environment_step_count_observed"], 1)
        self.assertEqual(result["move_attempt_count"], 2)
        self.assertEqual(result["move_failure_count"], 2)
        self.assertEqual(result["congestion_rate"], 1.0)

if __name__ == "__main__":
    unittest.main()
