import importlib
import unittest

importlib.import_module("cppimport.import_hook")

from planning.planner import planner


INF = 1_000_000_000
START = (0, 0)
GOAL = (0, 2)
FIRST_STEP = (0, 1)
SECOND_GOAL = (2, 0)
SECOND_FIRST_STEP = (1, 0)
MOVED_START = (0, 2)


def _new_planner():
    local_planner = planner(100)
    local_planner.update_obstacles([], [], START)
    return local_planner


def _next_node(local_planner):
    return local_planner.get_next_node(False)[1]


class PlannerExecutionFeedbackTests(unittest.TestCase):
    def test_cancelled_proposal_is_not_learned_as_failed(self):
        local_planner = _new_planner()
        local_planner.observe_position(START)
        local_planner.plan_path(START, GOAL)
        self.assertEqual(_next_node(local_planner), FIRST_STEP)

        # Arbitration selected another policy, so this AO-RePlan proposal was
        # never executed and must not become a bad action.
        local_planner.cancel_desired()
        local_planner.update_obstacles([], [], START)
        local_planner.observe_position(START)
        local_planner.plan_path(START, GOAL)
        self.assertEqual(_next_node(local_planner), FIRST_STEP)

    def test_uncancelled_failure_keeps_legacy_avoidance(self):
        legacy = _new_planner()
        split = _new_planner()

        legacy.update_path(START, GOAL)
        split.observe_position(START)
        split.plan_path(START, GOAL)
        self.assertEqual(_next_node(legacy), FIRST_STEP)
        self.assertEqual(_next_node(split), FIRST_STEP)

        # Both planners stayed at START after proposing FIRST_STEP.  The split
        # API must consume that failure exactly like legacy update_path.
        legacy.update_obstacles([], [], START)
        split.update_obstacles([], [], START)
        legacy.update_path(START, GOAL)
        split.observe_position(START)
        split.plan_path(START, GOAL)
        legacy_next = _next_node(legacy)
        split_next = _next_node(split)

        self.assertEqual(split_next, legacy_next)
        self.assertNotEqual(split_next, FIRST_STEP)
        self.assertLess(split_next[0], INF)

    def test_successful_feedback_is_not_marked_as_failed(self):
        local_planner = _new_planner()
        local_planner.observe_position(START)
        local_planner.plan_path(START, GOAL)
        self.assertEqual(_next_node(local_planner), FIRST_STEP)

        local_planner.update_obstacles([], [], FIRST_STEP)
        local_planner.observe_position(FIRST_STEP)
        local_planner.plan_path(FIRST_STEP, GOAL)
        self.assertEqual(_next_node(local_planner), GOAL)

    def test_observe_without_pending_proposal_preserves_failed_actions(self):
        local_planner = _new_planner()
        local_planner.plan_path(START, GOAL)
        self.assertEqual(_next_node(local_planner), FIRST_STEP)

        # The requested move failed.  The first observe consumes that real
        # feedback; subsequent observes have no proposal to consume and must
        # not erase the recorded failure.
        local_planner.observe_position(START)
        local_planner.observe_position(START)
        local_planner.plan_path(START, GOAL)
        next_node = _next_node(local_planner)
        self.assertNotEqual(next_node, FIRST_STEP)
        self.assertLess(next_node[0], INF)

    def test_failure_survives_obstacle_refresh_during_skipped_step(self):
        local_planner = _new_planner()
        local_planner.plan_path(START, GOAL)
        self.assertEqual(_next_node(local_planner), FIRST_STEP)

        # The selected planner proposal failed, then the switcher used ARPE
        # for the current step. That step and the next planner query refresh dynamic
        # obstacles; the real failure must still be applied at the same start.
        local_planner.update_obstacles([], [], START)
        local_planner.observe_position(START)
        local_planner.update_obstacles([], [], START)
        local_planner.observe_position(START)
        local_planner.plan_path(START, GOAL)
        next_node = _next_node(local_planner)
        self.assertNotEqual(next_node, FIRST_STEP)
        self.assertLess(next_node[0], INF)

    def test_failure_memory_does_not_become_ghost_at_new_start(self):
        local_planner = _new_planner()
        local_planner.plan_path(START, GOAL)
        self.assertEqual(_next_node(local_planner), FIRST_STEP)
        local_planner.update_obstacles([], [], START)
        local_planner.observe_position(START)

        # ARPE moved the agent to another start before AO was queried again.
        # The old A->B failure is local to A and must not poison later plans
        # from C merely because C becomes planner.start.
        local_planner.update_obstacles([], [], MOVED_START)
        local_planner.observe_position(MOVED_START)
        local_planner.plan_path(MOVED_START, START)
        self.assertEqual(_next_node(local_planner), FIRST_STEP)
        local_planner.cancel_desired()

        local_planner.update_obstacles([], [], MOVED_START)
        local_planner.observe_position(MOVED_START)
        local_planner.plan_path(MOVED_START, START)
        self.assertEqual(_next_node(local_planner), FIRST_STEP)

    def test_legacy_update_path_without_pending_clears_failed_actions(self):
        local_planner = _new_planner()
        local_planner.plan_path(START, GOAL)
        self.assertEqual(_next_node(local_planner), FIRST_STEP)
        local_planner.observe_position(START)

        # Legacy update_path historically clears bad_actions when no proposal
        # is pending.  Plan toward a different first move, then fail that move:
        # only the new failure may be avoided when planning toward GOAL again.
        local_planner.update_obstacles([], [], START)
        local_planner.update_path(START, SECOND_GOAL)
        self.assertEqual(_next_node(local_planner), SECOND_FIRST_STEP)

        local_planner.update_obstacles([], [], START)
        local_planner.update_path(START, GOAL)
        self.assertEqual(_next_node(local_planner), FIRST_STEP)


if __name__ == "__main__":
    unittest.main()
