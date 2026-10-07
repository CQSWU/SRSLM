import unittest
from planning.aoreplan import planner

INF = 1_000_000_000
START = (0, 0)
GOAL = (0, 2)
FIRST_STEP = (0, 1)
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

        local_planner.cancel_desired()
        local_planner.update_obstacles([], [], START)
        local_planner.observe_position(START)
        local_planner.plan_path(START, GOAL)
        self.assertEqual(_next_node(local_planner), FIRST_STEP)

    def test_uncancelled_failure_avoids_failed_destination(self):
        local_planner = _new_planner()
        local_planner.observe_position(START)
        local_planner.plan_path(START, GOAL)
        self.assertEqual(_next_node(local_planner), FIRST_STEP)
        local_planner.update_obstacles([], [], START)
        local_planner.observe_position(START)
        local_planner.plan_path(START, GOAL)
        next_node = _next_node(local_planner)
        self.assertNotEqual(next_node, FIRST_STEP)
        self.assertLess(next_node[0], INF)

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

        local_planner.update_obstacles([], [], MOVED_START)
        local_planner.observe_position(MOVED_START)
        local_planner.plan_path(MOVED_START, START)
        self.assertEqual(_next_node(local_planner), FIRST_STEP)
        local_planner.cancel_desired()

        local_planner.update_obstacles([], [], MOVED_START)
        local_planner.observe_position(MOVED_START)
        local_planner.plan_path(MOVED_START, START)
        self.assertEqual(_next_node(local_planner), FIRST_STEP)


if __name__ == "__main__":
    unittest.main()
