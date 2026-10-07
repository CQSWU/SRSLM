import sys

import pytest

from planning.planner_python import INF, planner

START = (0, 0)
GOAL = (0, 2)
FIRST_STEP = (0, 1)


def _new_planner():
    result = planner(100)
    result.update_obstacles([], [], START)
    return result


def test_python_planner_returns_cpp_compatible_next_node():
    local_planner = _new_planner()
    local_planner.plan_path(START, GOAL)

    assert local_planner.get_next_node(False) == (START, FIRST_STEP)
    assert local_planner.desired_position == FIRST_STEP


def test_path_execution_feedback_caches_the_failed_first_step_not_goal():
    local_planner = _new_planner()
    local_planner.plan_path(START, GOAL)
    local_planner.get_next_node(False)
    local_planner.observe_position(START)
    assert FIRST_STEP in local_planner.bad_actions
    assert GOAL not in local_planner.bad_actions


def test_relative_obstacles_and_dynamic_agents_block_the_same_cells_as_cpp():
    local_planner = planner(100)
    shifted_start = (10, 10)
    shifted_goal = (10, 12)
    local_planner.update_obstacles([(0, 1)], [], shifted_start)
    local_planner.plan_path(shifted_start, shifted_goal)
    static_step = local_planner.get_next_node(False)[1]
    assert static_step != (10, 11)
    assert static_step[0] < INF

    dynamic = _new_planner()
    dynamic.update_obstacles([], [(0, 1)], START)
    dynamic.plan_path(START, GOAL)
    dynamic_step = dynamic.get_next_node(False)[1]
    assert dynamic_step != FIRST_STEP
    assert dynamic_step[0] < INF


def test_execution_feedback_and_cancel_match_cpp_state_transitions():
    failed = _new_planner()
    failed.plan_path(START, GOAL)
    assert failed.get_next_node(False)[1] == FIRST_STEP
    failed.observe_position(START)
    failed.update_obstacles([], [], START)
    failed.plan_path(START, GOAL)
    assert failed.get_next_node(False)[1] != FIRST_STEP

    cancelled = _new_planner()
    cancelled.plan_path(START, GOAL)
    assert cancelled.get_next_node(False)[1] == FIRST_STEP
    cancelled.cancel_desired()
    cancelled.update_obstacles([], [], START)
    cancelled.observe_position(START)
    cancelled.plan_path(START, GOAL)
    assert cancelled.get_next_node(False)[1] == FIRST_STEP


def test_no_exact_path_uses_cpp_inf_sentinel():
    local_planner = _new_planner()
    local_planner.update_obstacles(
        [(0, 1), (1, 0), (-1, 0), (0, -1)],
        [],
        START,
    )
    local_planner.plan_path(START, GOAL)

    assert local_planner.get_next_node(False) == (START, (INF, INF))


def test_limited_search_retains_best_move_without_an_exact_path():
    local_planner = planner(2)
    local_planner.plan_path(START, (0, 3))
    assert local_planner.get_next_node(True) == (START, FIRST_STEP)
    assert local_planner.get_next_node(False) == (START, (INF, INF))


def _backend_trace(backend, steps, cache_failure):
    local_planner = backend(steps)
    border = [
        (i, j)
        for i in range(7)
        for j in range(7)
        if i in (0, 6) or j in (0, 6)
    ]
    start, goal = (3, 1), (3, 5)
    local_planner.update_obstacles(border + [(3, 3)], [], (0, 0))
    trace = []
    for occupied in ([], [(2, 1)], []):
        local_planner.update_obstacles([], occupied, (0, 0))
        trace.append(local_planner.proposal_failed(start))
        local_planner.observe_position(start, cache_failure)
        local_planner.plan_path(start, goal)
        trace.append(local_planner.get_next_node(True))
    local_planner.cancel_desired()
    trace.append(local_planner.proposal_failed(start))
    local_planner.observe_position(start, cache_failure)
    local_planner.release_failed_actions()
    local_planner.plan_path(start, goal)
    trace.append(local_planner.get_next_node(False))
    local_planner.update_static_path(start, goal)
    trace.append(local_planner.get_next_node(False))
    local_planner.cancel_desired()

    local_planner.update_obstacles([(0, 1)], [], start)
    moved = (2, 1)
    local_planner.observe_position(moved, cache_failure)
    local_planner.update_obstacles([], [], moved)
    local_planner.plan_path(moved, goal)
    trace.append(local_planner.get_next_node(True))
    local_planner.update_obstacles([(1, 0), (-1, 0), (0, 1), (0, -1)], [], moved)
    local_planner.observe_position(moved, cache_failure)
    local_planner.plan_path(moved, goal)
    trace.append(local_planner.get_next_node(True))
    trace.append(local_planner.get_next_node(False))
    return trace


@pytest.mark.skipif(sys.platform == "win32", reason="Native backend is not used on Windows")
@pytest.mark.parametrize("steps", [1, 2, 8, 100])
@pytest.mark.parametrize("cache_failure", [False, True])
def test_native_and_python_backends_match_state_transition_trace(steps, cache_failure):
    from planning.aoreplan import planner as NativePlanner

    assert _backend_trace(planner, steps, cache_failure) == _backend_trace(
        NativePlanner, steps, cache_failure
    )
