from planning.python_planner import INF, planner


START = (0, 0)
GOAL = (0, 2)
FIRST_STEP = (0, 1)


def _new_planner():
    result = planner(100)
    result.update_obstacles([], [], START)
    return result


def test_python_planner_returns_cpp_compatible_path_and_next_node():
    local_planner = _new_planner()
    local_planner.plan_path(START, GOAL)

    assert local_planner.get_path(False) == [START, FIRST_STEP, GOAL]

    # Replan because get_path, like the C++ binding, records GOAL as its
    # pending desired position rather than the primitive first step.
    local_planner.cancel_desired()
    local_planner.plan_path(START, GOAL)
    assert local_planner.get_next_node(False) == (START, FIRST_STEP)


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
    assert local_planner.get_path(False) == []

