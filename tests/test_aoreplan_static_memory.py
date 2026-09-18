"""Static probes use each agent's observed walls, not just its latest crop.

Every input below is an 11x11 observation. No full-map API is passed to the
policy. On Linux the same assertions also exercise the native planner.
"""

import sys

import numpy as np
import pytest

import planning.ao_replan_algo as ao
from agents.ao_replan import AORePlan, AORePlanConfig
from planning.aoreplan_branch import AORePlanBranch
from planning.python_planner import planner as PythonPlanner


BACKENDS = [("python", PythonPlanner)]
if sys.platform != "win32":
    from planning.planner import planner as NativePlanner

    BACKENDS.append(("native", NativePlanner))

MOVES = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
MAX_STEPS = 1000
CURRENT = (0, 0)
HISTORY = (-1, 5)
GOAL = (0, 9)
NEAR_WALLS = {(0, col) for col in range(1, 6)} | {
    (-2, col) for col in range(6)
}
REMOTE_WALL = (-1, 6)
WALLS = NEAR_WALLS | {REMOTE_WALL}


def observation(position=CURRENT, target=GOAL, *, walls=WALLS, occupied=()):
    """Crop a synthetic scene into the only map information the policy sees."""
    position = tuple(position)
    obstacles = np.zeros((11, 11), dtype=np.int8)
    agents = np.zeros_like(obstacles)
    for cells, output in ((walls, obstacles), (occupied, agents)):
        for row, col in cells:
            di, dj = row - position[0], col - position[1]
            if -5 <= di <= 5 and -5 <= dj <= 5:
                output[5 + di, 5 + dj] = 1
    assert obstacles[5, 5] == 0, "The supplied current position must be free."
    return {
        "xy": position,
        "target_xy": tuple(target),
        "obstacles": obstacles,
        "agents": agents,
    }


def first_action(local_planner, *, exact=True):
    start, destination = local_planner.get_next_node(not exact)
    if destination[0] >= ao.INF:
        return None
    return MOVES.index((destination[0] - start[0], destination[1] - start[1]))


@pytest.fixture(params=BACKENDS, ids=[name for name, _ in BACKENDS])
def backend(request, monkeypatch):
    monkeypatch.setattr(ao, "planner", request.param[1])
    return request.param[1]


def new_check():
    return ao.StaticAStarCheck(max_steps=MAX_STEPS)


def fresh_action(obs):
    check = new_check()
    check.observe([obs])
    return check.get_action(0, obs)


class IdleBase:
    def __init__(self):
        self.rnd = np.random.default_rng(0)

    def act(self, observations, skip_agents=None):
        skip = skip_agents or [False] * len(observations)
        return [
            None if skip[index] or obs["xy"] == obs["target_xy"] else 0
            for index, obs in enumerate(observations)
        ]


class SequenceBase:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.rnd = np.random.default_rng(0)

    def act(self, observations, skip_agents=None):
        assert len(observations) == 1
        assert not skip_agents or not skip_agents[0]
        return [next(self.actions)]


def test_remote_previously_observed_wall_changes_static_first_step(backend):
    current = observation()
    seen = observation(HISTORY)
    # The remote wall is present in history but lies beyond the current crop.
    assert seen["obstacles"][5, 6] == 1
    assert REMOTE_WALL[1] - CURRENT[1] > 5
    check = new_check()
    check.observe([seen])
    planner_id = check._planners[0]
    check.observe([current])
    assert check._planners[0] is planner_id
    assert fresh_action(current) == 1  # Up into the apparently open corridor.
    assert check.get_action(0, current) == 2  # Down around its remembered end.


def test_observed_walls_are_private_to_each_agent(backend):
    check = new_check()
    check.observe([
        observation(HISTORY),
        observation(HISTORY, walls=NEAR_WALLS),
    ])
    current = observation()
    check.observe([current, current])
    assert check._planners[0] is not check._planners[1]
    assert check.get_action(0, current) == 2
    assert check.get_action(1, current) == 1


def test_non_probe_steps_accumulate_walls_before_first_reverse(backend):
    # A physically possible leftward walk through the upper corridor, then
    # down to CURRENT. Only the final, upward proposal is a reverse.
    wrapper = ao.AORePlanWrapper(
        SequenceBase([3] * 5 + [2, 1]), max_steps=MAX_STEPS
    )
    for col in range(5, -1, -1):
        assert wrapper.act([observation((-1, col))]) == [3 if col else 2]
        assert wrapper.last_static_astar_invoked_mask == [False]
    assert wrapper.act([observation()]) == [2]
    assert wrapper.last_reverse_mask == [True]
    assert wrapper.last_static_astar_invoked_mask == [True]
    assert fresh_action(observation()) == 1


@pytest.mark.parametrize("inactive", ["skip", "goal"])
def test_skipped_and_at_goal_steps_still_observe_walls(backend, inactive):
    wrapper = ao.AORePlanWrapper(IdleBase(), max_steps=MAX_STEPS)
    old = observation(HISTORY, HISTORY if inactive == "goal" else GOAL)
    wrapper.act([old], skip_agents=[inactive == "skip"])
    assert wrapper.last_static_astar_invoked_mask == [False]
    wrapper.act([observation()], skip_agents=[False])
    assert wrapper.static_astar.get_action(0, observation()) == 2


def test_target_changes_preserve_static_memory(backend):
    wrapper = ao.AORePlanWrapper(IdleBase(), max_steps=MAX_STEPS)
    wrapper.act([observation(HISTORY, (0, 8))])
    original_planner = wrapper.static_astar._planners[0]
    for target in ((0, -3), GOAL):
        wrapper.act([observation(target=target)])
        assert wrapper.static_astar._planners[0] is original_planner
        assert wrapper.last_static_astar_invoked_mask == [False]
    assert wrapper.static_astar.get_action(0, observation()) == 2


@pytest.mark.parametrize("adapter", ["standalone", "branch"])
def test_episode_reset_discards_previous_episode_walls(backend, adapter):
    if adapter == "standalone":
        policy = AORePlan(AORePlanConfig(max_planning_steps=MAX_STEPS))
        policy.after_reset()

        def step(obs):
            policy.act([obs])

        reset = policy.after_reset
        get_wrapper = lambda: policy._ao_wrapper
    else:
        policy = AORePlanBranch(max_steps=MAX_STEPS, seed=0)

        def step(obs):
            policy.propose([obs])
            policy.commit([False])

        reset = policy.reset
        get_wrapper = lambda: policy._wrapper

    # Goal observations are sufficient to acquire walls, without any probe.
    step(observation(HISTORY, HISTORY))
    old = get_wrapper().static_astar
    old.observe([observation()])
    assert old.get_action(0, observation()) == 2
    reset()
    current = get_wrapper().static_astar
    assert current is not old
    assert current._planners is None
    step(observation())
    assert current.get_action(0, observation()) == 1
    # Reset created new storage rather than mutating another episode's object.
    assert old.get_action(0, observation()) == 2


def test_static_query_ignores_current_agent_positions_and_failure_cache(backend):
    obs = observation(target=(0, 2), walls=(), occupied=((0, 1),))
    check = new_check()
    check.observe([obs])
    local = check._planners[0]
    # Deliberately seed a failed eastward proposal in this private planner.
    local.update_obstacles([], [], (-5, -5))
    local.plan_path(CURRENT, (0, 2))
    assert first_action(local) == 4
    local.observe_position(CURRENT)
    # Also add transient occupancy without relying on backend-private fields.
    local.update_obstacles([], [(4, 5), (6, 5)], (-5, -5))
    local.plan_path(CURRENT, (0, 2))
    assert first_action(local) != 4
    assert check.get_action(0, obs) == 4


def test_static_query_does_not_change_dynamic_failure_feedback(backend):
    obs = observation(target=(0, 2), walls=())
    base = ao.AORePlanBase(max_steps=MAX_STEPS, seed=0)
    assert base.act([obs]) == [4]
    base.commit_proposals([True])
    assert base.act([obs])[0] not in (None, 4)
    base.commit_proposals([False])
    check = new_check()
    check.observe([obs])
    assert check._planners[0] is not base.planner[0]
    assert check.get_action(0, obs) == 4
    assert base.act([obs])[0] not in (None, 4)


def test_static_query_leaves_no_pending_execution_feedback(backend):
    obs = observation(target=(0, 2), walls=())
    check = new_check()
    check.observe([obs])
    assert check.get_action(0, obs) == 4
    local = check._planners[0]
    # A static query is not a submitted dynamic proposal. If its desired
    # position survived, this non-movement would incorrectly blacklist east.
    local.observe_position(CURRENT)
    local.update_obstacles([], [], (-5, -5))
    local.plan_path(CURRENT, (0, 2))
    assert first_action(local) == 4
