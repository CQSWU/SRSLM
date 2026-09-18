"""Failure-cache lifetime regressions for both planner backends."""
import sys

import numpy as np
import pytest

import planning.ao_replan_algo as ao
from planning.python_planner import planner as PythonPlanner


MOVES = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
BACKENDS = [("python", PythonPlanner)]
if sys.platform != "win32":
    from planning.planner import planner as NativePlanner
    BACKENDS.append(("native", NativePlanner))


def observation(target=(0, 2), *, occupied=(), walls=()):
    obstacles = np.zeros((11, 11), dtype=np.int8)
    agents = np.zeros_like(obstacles)
    for action in walls:
        di, dj = MOVES[action]
        obstacles[5 + di, 5 + dj] = 1
    for action in occupied:
        di, dj = MOVES[action]
        agents[5 + di, 5 + dj] = 1
    return [{"xy": (0, 0), "target_xy": target,
             "obstacles": obstacles, "agents": agents}]


@pytest.fixture(params=BACKENDS, ids=[name for name, _ in BACKENDS])
def base(request, monkeypatch):
    monkeypatch.setattr(ao, "planner", request.param[1])
    return ao.AORePlanBase(max_steps=1000, seed=0)


def exhaust_four_directions(base):
    # All proposals are submitted, but the observed position never changes.
    for action in range(1, 5):
        di, dj = MOVES[action]
        assert base.act(observation((2 * di, 2 * dj))) == [action]
        base.commit_proposals([True])
    assert base.act(observation()) == [None]
    base.commit_proposals([False])


def test_exhausted_cache_returns_none_now_but_recovers_next_query(base):
    queries = []
    original = base._get_next_node

    def counted(local_planner):
        queries.append(1)
        return original(local_planner)

    base._get_next_node = counted
    exhaust_four_directions(base)
    # No hidden retry or extra search within the no-action decision.
    assert len(queries) == 5
    assert base.act(observation()) == [4]
    assert len(queries) == 6


def test_exhaustion_does_not_remove_current_agents(base):
    exhaust_four_directions(base)
    assert base.act(observation(occupied=(1, 2, 3, 4))) == [None]
    base.commit_proposals([False])
    assert base.act(observation(occupied=(1, 2, 3, 4))) == [None]
    base.commit_proposals([False])
    assert base.act(observation()) == [4]


def test_exhaustion_does_not_remove_accumulated_static_obstacles(base):
    exhaust_four_directions(base)
    assert base.act(observation(walls=(1, 2, 3, 4))) == [None]
    base.commit_proposals([False])
    # Even if omitted from a later crop, learned static walls remain.
    assert base.act(observation()) == [None]


def test_real_failure_survives_skipped_queries(base):
    assert base.act(observation()) == [4]
    base.commit_proposals([True])
    for _ in range(2):
        assert base.act(observation(), skip_agents=[True]) == [None]
        base.commit_proposals([False])
    candidate = base.act(observation())[0]
    assert candidate is not None and candidate != 4


def test_at_goal_none_is_not_treated_as_exhausted_search(base):
    assert base.act(observation()) == [4]
    base.commit_proposals([True])
    assert base.act(observation((0, 0))) == [None]
    base.commit_proposals([False])
    candidate = base.act(observation())[0]
    assert candidate is not None and candidate != 4


def test_cancelled_valid_proposal_keeps_prior_failure_but_not_fake_failure(base):
    assert base.act(observation()) == [4]
    base.commit_proposals([True])
    assert base.act(observation((2, 0))) == [2]
    # This viable proposal was replaced by static A* / the learning branch.
    base.commit_proposals([False])
    assert base.act(observation((2, 0))) == [2]
    base.commit_proposals([False])
    candidate = base.act(observation())[0]
    assert candidate is not None and candidate != 4


def test_release_is_per_agent_not_shared_between_planners(base):
    for action in range(1, 5):
        di, dj = MOVES[action]
        batch = observation((2 * di, 2 * dj)) + observation()
        candidates = base.act(batch)
        assert candidates[0] == action
        # Agent1's first submitted right fails; later proposals are cancelled.
        if action == 1:
            base.commit_proposals([True, True])
        else:
            base.commit_proposals([True, False])
    # Agent0 exhausts its four failed exits; agent1's right failure remains.
    candidates = base.act(observation() + observation())
    assert candidates[0] is None
    assert candidates[1] is not None and candidates[1] != 4
    base.commit_proposals([False, False])
    candidates = base.act(observation() + observation())
    assert candidates[0] == 4
    assert candidates[1] is not None and candidates[1] != 4

