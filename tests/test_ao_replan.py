import inspect
from types import SimpleNamespace

import numpy as np

from agents.ao_replan import AORePlan, AORePlanConfig
from planning.ao_replan_algo import AORePlanBase, AORePlanWrapper, StaticAStarCheck
from planning.aoreplan_branch import AORePlanBranch


class FixedRng:
    def __init__(self, coin):
        self.coin = coin

    def random(self):
        return self.coin

    def shuffle(self, actions):
        return None


class ActionSequence:
    def __init__(self, actions, coin=0.5):
        self.actions = iter(actions)
        self.rnd = FixedRng(coin)

    def act(self, observations, skip_agents=None):
        del observations, skip_agents
        return [next(self.actions)]


class FixedStaticAStar:
    def __init__(self, action):
        self.action = action
        self.calls = 0

    def get_action(self, observation):
        del observation
        self.calls += 1
        return self.action


def observation(position=(5, 5), target=(5, 8), *, agent_actions=(), obstacle_actions=()):
    obstacles = np.zeros((11, 11), dtype=np.int8)
    agents = np.zeros((11, 11), dtype=np.int8)
    moves = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
    for action in agent_actions:
        di, dj = moves[action]
        agents[5 + di, 5 + dj] = 1
    for action in obstacle_actions:
        di, dj = moves[action]
        obstacles[5 + di, 5 + dj] = 1
    return [{
        "obstacles": obstacles,
        "agents": agents,
        "xy": position,
        "target_xy": target,
    }]


def test_production_api_has_no_retired_boolean_switches():
    retired = {
        "use_best_move", "no_path_random", "fix_nones",
        "handoff_on_reverse", "max_probe_wait_steps",
    }
    assert retired.isdisjoint(AORePlanConfig.__fields__)
    for constructor in (AORePlanBase, StaticAStarCheck, AORePlanWrapper, AORePlanBranch):
        for parameter in inspect.signature(constructor).parameters.values():
            assert parameter.annotation is not bool
            assert not isinstance(parameter.default, bool)


def test_dynamic_planner_requests_exact_path_then_best_move():
    calls = []
    fake = SimpleNamespace(get_next_node=lambda value: calls.append(value) or None)
    assert AORePlanBase._get_next_node(fake) is None
    assert calls == [True]


def test_no_path_uses_original_wait_or_obstacle_only_random_fallback():
    wait_wrapper = AORePlanWrapper(ActionSequence([None], coin=0.5))
    wait_wrapper.static_astar = FixedStaticAStar(4)
    assert wait_wrapper.act(observation()) == [0]
    assert wait_wrapper.last_no_path_fallback_mask == [True]
    assert wait_wrapper.last_static_astar_invoked_mask == [False]

    random_wrapper = AORePlanWrapper(ActionSequence([None], coin=0.500001))
    random_wrapper.static_astar = FixedStaticAStar(4)
    assert random_wrapper.act(observation(agent_actions=(1,))) == [1]
    assert random_wrapper.last_static_astar_invoked_mask == [False]


def test_reverse_uses_previous_timestep_and_wait_cannot_repeat():
    wrapper = AORePlanWrapper(ActionSequence([1, 2, 2, 2]))
    static_astar = FixedStaticAStar(None)
    wrapper.static_astar = static_astar
    assert wrapper.act(observation(position=(5, 5))) == [1]
    assert wrapper.act(observation(position=(4, 5))) == [0]
    assert wrapper.last_reverse_mask == [True]
    assert wrapper.act(observation(position=(4, 5))) == [2]
    assert wrapper.last_reverse_mask == [False]
    assert wrapper.act(observation(position=(4, 5))) == [2]
    assert static_astar.calls == 1


def test_static_astar_first_step_is_locally_collision_checked():
    wrapper = AORePlanWrapper(ActionSequence([1, 2]))
    wrapper.static_astar = FixedStaticAStar(4)
    assert wrapper.act(observation(position=(5, 5))) == [1]
    assert wrapper.act(observation(position=(4, 5), agent_actions=(4,))) == [0]
    assert wrapper.last_static_astar_invoked_mask == [True]
    assert wrapper.last_dynamic_override_mask == [True]


def test_static_astar_same_reverse_keeps_dynamic_feedback():
    wrapper = AORePlanWrapper(ActionSequence([1, 2]))
    wrapper.static_astar = FixedStaticAStar(2)
    assert wrapper.act(observation(position=(5, 5))) == [1]
    assert wrapper.act(observation(position=(4, 5))) == [2]
    assert wrapper.last_static_astar_invoked_mask == [True]
    assert wrapper.last_dynamic_override_mask == [False]


def test_goal_returns_explicit_wait_not_no_path():
    wrapper = AORePlanWrapper(ActionSequence([None]))
    wrapper.static_astar = FixedStaticAStar(4)
    assert wrapper.act(observation(position=(5, 5), target=(5, 5))) == [0]
    assert wrapper.last_planned_mask == [True]
    assert wrapper.last_no_path_fallback_mask == [False]


def test_standalone_metrics_use_new_names_and_denominators():
    algorithm = AORePlan(AORePlanConfig())
    algorithm.agent = ActionSequence([0])
    algorithm._base = SimpleNamespace(commit_proposals=lambda _mask: None)
    algorithm._ao_wrapper = SimpleNamespace(
        last_planned_mask=[True],
        last_dynamic_override_mask=[True],
        last_raw_dynamic_actions=[1],
        last_static_astar_invoked_mask=[True],
        last_no_path_fallback_mask=[False],
    )
    algorithm.act(observation())
    assert algorithm.static_astar_query_count == 1
    assert algorithm.static_astar_query_denominator == 1
    assert algorithm.static_astar_query_rate == 1.0
    assert algorithm.no_path_fallback_count == 0
    assert not hasattr(algorithm, "probe_invocation_rate")
