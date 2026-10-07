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

    def observe(self, observations):
        pass

    def get_action(self, index, observation):
        del index, observation
        self.calls += 1
        return self.action


def observation(
    position=(5, 5), target=(5, 8), *, agent_actions=(), obstacle_actions=()
):
    obstacles = np.zeros((11, 11), dtype=np.int8)
    agents = np.zeros((11, 11), dtype=np.int8)
    moves = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
    for action in agent_actions:
        di, dj = moves[action]
        agents[5 + di, 5 + dj] = 1
    for action in obstacle_actions:
        di, dj = moves[action]
        obstacles[5 + di, 5 + dj] = 1
    return [
        {
            "obstacles": obstacles,
            "agents": agents,
            "xy": position,
            "target_xy": target,
        }
    ]


def test_production_api_has_no_retired_boolean_switches():
    retired = {
        "use_best_move",
        "no_path_random",
        "fix_nones",
        "handoff_on_reverse",
        "max_probe_wait_steps",
    }
    assert retired.isdisjoint(AORePlanConfig.__fields__)
    for constructor in (
        AORePlanBase,
        StaticAStarCheck,
        AORePlanWrapper,
        AORePlanBranch,
    ):
        for parameter in inspect.signature(constructor).parameters.values():
            assert parameter.annotation is not bool
            assert not isinstance(parameter.default, bool)


def test_dynamic_planner_requests_exact_path_then_best_move():
    calls = []
    fake = SimpleNamespace(
        get_next_node=lambda value: calls.append(value) or None,
        update_obstacles=lambda *_args: None,
        proposal_failed=lambda *_args: False,
        observe_position=lambda *_args: None,
        plan_path=lambda *_args: None,
        release_failed_actions=lambda: None,
    )
    base = AORePlanBase(seed=0)
    base.planner = [fake]
    assert base.act(observation()) == [None]
    assert calls == [True]


def test_no_path_uses_original_wait_or_obstacle_only_random_fallback():
    wait_wrapper = AORePlanWrapper(ActionSequence([None], coin=0.5))
    wait_wrapper.static_astar = FixedStaticAStar(4)
    assert wait_wrapper.act(observation()) == [0]
    assert wait_wrapper.static_astar.calls == 0

    random_wrapper = AORePlanWrapper(ActionSequence([None], coin=0.500001))
    random_wrapper.static_astar = FixedStaticAStar(4)
    assert random_wrapper.act(observation(agent_actions=(1,))) == [1]
    assert random_wrapper.static_astar.calls == 0


def test_reverse_uses_previous_timestep_and_wait_cannot_repeat():
    wrapper = AORePlanWrapper(ActionSequence([1, 2, 2, 2]))
    static_astar = FixedStaticAStar(None)
    wrapper.static_astar = static_astar
    assert wrapper.act(observation(position=(5, 5))) == [1]
    assert wrapper.act(observation(position=(4, 5))) == [0]
    assert static_astar.calls == 1
    assert wrapper.act(observation(position=(4, 5))) == [2]
    assert static_astar.calls == 1
    assert wrapper.act(observation(position=(4, 5))) == [2]
    assert static_astar.calls == 1


def test_static_astar_first_step_is_locally_collision_checked():
    wrapper = AORePlanWrapper(ActionSequence([1, 2]))
    wrapper.static_astar = FixedStaticAStar(4)
    assert wrapper.act(observation(position=(5, 5))) == [1]
    assert wrapper.act(observation(position=(4, 5), agent_actions=(4,))) == [0]
    assert wrapper.static_astar.calls == 1
    assert wrapper.last_dynamic_override_mask == [True]


def test_static_astar_same_reverse_keeps_dynamic_feedback():
    wrapper = AORePlanWrapper(ActionSequence([1, 2]))
    wrapper.static_astar = FixedStaticAStar(2)
    assert wrapper.act(observation(position=(5, 5))) == [1]
    assert wrapper.act(observation(position=(4, 5))) == [2]
    assert wrapper.static_astar.calls == 1
    assert wrapper.last_dynamic_override_mask == [False]


def test_goal_returns_explicit_wait_not_no_path():
    wrapper = AORePlanWrapper(ActionSequence([None]))
    wrapper.static_astar = FixedStaticAStar(4)
    assert wrapper.act(observation(position=(5, 5), target=(5, 5))) == [0]
    assert wrapper.static_astar.calls == 0


def test_standalone_does_not_commit_replaced_dynamic_actions():
    commits = []
    algorithm = AORePlan(AORePlanConfig())
    base = ActionSequence([None], coin=0.5)
    base.commit_proposals = lambda mask: commits.append(mask)
    algorithm._ao_wrapper = AORePlanWrapper(base)
    algorithm._ao_wrapper.static_astar = FixedStaticAStar(4)
    assert algorithm.act(observation()) == [0]
    assert commits == [[False]]


def test_wrapper_commits_only_executed_unreplaced_proposals():
    commits = []
    wrapper = AORePlanWrapper(
        SimpleNamespace(
            rnd=FixedRng(0.5),
            commit_proposals=lambda mask: commits.append(mask),
        )
    )
    wrapper.last_dynamic_override_mask = [False, True, False, True]
    wrapper.commit_proposals([True, True, False, False])
    assert commits == [[True, False, False, False]]
