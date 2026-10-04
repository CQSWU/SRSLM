import importlib
import shutil
import sys

import numpy as np
from pogema import GridConfig

if sys.platform == "win32":
    from planning.python_planner import planner
else:
    cppimport = importlib.import_module("cppimport")
    if shutil.which("g++") is None:
        cppimport.settings["release_mode"] = True
    importlib.import_module("cppimport.import_hook")
    from planning.planner import planner

    if not hasattr(planner, "proposal_failed"):
        raise ImportError(
            "The installed planner extension predates randomized failure caching. "
            "Rebuild planning/planner.cpp with cppimport before running AORePlan."
        )

INF = 1_000_000_000
FAILURE_CACHE_PROBABILITY = 0.5
_FAILURE_CACHE_SEED_SALT = 0xFA11CA
_MOVES = tuple(tuple(move) for move in GridConfig().MOVES)
_ACTION_BY_DELTA = {move: index for index, move in enumerate(_MOVES)}


class AORePlanBase:
    def __init__(self, max_steps: int = INF, seed=None):
        self.planner = None
        self.max_steps = int(max_steps)
        self.rnd = np.random.default_rng(seed)
        cache_seed = (
            None
            if seed is None
            else np.random.SeedSequence([int(seed), _FAILURE_CACHE_SEED_SALT])
        )

        self.failure_cache_rnd = np.random.default_rng(cache_seed)

    def act(self, observations, skip_agents=None):
        count = len(observations)
        if skip_agents is None:
            skip = [False] * count
        else:
            if len(skip_agents) != count:
                raise ValueError("skip_agents and observations must have equal sizes.")
            skip = [bool(value) for value in skip_agents]

        if self.planner is None:
            self.planner = [planner(self.max_steps) for _ in range(count)]
        elif len(self.planner) != count:
            raise ValueError(
                "AORePlan planner count differs from observations. "
                "Call after_reset() before changing the agent count."
            )

        actions = []
        for index, observation in enumerate(observations):
            obstacle_map = np.asarray(observation["obstacles"])
            radius = obstacle_map.shape[0] // 2
            position = tuple(int(value) for value in observation["xy"])
            target = tuple(int(value) for value in observation["target_xy"])
            local_planner = self.planner[index]
            local_planner.update_obstacles(
                np.transpose(np.nonzero(obstacle_map)),
                np.transpose(np.nonzero(observation["agents"])),
                (
                    position[0] - radius,
                    position[1] - radius,
                ),
            )

            cache_failed_action = True
            if local_planner.proposal_failed(position):
                cache_failed_action = bool(
                    self.failure_cache_rnd.random() < FAILURE_CACHE_PROBABILITY
                )
            local_planner.observe_position(position, cache_failed_action)

            if position == target or skip[index]:
                actions.append(None)
                continue

            local_planner.plan_path(position, target)
            path = self._get_next_node(local_planner)
            if path is None or path[1][0] >= INF:
                local_planner.release_failed_actions()
                actions.append(None)
                continue
            delta = (
                path[1][0] - path[0][0],
                path[1][1] - path[0][1],
            )
            actions.append(_ACTION_BY_DELTA[delta])
        return actions

    @staticmethod
    def _get_next_node(local_planner):

        return local_planner.get_next_node(True)

    def commit_proposals(self, executed_mask):

        if self.planner is None:
            raise RuntimeError("AORePlan must act before proposals can be committed.")
        if len(executed_mask) != len(self.planner):
            raise ValueError(
                "executed_mask and AORePlan planners must have equal sizes."
            )
        for index, executed in enumerate(executed_mask):
            if not bool(executed):
                self.planner[index].cancel_desired()


def _local_cell_is_free(observation, action, moves=_MOVES):

    if action in (None, 0):
        return False
    try:
        obstacles = np.asarray(observation["obstacles"])
        agents = np.asarray(observation["agents"])
        center_i = obstacles.shape[0] // 2
        center_j = obstacles.shape[1] // 2
        di, dj = moves[int(action)]
        i = center_i + int(di)
        j = center_j + int(dj)
        return bool(
            0 <= i < obstacles.shape[0]
            and 0 <= j < obstacles.shape[1]
            and obstacles[i, j] == 0
            and agents[i, j] == 0
        )
    except (IndexError, KeyError, TypeError, ValueError):
        return False


def original_random_or_stay(observation, rnd, moves=_MOVES):

    if rnd.random() <= 0.5:
        return 0

    actions = [1, 2, 3, 4]
    rnd.shuffle(actions)
    obstacles = np.asarray(observation["obstacles"])
    center_i = obstacles.shape[0] // 2
    center_j = obstacles.shape[1] // 2
    for action in actions:
        di, dj = moves[int(action)]
        if obstacles[center_i + int(di), center_j + int(dj)] == 0:
            return action
    return 0


class StaticAStarCheck:
    def __init__(self, max_steps: int = INF):
        self.max_steps = int(max_steps)
        self._planners = None

    def observe(self, observations):
        count = len(observations)
        if self._planners is None:
            self._planners = [planner(self.max_steps) for _ in range(count)]
        elif len(self._planners) != count:
            raise ValueError("Static A* agent count changed without reset.")
        for local_planner, observation in zip(self._planners, observations):
            obstacle_map = np.asarray(observation["obstacles"])
            position = tuple(int(value) for value in observation["xy"])
            local_planner.update_obstacles(
                np.transpose(np.nonzero(obstacle_map)),
                [],
                (
                    position[0] - obstacle_map.shape[0] // 2,
                    position[1] - obstacle_map.shape[1] // 2,
                ),
            )

    def get_action(self, index, observation):
        if self._planners is None:
            raise RuntimeError("Observe the agent batch before querying static A*.")
        local_planner = self._planners[index]

        local_planner.update_static_path(
            tuple(observation["xy"]),
            tuple(observation["target_xy"]),
        )
        path = local_planner.get_next_node(False)
        local_planner.cancel_desired()
        if path is None or path[1][0] >= INF:
            return None
        delta = (
            path[1][0] - path[0][0],
            path[1][1] - path[0][1],
        )
        return _ACTION_BY_DELTA[delta]


class AORePlanWrapper:
    def __init__(
        self,
        agent,
        max_steps: int = INF,
    ):
        self.agent = agent
        self.rnd = agent.rnd
        self.static_astar = StaticAStarCheck(max_steps=max_steps)
        self.moves = _MOVES

        self.previous_position = None
        self.last_target = None
        self.last_raw_dynamic_actions = None
        self.last_static_astar_invoked_mask = None
        self.last_no_path_fallback_mask = None
        self.last_reverse_mask = None
        self.last_dynamic_override_mask = None

    def _ensure_state(self, count):
        if self.previous_position is None:
            self.previous_position = [None] * count
            self.last_target = [None] * count
            return
        if len(self.previous_position) != count:
            raise ValueError("AORePlan agent count changed without reset.")

    def _static_astar_action(self, index, observation):
        raw_action = self.static_astar.get_action(index, observation)
        self.last_static_astar_invoked_mask[index] = True
        conflict = bool(
            raw_action not in (None, 0)
            and not _local_cell_is_free(
                observation,
                raw_action,
                moves=self.moves,
            )
        )
        if raw_action is None or conflict:
            return 0
        return int(raw_action)

    def _returns_to_previous_position(self, position, action, previous):

        if action in (None, 0) or previous is None:
            return False
        dx, dy = self.moves[int(action)]
        return (
            position[0] + dx,
            position[1] + dy,
        ) == previous

    def _reset_diagnostics(self, actions):
        count = len(actions)
        self.last_raw_dynamic_actions = list(actions)
        self.last_static_astar_invoked_mask = [False] * count
        self.last_no_path_fallback_mask = [False] * count
        self.last_reverse_mask = [False] * count
        self.last_dynamic_override_mask = [False] * count

    def act(self, observations, skip_agents=None):
        actions = list(self.agent.act(observations, skip_agents=skip_agents))
        self._ensure_state(len(actions))

        self.static_astar.observe(observations)
        self._reset_diagnostics(actions)

        for index, raw_action in enumerate(self.last_raw_dynamic_actions):
            observation = observations[index]
            target = tuple(int(value) for value in observation["target_xy"])
            if self.last_target[index] != target:
                self.previous_position[index] = None
                self.last_target[index] = target

            position = tuple(int(value) for value in observation["xy"])
            previous = self.previous_position[index]

            self.previous_position[index] = position

            if skip_agents is not None and bool(skip_agents[index]):
                continue

            if position == target:
                actions[index] = 0
                continue

            if raw_action is None:
                self.last_no_path_fallback_mask[index] = True
                actions[index] = original_random_or_stay(
                    observation,
                    self.rnd,
                    moves=self.moves,
                )
                self.last_dynamic_override_mask[index] = True
                continue

            if raw_action == 0:
                continue

            reverse = self._returns_to_previous_position(
                position,
                raw_action,
                previous,
            )
            self.last_reverse_mask[index] = reverse
            if not reverse:
                continue

            static_action = self._static_astar_action(index, observation)
            if not self._returns_to_previous_position(
                position,
                static_action,
                previous,
            ):
                actions[index] = int(static_action)
                self.last_dynamic_override_mask[index] = True

        return actions
