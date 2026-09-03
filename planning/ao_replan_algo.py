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


INF = 1_000_000_000


class AORePlanBase:
    """Dynamic C++ replanner with explicit proposal feedback."""

    def __init__(self, max_steps: int = INF, seed=None):
        grid_config = GridConfig()
        self.actions = {
            tuple(grid_config.MOVES[index]): index
            for index in range(len(grid_config.MOVES))
        }
        self.planner = None
        self.max_steps = int(max_steps)
        self.rnd = np.random.default_rng(seed)

    def act(self, observations, skip_agents=None):
        count = len(observations)
        if skip_agents is None:
            skip = [False] * count
        else:
            if len(skip_agents) != count:
                raise ValueError(
                    "skip_agents and observations must have equal sizes."
                )
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
            # Every current caller uses explicit proposal feedback. A cancelled
            # proposal has no desired position, so observing it is a no-op.
            local_planner.observe_position(position)

            if position == target or skip[index]:
                actions.append(None)
                continue

            local_planner.plan_path(position, target)
            path = self._get_next_node(local_planner)
            if path is None or path[1][0] >= INF:
                actions.append(None)
                continue
            delta = (
                path[1][0] - path[0][0],
                path[1][1] - path[0][1],
            )
            actions.append(self.actions[delta])
        return actions

    @staticmethod
    def _get_next_node(local_planner):
        """Return an exact-path step or the dynamic planner's BestMove."""

        return local_planner.get_next_node(True)

    def commit_proposals(self, executed_mask):
        """Retain feedback only for dynamic proposals that physically ran."""

        if self.planner is None:
            raise RuntimeError(
                "AORePlan must act before proposals can be committed."
            )
        if len(executed_mask) != len(self.planner):
            raise ValueError(
                "executed_mask and AORePlan planners must have equal sizes."
            )
        for index, executed in enumerate(executed_mask):
            if not bool(executed):
                self.planner[index].cancel_desired()


def _local_cell_is_free(observation, action, moves=None):
    """Return whether a movement target is free in the current local view."""

    if action in (None, 0):
        return False
    moves = moves or GridConfig().MOVES
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


def original_random_or_stay(observation, rnd, moves=None):
    """Mirror the original RePlan no-path fallback: 50% wait, 50% random.

    This reproduces ``NoPathSoRandomOrStayWrapper`` from the upstream RePlan
    exactly, including its obstacle-only screening: the sampled move may still
    enter a visible agent, which under ``block_both`` simply does not move the
    agent. AORePlan deliberately keeps this branch identical to the original
    so that the static A* check on the reverse branch is the single difference
    between the two planners.
    """

    moves = moves or GridConfig().MOVES
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
    """Run one fresh static-map A* query that ignores other agents."""

    def __init__(self, max_steps: int = INF):
        self.max_steps = int(max_steps)
        grid_config = GridConfig()
        self.actions = {
            tuple(grid_config.MOVES[index]): index
            for index in range(len(grid_config.MOVES))
        }

    def get_action(self, observation):
        local_planner = planner(self.max_steps)
        obstacle_map = np.asarray(observation["obstacles"])
        radius = obstacle_map.shape[0] // 2
        local_planner.update_obstacles(
            np.transpose(np.nonzero(obstacle_map)),
            [],
            (
                observation["xy"][0] - radius,
                observation["xy"][1] - radius,
            ),
        )
        local_planner.plan_path(
            tuple(observation["xy"]),
            tuple(observation["target_xy"]),
        )
        path = local_planner.get_next_node(False)
        if path is None or path[1][0] >= INF:
            return None
        delta = (
            path[1][0] - path[0][0],
            path[1][1] - path[0][1],
        )
        return self.actions[delta]


class AORePlanWrapper:
    """Apply the fixed AORePlan reverse/static-A* policy.

    AORePlan differs from the original RePlan in exactly one place.  When the
    dynamic planner fails to produce a move at all -- goal unreachable and
    BestMove unavailable -- both planners fall back to the original 50% wait /
    50% random rule.  When the dynamic proposal is a reverse, RePlan applies
    that same coin flip while AORePlan runs a static-map A* check and only
    waits if the check has no complete path or its first step is locally
    blocked.

    A reverse is measured against the position held at the *previous*
    timestep, recorded on every step including waits and blocked moves.  After
    a step in which the agent did not move, the previous position equals the
    current one, so no move action can be a reverse and the dynamic proposal is
    executed directly.  That breaks a repeated static-A* wait without a wait
    counter or a random escape action.
    """

    def __init__(
        self,
        agent,
        max_steps: int = INF,
    ):
        self.agent = agent
        self.rnd = agent.rnd
        self.static_astar = StaticAStarCheck(max_steps=max_steps)
        self.moves = tuple(tuple(move) for move in GridConfig().MOVES)

        self.previous_position = None
        self.last_target = None
        self.last_planned_mask = None
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
        raw_action = self.static_astar.get_action(observation)
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

    def _use_static_astar_action(self, index, actions, action):
        actions[index] = int(action)
        self.last_planned_mask[index] = True
        self.last_dynamic_override_mask[index] = True

    def _returns_to_previous_position(self, position, action, previous):
        """Return whether ``action`` moves back onto the previous position.

        ``previous`` is the position held one timestep ago, recorded whether or
        not the agent actually moved.  A wait therefore makes ``previous`` equal
        to ``position``, and no move action can then be a reverse.
        """

        if action in (None, 0) or previous is None:
            return False
        dx, dy = self.moves[int(action)]
        return (
            position[0] + dx,
            position[1] + dy,
        ) == previous

    def _reset_diagnostics(self, actions):
        count = len(actions)
        self.last_planned_mask = [action is not None for action in actions]
        self.last_raw_dynamic_actions = list(actions)
        self.last_static_astar_invoked_mask = [False] * count
        self.last_no_path_fallback_mask = [False] * count
        self.last_reverse_mask = [False] * count
        self.last_dynamic_override_mask = [False] * count

    def _resolve_dynamic_no_path(self, index, actions, observation):
        """Apply the original RePlan fallback after BestMove fails.

        No static A* runs here: goal unreachable with BestMove unavailable is the
        one branch AORePlan keeps identical to the original planner.
        ``last_no_path_fallback_mask`` marks that this fallback fired.
        """

        self.last_no_path_fallback_mask[index] = True
        actions[index] = original_random_or_stay(
            observation,
            self.rnd,
            moves=self.moves,
        )
        self.last_planned_mask[index] = True
        self.last_dynamic_override_mask[index] = True

    def _resolve_reverse(
        self,
        index,
        actions,
        observation,
        position,
        previous,
    ):
        static_action = self._static_astar_action(index, observation)
        # If static A* returns the same reverse, the map geometry itself
        # requires that move, so keep the dynamic candidate and its feedback.
        # A guarded wait is never a reverse, so it is covered by the same
        # predicate and replaces the dynamic proposal.
        if not self._returns_to_previous_position(
            position,
            static_action,
            previous,
        ):
            self._use_static_astar_action(index, actions, static_action)

    def act(self, observations, skip_agents=None):
        actions = list(self.agent.act(observations, skip_agents=skip_agents))
        self._ensure_state(len(actions))
        self._reset_diagnostics(actions)

        for index, raw_action in enumerate(self.last_raw_dynamic_actions):
            observation = observations[index]
            target = tuple(int(value) for value in observation["target_xy"])
            if self.last_target[index] != target:
                self.previous_position[index] = None
                self.last_target[index] = target

            position = tuple(int(value) for value in observation["xy"])
            previous = self.previous_position[index]
            # The previous position is recorded on every step, whether or not
            # the agent moved, and is only read before being overwritten here.
            self.previous_position[index] = position

            if skip_agents is not None and bool(skip_agents[index]):
                continue

            # Reaching a goal is not a dynamic no-path event. AORePlan still
            # returns a complete primitive action, namely wait.
            if position == target:
                actions[index] = 0
                self.last_planned_mask[index] = True
                continue

            if raw_action is None:
                self._resolve_dynamic_no_path(index, actions, observation)
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

            self._resolve_reverse(
                index,
                actions,
                observation,
                position,
                previous,
            )

        return actions
