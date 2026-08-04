import importlib
import numpy as np

importlib.import_module("cppimport.import_hook")

from pogema import GridConfig


INF = 1_000_000_000

from planning.planner import planner



class AORePlanBase:

    """C++ replanning core used by AO-RePlan."""


    def __init__(self, use_best_move: bool = True, max_steps: int = INF, seed=None,

                 ignore_other_agents=False):

        self.use_best_move = use_best_move

        gc: GridConfig = GridConfig()

        self.actions = {tuple(gc.MOVES[i]): i for i in range(len(gc.MOVES))}

        self.steps = 0


        self.planner = None

        self.max_steps = max_steps

        self.rnd = np.random.default_rng(seed)

        self.ignore_other_agents = ignore_other_agents


    def act(self, obs, skip_agents=None):

        num_agents = len(obs)
        execution_synchronized = skip_agents is not None
        if execution_synchronized and len(skip_agents) != num_agents:
            raise ValueError(
                "skip_agents and observations must have equal sizes."
            )

        if self.planner is None:

            self.planner = [planner(self.max_steps) for _ in range(num_agents)]
        elif len(self.planner) != num_agents:
            raise ValueError(
                "AO-RePlan planner count differs from observations. "
                "Call after_reset() before changing the agent count."
            )

        obs_radius = len(obs[0]['obstacles']) // 2

        actions = []


        for agent_idx in range(num_agents):

            if (
                not execution_synchronized
                and obs[agent_idx]['xy'] == obs[agent_idx]['target_xy']
            ):

                actions.append(None)

                continue


            obstacles = np.transpose(np.nonzero(obs[agent_idx]['obstacles']))

            if self.ignore_other_agents:

                # The pybind planner expects a sequence of coordinates.  An
                # empty sequence expresses the intended static-only query;
                # ``None`` is rejected before A* can run.
                other_agents = []

            else:

                other_agents = np.transpose(np.nonzero(obs[agent_idx]['agents']))

            self.planner[agent_idx].update_obstacles(

                obstacles,

                other_agents,

                (obs[agent_idx]['xy'][0] - obs_radius, obs[agent_idx]['xy'][1] - obs_radius),

            )

            if execution_synchronized:
                # Consume feedback only for a proposal that the switcher
                # explicitly committed on the preceding step. Cancelled
                # proposals have no pending desired position and are a no-op.
                self.planner[agent_idx].observe_position(
                    obs[agent_idx]['xy']
                )

            if (
                obs[agent_idx]['xy'] == obs[agent_idx]['target_xy']
                or (
                    execution_synchronized
                    and bool(skip_agents[agent_idx])
                )
            ):

                actions.append(None)

                continue


            if execution_synchronized:
                self.planner[agent_idx].plan_path(
                    obs[agent_idx]['xy'],
                    obs[agent_idx]['target_xy'],
                )
            else:
                # Preserve the independent AO-RePlan baseline's historical
                # update_path semantics exactly.
                self.planner[agent_idx].update_path(
                    obs[agent_idx]['xy'],
                    obs[agent_idx]['target_xy'],
                )

            path = self.planner[agent_idx].get_next_node(self.use_best_move)

            if path is not None and path[1][0] < INF:

                delta = (path[1][0] - path[0][0], path[1][1] - path[0][1])

                actions.append(self.actions[delta])

            else:

                actions.append(None)


        self.steps += 1

        return actions


    def commit_proposals(self, executed_mask):
        """Keep feedback only for AO proposals that were really executed."""

        if self.planner is None:
            raise RuntimeError(
                "AO-RePlan must act before proposals can be committed."
            )
        if len(executed_mask) != len(self.planner):
            raise ValueError(
                "executed_mask and AO-RePlan planners must have equal sizes."
            )
        for agent_idx, executed in enumerate(executed_mask):
            if not bool(executed):
                self.planner[agent_idx].cancel_desired()


    def get_path(self):

        results = []

        for idx in range(len(self.planner)):

            results.append(self.planner[idx].get_path(use_best_node=False))

        return results



class FixNonesWrapper:

    def __init__(self, agent):

        self.agent = agent

        self.rnd = self.agent.rnd


    def act(self, obs, skip_agents=None):

        actions = self.agent.act(obs, skip_agents=skip_agents)

        for idx in range(len(actions)):

            if skip_agents is not None and bool(skip_agents[idx]):

                continue

            if actions[idx] is None:

                actions[idx] = 0

        return actions



class NoPathSoRandomOrStayWrapper:

    def __init__(self, agent):

        self.agent = agent

        self.rnd = self.agent.rnd


    def act(self, obs, skip_agents=None):

        actions = self.agent.act(obs, skip_agents=skip_agents)

        for idx in range(len(actions)):

            if skip_agents is not None and bool(skip_agents[idx]):

                continue

            if actions[idx] is None:

                if self.rnd.random() <= 0.5:

                    actions[idx] = 0

                else:

                    actions[idx] = self.get_random_move(obs, idx)

        return actions


    def get_random_move(self, obs, agent_id):

        deltas = GridConfig().MOVES

        actions = [1, 2, 3, 4]


        self.agent.rnd.shuffle(actions)

        for idx in actions:

            i = len(obs[agent_id]['obstacles']) // 2 + deltas[idx][0]

            j = len(obs[agent_id]['obstacles']) // 2 + deltas[idx][1]

            if obs[agent_id]['obstacles'][i][j] == 0:

                return idx

        return 0



class IgnoreAgentsProbe:

    """Run a one-shot A* query against the current static observation."""


    def __init__(self, use_best_move: bool = True, max_steps: int = INF):

        self.use_best_move = use_best_move

        self.max_steps = max_steps

        gc = GridConfig()

        self.actions = {tuple(gc.MOVES[i]): i for i in range(len(gc.MOVES))}

    @staticmethod
    def _complete_path(raw_path, start, target):

        """Return a validated cardinal path, or ``None`` on probe failure."""

        try:

            path = tuple(

                (int(node[0]), int(node[1]))

                for node in raw_path

            )

        except (IndexError, TypeError, ValueError):

            return None

        if not path or path[0] != start or path[-1] != target:

            return None

        if any(

            abs(first[0] - second[0]) + abs(first[1] - second[1]) != 1

            for first, second in zip(path, path[1:])

        ):

            return None

        return path

    def get_path(self, obs_i, agent_index=0):

        """Return a complete local static Probe path without retaining state.

        The dynamic-agent channel is ignored and a fresh planner is used for
        every call. Callers may therefore test a tentative virtual obstacle
        without contaminating AO-RePlan's accumulated probe memory.
        """

        del agent_index

        try:

            start = tuple(int(value) for value in obs_i['xy'])

            target = tuple(int(value) for value in obs_i['target_xy'])

            obstacle_map = np.asarray(obs_i['obstacles'])

            if (

                len(start) != 2

                or len(target) != 2

                or obstacle_map.ndim != 2

                or obstacle_map.shape[0] != obstacle_map.shape[1]

                or obstacle_map.shape[0] == 0

                or obstacle_map.shape[0] % 2 == 0

            ):

                return None

        except (KeyError, TypeError, ValueError):

            return None

        if start == target:

            return (start,)

        local_planner = planner(self.max_steps)

        obs_radius = obstacle_map.shape[0] // 2

        obstacles = np.transpose(np.nonzero(obstacle_map))

        try:

            local_planner.update_obstacles(

                obstacles,

                [],

                (start[0] - obs_radius, start[1] - obs_radius),

            )

            local_planner.plan_path(start, target)

            raw_path = local_planner.get_path(False)

        except Exception:

            return None

        finally:

            try:

                local_planner.cancel_desired()

            except Exception:

                pass

        return self._complete_path(raw_path, start, target)

    def get_action(self, obs_i, agent_index=0):

        local_planner = planner(self.max_steps)

        obs_radius = len(obs_i['obstacles']) // 2

        obstacles = np.transpose(np.nonzero(obs_i['obstacles']))

        local_planner.update_obstacles(

            obstacles,

            [],

            (obs_i['xy'][0] - obs_radius, obs_i['xy'][1] - obs_radius),

        )

        local_planner.update_path(tuple(obs_i['xy']), tuple(obs_i['target_xy']))

        path = local_planner.get_next_node(self.use_best_move)

        if path is not None and path[1][0] < INF:

            delta = (path[1][0] - path[0][0], path[1][1] - path[0][1])

            return self.actions[delta]

        return None


class StaticMapMemoryProbe(IgnoreAgentsProbe):

    """Run A* against each agent's accumulated static-map observations."""


    def __init__(self, use_best_move: bool = True, max_steps: int = INF):

        super().__init__(use_best_move=use_best_move, max_steps=max_steps)

        self.planners = {}


    def get_action(self, obs_i, agent_index=0):

        local_planner = self.planners.get(agent_index)

        if local_planner is None:

            local_planner = planner(self.max_steps)

            self.planners[agent_index] = local_planner

        obs_radius = len(obs_i['obstacles']) // 2

        obstacles = np.transpose(np.nonzero(obs_i['obstacles']))

        local_planner.update_obstacles(

            obstacles,

            [],

            (obs_i['xy'][0] - obs_radius, obs_i['xy'][1] - obs_radius),

        )

        local_planner.update_static_path(

            tuple(obs_i['xy']),

            tuple(obs_i['target_xy']),

        )

        path = local_planner.get_next_node(self.use_best_move)

        if path is not None and path[1][0] < INF:

            delta = (path[1][0] - path[0][0], path[1][1] - path[0][1])

            return self.actions[delta]

        return None



class AORePlanWrapper:

    """Handle reversals caused only by other agents.

    AO-RePlan replaces a return move with the static probe action.
    External switchers may request a handoff instead of executing that decision.
    """


    def __init__(self, agent, use_best_move: bool = True, max_steps: int = INF,

                 handoff_on_reverse: bool = False):

        self.agent = agent

        self.rnd = agent.rnd

        self.probe = IgnoreAgentsProbe(

            use_best_move=use_best_move,

            max_steps=max_steps,

        )

        self.switch_probe = StaticMapMemoryProbe(

            use_best_move=use_best_move,

            max_steps=max_steps,

        )

        self.handoff_on_reverse = handoff_on_reverse

        self.moves = tuple(tuple(move) for move in GridConfig().MOVES)

        self.position_history = None

        self.last_target = None

        self.last_teammate_blocked_mask = None

        self.last_planned_mask = None
        self.last_probe_replacement_mask = None
        self.last_raw_dynamic_actions = None
        self.last_static_probe_actions = None
        self.last_forward_clear_mask = None


    def _ensure_state(self, n):

        if self.position_history is None:

            self.position_history = [[] for _ in range(n)]

            self.last_target = [None] * n


    def _update_position_history(self, index, observation):

        position = tuple(int(value) for value in observation["xy"])

        history = self.position_history[index]

        if not history or position != history[-1]:

            history.append(position)

            if len(history) > 2:

                del history[:-2]

        return position, history


    def _returns_to_previous_position(self, position, action, history):

        if action in (None, 0) or len(history) < 2:

            return False

        dx, dy = self.moves[int(action)]

        next_position = (position[0] + dx, position[1] + dy)

        return next_position == history[-2]


    def act(self, obs, skip_agents=None):

        actions = self.agent.act(obs, skip_agents=skip_agents)

        self._ensure_state(len(actions))

        self.last_teammate_blocked_mask = [False] * len(actions)
        self.last_probe_replacement_mask = [False] * len(actions)
        self.last_raw_dynamic_actions = list(actions)
        self.last_static_probe_actions = [None] * len(actions)
        self.last_forward_clear_mask = [False] * len(actions)

        # Record provenance before an outer no-path wrapper can replace None
        # with a random move or wait. A valid teammate probe replacement is
        # still an AO-RePlan decision and therefore remains marked as planned.
        self.last_planned_mask = [action is not None for action in actions]


        for i, action in enumerate(self.last_raw_dynamic_actions):

            cur_target = tuple(obs[i]['target_xy'])

            if self.last_target[i] != cur_target:

                self.position_history[i].clear()

                self.last_target[i] = cur_target

            position, history = self._update_position_history(i, obs[i])

            if skip_agents is not None and bool(skip_agents[i]):

                continue


            if action in (None, 0):

                continue

            returns_to_previous = self._returns_to_previous_position(

                position, action, history

            )

            # The independent AO-RePlan baseline historically probes only
            # reversals. Shadow/handoff mode additionally records agreement
            # on ordinary forward proposals for the hybrid controller.
            if not self.handoff_on_reverse and not returns_to_previous:

                continue

            probe_action = self.probe.get_action(obs[i], i)

            self.last_static_probe_actions[i] = probe_action

            self.last_forward_clear_mask[i] = bool(

                probe_action not in (None, 0)

                and int(action) == int(probe_action)

                and not returns_to_previous

            )

            if returns_to_previous:

                if (

                    probe_action is not None

                    and not self._returns_to_previous_position(

                        position, probe_action, history

                    )

                ):

                    if self.handoff_on_reverse:

                        self.last_teammate_blocked_mask[i] = True

                        actions[i] = None

                        self.last_planned_mask[i] = False

                        continue

                    actions[i] = probe_action

                    self.last_planned_mask[i] = True
                    self.last_probe_replacement_mask[i] = True

                    continue


        return actions
