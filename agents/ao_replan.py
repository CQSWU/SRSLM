from typing import Literal

from pydantic import Extra, Field


from agents.utils_agents import AlgoBase

from planning.ao_replan_algo import (

    AORePlanBase,

    AORePlanWrapper,

    FixNonesWrapper,

    NoPathSoRandomOrStayWrapper,

)



class AORePlanConfig(AlgoBase, extra=Extra.forbid):

    name: Literal['AORePlan'] = 'AORePlan'

    no_path_random: bool = True

    fix_nones: bool = True

    handoff_on_reverse: bool = False

    use_best_move: bool = True

    max_planning_steps: int = Field(10000, gt=0)

    device: str = 'cpu'



REVERSE_ACTION = {1: 2, 2: 1, 3: 4, 4: 3}



class AORePlan:

    def __init__(self, cfg: AORePlanConfig):

        self.cfg = cfg

        self.agent = None

        self._ao_wrapper = None
        self._base = None

        self._last_actions = None

        self._reverse_count = 0

        self._total_action_count = 0

    def act(self, observations, rewards=None, dones=None, info=None, skip_agents=None):

        actions = self.agent.act(observations, skip_agents)

        if not self.cfg.handoff_on_reverse:

            self._record_actions(actions)

        return actions


    def _record_actions(self, actions):

        n = len(actions)

        if self._last_actions is None or len(self._last_actions) != n:

            self._last_actions = [0] * n

        for i, a in enumerate(actions):

            if a is not None and a != 0:

                self._total_action_count += 1

                last = self._last_actions[i]

                if last != 0 and int(a) == REVERSE_ACTION.get(int(last)):

                    self._reverse_count += 1

                self._last_actions[i] = int(a)



    def record_executed_actions(self, actions):

        if not self.cfg.handoff_on_reverse:

            return

        self._record_actions(actions)


    def commit_proposals(self, executed_mask):
        """Synchronize AO-RePlan memory with switcher-selected actions."""

        if self._base is None or self._ao_wrapper is None:
            raise RuntimeError(
                "AO-RePlan must be reset before proposals can be committed."
            )
        planned_mask = self._ao_wrapper.last_planned_mask
        replaced_mask = self._ao_wrapper.last_probe_replacement_mask
        if planned_mask is None or replaced_mask is None:
            raise RuntimeError(
                "AO-RePlan must act before proposals can be committed."
            )
        if (
            len(executed_mask) != len(planned_mask)
            or len(executed_mask) != len(replaced_mask)
        ):
            raise ValueError(
                "executed_mask and AO-RePlan proposals must have equal sizes."
            )
        if any(
            bool(executed) and not bool(planned)
            for executed, planned in zip(executed_mask, planned_mask)
        ):
            raise ValueError(
                "Only planned AO-RePlan actions can be committed."
            )

        # A valid static probe may replace the C++ planner's original
        # proposal. The probe action can be executed, but the original
        # pending desired position must be cancelled rather than receiving
        # feedback for a move it did not propose.
        base_executed_mask = [
            bool(executed) and not bool(replaced)
            for executed, replaced in zip(
                executed_mask,
                replaced_mask,
            )
        ]
        self._base.commit_proposals(base_executed_mask)


    def probe_action(self, observation, agent_index=0):

        """Return an action from the agent's accumulated static-map probe."""

        if self._ao_wrapper is None:

            raise RuntimeError("AO-RePlan must be reset before using its probe.")

        return self._ao_wrapper.switch_probe.get_action(
            observation,
            agent_index,
        )

    def probe_path(self, observation, agent_index=0):

        """Return AO-RePlan's stateless local agent-ignoring Probe path.

        A fresh planner is used by this query, so tentative virtual obstacles
        can be tested and rejected without changing accumulated Probe memory.
        """

        if self._ao_wrapper is None:

            raise RuntimeError("AO-RePlan must be reset before using its probe.")

        return self._ao_wrapper.probe.get_path(
            observation,
            agent_index,
        )

    @property
    def teammate_blocked_mask(self):
        """Per-agent mask set when probe attributes a reverse to teammates."""
        if self._ao_wrapper is None:
            return None
        mask = self._ao_wrapper.last_teammate_blocked_mask
        if mask is None:
            return None
        return tuple(mask)

    @property
    def raw_dynamic_actions(self):
        """Raw dynamic-planner proposals from the latest act call."""
        if self._ao_wrapper is None:
            return None
        actions = self._ao_wrapper.last_raw_dynamic_actions
        if actions is None:
            return None
        return tuple(actions)

    @property
    def static_probe_actions(self):
        """Static-map probe actions paired with the latest dynamic proposals."""
        if self._ao_wrapper is None:
            return None
        actions = self._ao_wrapper.last_static_probe_actions
        if actions is None:
            return None
        return tuple(actions)

    @property
    def forward_clear_mask(self):
        """Whether dynamic and static planners agree on a forward move."""
        if self._ao_wrapper is None:
            return None
        mask = self._ao_wrapper.last_forward_clear_mask
        if mask is None:
            return None
        return tuple(mask)

    @property
    def last_planned_mask(self):
        """Whether each returned action came from AO planning or a valid probe."""
        if self._ao_wrapper is None:
            return None
        mask = self._ao_wrapper.last_planned_mask
        if mask is None:
            return None
        return tuple(mask)

    @property

    def reverse_action_rate(self):

        if self._total_action_count == 0:

            return 0.0

        return self._reverse_count / self._total_action_count


    def after_step(self, dones):

        if all(dones):
            self.agent = None

            self._ao_wrapper = None
            self._base = None

            self._last_actions = None

    def after_reset(self):

        base = AORePlanBase(

            use_best_move=self.cfg.use_best_move,

            max_steps=self.cfg.max_planning_steps,

            seed=self.cfg.seed,

        )

        wrapper = AORePlanWrapper(

            base,

            use_best_move=self.cfg.use_best_move,

            max_steps=self.cfg.max_planning_steps,

            handoff_on_reverse=self.cfg.handoff_on_reverse,

        )

        agent = wrapper

        if self.cfg.no_path_random:

            agent = NoPathSoRandomOrStayWrapper(agent)

        elif self.cfg.fix_nones:

            agent = FixNonesWrapper(agent)

        self.agent = agent

        self._ao_wrapper = wrapper
        self._base = base

        self._last_actions = None

        self._reverse_count = 0

        self._total_action_count = 0
