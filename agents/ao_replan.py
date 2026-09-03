from typing import Literal

from pogema import GridConfig
from pydantic import Extra, Field

from agents.reverse_metrics import ExecutedPositionReverseCounter
from agents.utils_agents import AlgoBase
from planning.ao_replan_algo import AORePlanBase, AORePlanWrapper


class AORePlanConfig(AlgoBase, extra=Extra.forbid):
    name: Literal["AORePlan"] = "AORePlan"
    max_planning_steps: int = Field(10_000, gt=0)


class AORePlan:
    """Standalone AORePlan with one fixed planning policy."""

    WRAPPER_CLASS = AORePlanWrapper

    def __init__(self, cfg: AORePlanConfig):
        self.cfg = cfg
        self.agent = None
        self._ao_wrapper = None
        self._base = None
        self._reverse_counter = ExecutedPositionReverseCounter(
            GridConfig().MOVES
        )
        self._raw_plan_movement_count = 0
        self._static_astar_query_count = 0
        self._no_path_fallback_count = 0

    def act(
        self,
        observations,
        rewards=None,
        dones=None,
        info=None,
        skip_agents=None,
    ):
        del rewards, dones, info
        skip = (
            list(skip_agents)
            if skip_agents is not None
            else [False] * len(observations)
        )
        actions = self.agent.act(observations, skip_agents=skip)
        self._commit_current_actions()
        self._reverse_counter.record(actions, observations)
        self._record_static_astar_metrics()
        return actions

    def _commit_current_actions(self):
        selected = list(self._ao_wrapper.last_planned_mask)
        base_mask = [
            bool(executed) and not bool(overridden)
            for executed, overridden in zip(
                selected,
                self._ao_wrapper.last_dynamic_override_mask,
            )
        ]
        self._base.commit_proposals(base_mask)

    def _record_static_astar_metrics(self):
        raw_movement = tuple(
            action not in (None, 0)
            for action in self._ao_wrapper.last_raw_dynamic_actions
        )
        self._raw_plan_movement_count += sum(raw_movement)
        self._static_astar_query_count += sum(
            bool(invoked) and movement
            for invoked, movement in zip(
                self._ao_wrapper.last_static_astar_invoked_mask,
                raw_movement,
            )
        )
        self._no_path_fallback_count += sum(
            bool(value)
            for value in self._ao_wrapper.last_no_path_fallback_mask
        )

    def set_grid_config(self, grid_config):
        if getattr(grid_config, "collision_system", None) != "block_both":
            raise ValueError(
                "AORePlan requires collision_system='block_both'."
            )

    def _latest(self, name):
        if self._ao_wrapper is None:
            return None
        values = getattr(self._ao_wrapper, name)
        return None if values is None else tuple(values)

    @property
    def raw_dynamic_actions(self):
        return self._latest("last_raw_dynamic_actions")

    @property
    def no_path_fallback_mask(self):
        return self._latest("last_no_path_fallback_mask")

    @property
    def reverse_mask(self):
        return self._latest("last_reverse_mask")

    @property
    def last_planned_mask(self):
        return self._latest("last_planned_mask")

    @property
    def reverse_action_rate(self):
        return self._reverse_counter.rate

    @property
    def reverse_action_count(self):
        return self._reverse_counter.reverse_count

    @property
    def reverse_action_denominator(self):
        return self._reverse_counter.movement_count

    @property
    def reverse_metric_version(self):
        return self._reverse_counter.METRIC_VERSION

    @property
    def static_astar_query_count(self):
        return self._static_astar_query_count

    @property
    def static_astar_query_denominator(self):
        return self._raw_plan_movement_count

    @property
    def static_astar_query_rate(self):
        if self._raw_plan_movement_count == 0:
            return 0.0
        return self._static_astar_query_count / self._raw_plan_movement_count

    @property
    def no_path_fallback_count(self):
        return self._no_path_fallback_count

    def after_step(self, dones):
        if all(dones):
            self.agent = None
            self._ao_wrapper = None
            self._base = None

    def after_reset(self):
        self._base = AORePlanBase(
            max_steps=self.cfg.max_planning_steps,
            seed=self.cfg.seed,
        )
        self._ao_wrapper = self.WRAPPER_CLASS(
            self._base,
            max_steps=self.cfg.max_planning_steps,
        )
        self.agent = self._ao_wrapper
        self._reverse_counter.reset()
        self._raw_plan_movement_count = 0
        self._static_astar_query_count = 0
        self._no_path_fallback_count = 0
