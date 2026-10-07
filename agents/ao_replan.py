from typing import Literal

from pydantic import Extra, Field

from agents.utils_agents import AlgoBase, SUPPORTED_COLLISION_SYSTEMS
from planning.ao_replan_algo import AORePlanBase, AORePlanWrapper


class AORePlanConfig(AlgoBase, extra=Extra.forbid):
    name: Literal["AORePlan"] = "AORePlan"
    max_planning_steps: int = Field(10_000, gt=0)


class AORePlan:
    def __init__(self, cfg: AORePlanConfig):
        self.cfg = cfg
        self._ao_wrapper = None

    def act(
        self,
        observations,
        rewards=None,
        dones=None,
        info=None,
        skip_agents=None,
    ):
        del rewards, dones, info
        actions = self._ao_wrapper.act(observations, skip_agents=skip_agents)
        self._ao_wrapper.commit_proposals([action is not None for action in actions])
        return actions

    def set_grid_config(self, grid_config):
        collision_system = getattr(grid_config, "collision_system", None)
        if collision_system not in SUPPORTED_COLLISION_SYSTEMS:
            raise ValueError(
                "AORePlan requires a supported collision system; "
                f"received collision_system={collision_system!r}, supported "
                f"{SUPPORTED_COLLISION_SYSTEMS}."
            )

    def after_step(self, dones):
        if all(dones):
            self._ao_wrapper = None

    def after_reset(self):
        base = AORePlanBase(
            max_steps=self.cfg.max_planning_steps,
            seed=self.cfg.seed,
        )
        self._ao_wrapper = AORePlanWrapper(
            base,
            max_steps=self.cfg.max_planning_steps,
        )
