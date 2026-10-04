from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

import numpy as np
from pydantic import Extra, Field

from agents.switcher import Switcher, SwitcherConfig
from agents.arpe import (
    ARPE,
    ARPEConfig,
    ArpeCandidateArtifact,
)
from agents.switcher_core import SwitcherController
from agents.utils_agents import AlgoBase
from planning.aoreplan_branch import AORePlanBranch


class SRSLMConfig(AlgoBase, extra=Extra.forbid):
    name: Literal["SRSLM"] = "SRSLM"
    switcher: SwitcherConfig = SwitcherConfig()
    max_planning_steps: int = Field(10_000, gt=0)
    candidate: ARPEConfig | None = None


class SRSLM:
    def set_grid_config(self, grid_config):
        self.controller.set_grid_config(grid_config)

    def set_env(self, env):
        self.controller.set_env(env)

    def after_step(self, dones):
        self.controller.after_step(dones)

    def get_additional_info(self):
        return self.get_switch_stats()

    def __init__(
        self,
        cfg: SRSLMConfig,
        *,
        project_root: Path | None = None,
        candidate_factory: Callable = ARPE.load,
        planner_factory: Callable = AORePlanBranch,
        switcher_factory: Callable = Switcher,
        controller_factory: Callable = SwitcherController,
    ):
        self.cfg = cfg
        switcher_cfg = cfg.switcher.copy(
            deep=True,
            update={"seed": cfg.seed},
        )
        self.switcher = switcher_factory(switcher_cfg)
        candidate = getattr(self.switcher, "candidate_artifact", None)
        if cfg.candidate is not None:
            candidate = ArpeCandidateArtifact.from_config(
                cfg.candidate,
                project_root or Path(__file__).resolve().parents[1],
            )
        if candidate is None:
            raise RuntimeError(
                "Switcher checkpoint does not contain ARPE candidate paths."
            )
        self.candidate = candidate_factory(
            candidate,
            seed=int(cfg.seed or 0),
            device=str(cfg.device),
        )
        planner = planner_factory(
            max_steps=cfg.max_planning_steps,
            seed=cfg.seed,
        )
        self.controller = controller_factory(self.candidate, planner)
        self.device = getattr(self.candidate, "device", cfg.device)

    def after_reset(self):
        self.controller.after_reset()
        self.switcher.after_reset()

    def act(self, observations, rewards=None, dones=None, infos=None):
        prepared = self.controller.prepare_actions(
            observations,
            rewards,
            dones,
            infos,
        )
        switch_allowed = np.asarray(
            prepared.switch_allowed_mask,
            dtype=bool,
        )
        if np.any(switch_allowed):
            active_state = {
                key: np.asarray(value)[switch_allowed]
                for key, value in prepared.switcher_state.items()
            }
            branches = self.switcher.choose(active_state)
        else:
            branches = np.empty(0, dtype=np.int64)
        return list(self.controller.resolve_actions(branches).actions)

    def get_switch_stats(self):
        result = self.controller.get_stats()
        result.update(self.switcher.get_stats())
        return result


__all__ = [
    "SRSLM",
    "SRSLMConfig",
]
