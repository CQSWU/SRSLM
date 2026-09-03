"""Deployed SRSLM with a learned probabilistic Switcher."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

import numpy as np
from pydantic import Extra, Field

from agents.switcher import Switcher, SwitcherConfig
from agents.switcher_caar_candidate import (
    CAAR_CANDIDATE_LABEL,
    CaarSwitcherCandidate,
)
from agents.switcher_core import SwitcherController
from agents.utils_agents import AlgoBase
from planning.aoreplan_branch import AORePlanBranch


SRSLM_MODE = "aoreplan_wait_bypass_switcher_v3"


class SRSLMConfig(AlgoBase, extra=Extra.forbid):
    name: Literal["SRSLM"] = "SRSLM"
    switcher: SwitcherConfig = SwitcherConfig()
    max_planning_steps: int = Field(10_000, gt=0)


class SRSLM:
    """Use CAAR for AORePlan waits and Switcher for AORePlan moves."""

    def __init__(
        self,
        cfg: SRSLMConfig,
        *,
        project_root: Path | None = None,
        candidate_factory: Callable = CaarSwitcherCandidate.load,
        planner_factory: Callable = AORePlanBranch,
        switcher_factory: Callable = Switcher,
    ):
        self.cfg = cfg
        switcher_cfg = cfg.switcher.copy(
            deep=True,
            update={"seed": cfg.seed},
        )
        self.switcher = switcher_factory(switcher_cfg)
        candidate = getattr(self.switcher, "candidate_artifact", None)
        if candidate is None:
            raise RuntimeError(
                "Switcher checkpoint does not pin its frozen CAAR candidate."
            )
        root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[1]
        )
        if candidate.project_root != root:
            raise RuntimeError(
                "Switcher candidate was resolved against a different project root."
            )
        self.candidate = candidate_factory(
            candidate,
            seed=int(cfg.seed or 0),
            device=str(cfg.device),
        )
        verification = self.candidate.verify_frozen()
        if verification.get("verified") is not True:
            raise RuntimeError("Frozen CAAR candidate verification failed.")
        planner = planner_factory(
            max_steps=cfg.max_planning_steps,
            seed=cfg.seed,
        )
        self.controller = SwitcherController(
            self.candidate,
            planner,
        )
        self.device = getattr(self.candidate, "device", cfg.device)

    def set_grid_config(self, grid_config):
        self.controller.set_grid_config(grid_config)

    def set_env(self, env):
        self.controller.set_env(env)

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

    def after_step(self, dones):
        self.controller.after_step(dones)

    def get_switch_stats(self):
        result = {
            "hybrid_mode": SRSLM_MODE,
            "switch_pair": [CAAR_CANDIDATE_LABEL, "AORePlan"],
            "switcher_training": "PPO",
            "value_predictor_loaded": False,
            "candidate_provenance": self.candidate.get_model_provenance(),
        }
        result.update(self.controller.get_stats())
        result.update(self.switcher.get_stats())
        return result

    def get_additional_info(self):
        return self.get_switch_stats()

    def get_action_correction_stats(self):
        return self.candidate.get_action_correction_stats()


__all__ = [
    "SRSLM_MODE",
    "SRSLM",
    "SRSLMConfig",
]
