"""Current-V3 ablations for SRSLM's deterministic wait detector."""

from __future__ import annotations

from typing import Callable, Literal

from pydantic import Extra, Field

from agents.caar import CAAR, CAARConfig
from agents.switcher import AllStateSwitcher, AllStateSwitcherConfig
from agents.switcher_core import (
    AllStateSwitcherController,
    WaitDetectOnlyController,
)
from agents.utils_agents import AlgoBase
from planning.aoreplan_branch import AORePlanBranch


NO_WAIT_DETECT_MODE = "all_state_switcher_v3"
WAIT_DETECT_ONLY_MODE = "aoreplan_wait_detect_only_v3"


class SRSLMNoWaitDetectConfig(AlgoBase, extra=Extra.forbid):
    name: Literal["SRSLM-NoWaitDetect"] = "SRSLM-NoWaitDetect"
    caar: CAARConfig = CAARConfig()
    switcher: AllStateSwitcherConfig = AllStateSwitcherConfig()
    max_planning_steps: int = Field(10_000, gt=0)


class SRSLMWaitDetectOnlyConfig(AlgoBase, extra=Extra.forbid):
    name: Literal["SRSLM-WaitDetectOnly"] = "SRSLM-WaitDetectOnly"
    caar: CAARConfig = CAARConfig()
    max_planning_steps: int = Field(10_000, gt=0)


def _frozen_caar(cfg, factory: Callable):
    caar = factory(cfg)
    for parameter in caar.ppo.parameters():
        parameter.requires_grad_(False)
    return caar


class SRSLMNoWaitDetect:
    """Use a separately trained Switcher on all AORePlan states."""

    def __init__(
        self,
        cfg: SRSLMNoWaitDetectConfig,
        *,
        caar_factory: Callable = CAAR,
        planner_factory: Callable = AORePlanBranch,
        switcher_factory: Callable = AllStateSwitcher,
    ):
        self.cfg = cfg
        caar_cfg = cfg.caar.copy(deep=True, update={"seed": cfg.seed})
        switcher_cfg = cfg.switcher.copy(deep=True, update={"seed": cfg.seed})
        self.caar = _frozen_caar(caar_cfg, caar_factory)
        self.switcher = switcher_factory(switcher_cfg)
        planner = planner_factory(
            max_steps=cfg.max_planning_steps,
            seed=cfg.seed,
        )
        self.controller = AllStateSwitcherController(self.caar, planner)
        self.device = getattr(self.caar, "device", cfg.device)

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
        branches = self.switcher.choose(prepared.switcher_state)
        return list(self.controller.resolve_actions(branches).actions)

    def after_step(self, dones):
        self.controller.after_step(dones)

    def get_switch_stats(self):
        result = {
            "hybrid_mode": NO_WAIT_DETECT_MODE,
            "ablation_name": "SRSLM-NoWaitDetect",
            "switch_pair": ["CAAR", "AORePlan"],
            "switcher_training": "PPO",
            "value_predictor_loaded": False,
        }
        result.update(self.controller.get_stats())
        result.update(self.switcher.get_stats())
        return result

    def get_additional_info(self):
        return self.get_switch_stats()

    def get_action_correction_stats(self):
        return self.caar.get_action_correction_stats()


class SRSLMWaitDetectOnly:
    """Use CAAR only when AORePlan waits; otherwise execute AORePlan."""

    def __init__(
        self,
        cfg: SRSLMWaitDetectOnlyConfig,
        *,
        caar_factory: Callable = CAAR,
        planner_factory: Callable = AORePlanBranch,
    ):
        self.cfg = cfg
        caar_cfg = cfg.caar.copy(deep=True, update={"seed": cfg.seed})
        self.caar = _frozen_caar(caar_cfg, caar_factory)
        planner = planner_factory(
            max_steps=cfg.max_planning_steps,
            seed=cfg.seed,
        )
        self.controller = WaitDetectOnlyController(self.caar, planner)
        self.device = getattr(self.caar, "device", cfg.device)

    def set_grid_config(self, grid_config):
        self.controller.set_grid_config(grid_config)

    def set_env(self, env):
        self.controller.set_env(env)

    def after_reset(self):
        self.controller.after_reset()

    def act(self, observations, rewards=None, dones=None, infos=None):
        self.controller.prepare_actions(
            observations,
            rewards,
            dones,
            infos,
        )
        return list(self.controller.resolve_actions().actions)

    def after_step(self, dones):
        self.controller.after_step(dones)

    def get_switch_stats(self):
        result = {
            "hybrid_mode": WAIT_DETECT_ONLY_MODE,
            "ablation_name": "SRSLM-WaitDetectOnly",
            "switch_pair": ["CAAR", "AORePlan"],
            "switcher_training": "none",
            "value_predictor_loaded": False,
            "switcher_model_choice_count": 0,
            "switcher_model_selected_ao_count": 0,
            "switcher_stochastic": False,
        }
        result.update(self.controller.get_stats())
        return result

    def get_additional_info(self):
        return self.get_switch_stats()

    def get_action_correction_stats(self):
        return self.caar.get_action_correction_stats()


__all__ = [
    "NO_WAIT_DETECT_MODE",
    "SRSLMNoWaitDetect",
    "SRSLMNoWaitDetectConfig",
    "SRSLMWaitDetectOnly",
    "SRSLMWaitDetectOnlyConfig",
    "WAIT_DETECT_ONLY_MODE",
]
