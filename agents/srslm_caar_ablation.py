"""Final SRSLM wait ablations using one selected CAAR and NoWait checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Literal

import numpy as np
from pydantic import Extra, Field

from agents.switcher_caar import AllStateCaarSwitcher, AllStateCaarSwitcherConfig
from agents.switcher_caar_candidate import (
    CAAR_CANDIDATE_LABEL,
    CaarCandidateArtifact,
    CaarSwitcherCandidate,
    CaarSwitcherCandidateConfig,
)
from agents.switcher_core import (
    AllStateSwitcherController,
    SwitcherController,
    OnlyWaitController,
)
from agents.utils_agents import AlgoBase
from planning.aoreplan_branch import AORePlanBranch


NO_WAIT_MODE = "all_state_switcher_caar"
ONLY_WAIT_MODE = "aoreplan_wait_detect_only_caar"


class SRSLMNoWaitConfig(AlgoBase, extra=Extra.forbid):
    name: Literal["SRSLM-NoWait"] = "SRSLM-NoWait"
    candidate: CaarSwitcherCandidateConfig
    switcher: AllStateCaarSwitcherConfig
    max_planning_steps: int = Field(10_000, gt=0)


class SRSLMOnlyWaitConfig(AlgoBase, extra=Extra.forbid):
    name: Literal["SRSLM-OnlyWait"] = "SRSLM-OnlyWait"
    candidate: CaarSwitcherCandidateConfig
    max_planning_steps: int = Field(10_000, gt=0)


def _frozen_candidate(cfg, project_root: Path, factory: Callable):
    artifact = CaarCandidateArtifact.from_config(cfg.candidate, project_root)
    candidate = factory(
        artifact,
        seed=int(cfg.seed or 0),
        device=str(cfg.candidate.device),
    )
    candidate.verify_frozen()
    return candidate


class _BaseDeployment:
    cfg: AlgoBase

    def set_grid_config(self, grid_config):
        self.controller.set_grid_config(grid_config)

    def set_env(self, env):
        self.controller.set_env(env)

    def after_step(self, dones):
        self.controller.after_step(dones)

    def get_additional_info(self):
        return self.get_switch_stats()

    def get_action_correction_stats(self):
        return self.candidate.get_action_correction_stats()


class SRSLMNoWait(_BaseDeployment):
    """Invoke the independently trained Switcher on every planner state."""

    def __init__(
        self,
        cfg: SRSLMNoWaitConfig,
        *,
        project_root: Path | None = None,
        candidate_factory: Callable = CaarSwitcherCandidate.load,
        planner_factory: Callable = AORePlanBranch,
        switcher_factory: Callable = AllStateCaarSwitcher,
    ):
        self.cfg = cfg
        root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[1]
        )
        self.candidate = _frozen_candidate(cfg, root, candidate_factory)
        switcher_cfg = cfg.switcher.copy(
            deep=True, update={"seed": cfg.seed, "device": cfg.device}
        )
        self.switcher = switcher_factory(switcher_cfg)
        candidate_artifact = getattr(self.candidate, "artifact", None)
        switcher_artifact = getattr(self.switcher, "candidate_artifact", None)
        if candidate_artifact is None or switcher_artifact is None:
            raise RuntimeError("NoWait requires both candidate artifact declarations.")
        if candidate_artifact.as_dict() != switcher_artifact.as_dict():
            raise RuntimeError("NoWait CAAR differs from the candidate pinned by its Switcher.")
        planner = planner_factory(max_steps=cfg.max_planning_steps, seed=cfg.seed)
        self.controller = AllStateSwitcherController(self.candidate, planner)
        self.device = getattr(self.candidate, "device", cfg.device)

    def after_reset(self):
        self.controller.after_reset()
        self.switcher.after_reset()

    def act(self, observations, rewards=None, dones=None, infos=None):
        prepared = self.controller.prepare_actions(observations, rewards, dones, infos)
        branches = self.switcher.choose(prepared.switcher_state)
        return list(self.controller.resolve_actions(branches).actions)

    def get_switch_stats(self):
        result = {
            "hybrid_mode": NO_WAIT_MODE,
            "ablation_name": "SRSLM-NoWait",
            "switch_pair": [CAAR_CANDIDATE_LABEL, "AORePlan"],
            "switcher_training": "PPO",
            "switcher_weight_source_algorithm": "SRSLM-NoWait",
            "switcher_training_decision_scope": "all_states",
            "value_predictor_loaded": False,
            "candidate_provenance": self.candidate.get_model_provenance(),
        }
        result.update(self.controller.get_stats())
        result.update(self.switcher.get_stats())
        return result


class SRSLMOnlyWait(_BaseDeployment):
    """Use CAAR on AORePlan waits and AORePlan on every non-wait state."""

    def __init__(
        self,
        cfg: SRSLMOnlyWaitConfig,
        *,
        project_root: Path | None = None,
        candidate_factory: Callable = CaarSwitcherCandidate.load,
        planner_factory: Callable = AORePlanBranch,
    ):
        self.cfg = cfg
        root = (
            Path(project_root).resolve()
            if project_root is not None
            else Path(__file__).resolve().parents[1]
        )
        self.candidate = _frozen_candidate(cfg, root, candidate_factory)
        planner = planner_factory(max_steps=cfg.max_planning_steps, seed=cfg.seed)
        self.controller = OnlyWaitController(self.candidate, planner)
        self.device = getattr(self.candidate, "device", cfg.device)

    def after_reset(self):
        self.controller.after_reset()

    def act(self, observations, rewards=None, dones=None, infos=None):
        self.controller.prepare_actions(observations, rewards, dones, infos)
        return list(self.controller.resolve_actions().actions)

    def get_switch_stats(self):
        result = {
            "hybrid_mode": ONLY_WAIT_MODE,
            "ablation_name": "SRSLM-OnlyWait",
            "switch_pair": [CAAR_CANDIDATE_LABEL, "AORePlan"],
            "switcher_training": "none",
            "value_predictor_loaded": False,
            "switcher_model_choice_count": 0,
            "switcher_model_selected_ao_count": 0,
            "switcher_stochastic": False,
            "candidate_provenance": self.candidate.get_model_provenance(),
        }
        result.update(self.controller.get_stats())
        return result


__all__ = [
    "NO_WAIT_MODE",
    "ONLY_WAIT_MODE",
    "SRSLMNoWait",
    "SRSLMNoWaitConfig",
    "SRSLMOnlyWait",
    "SRSLMOnlyWaitConfig",
]
