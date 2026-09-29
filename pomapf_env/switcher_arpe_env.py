"""Switcher environments using a frozen ARPE policy."""

from __future__ import annotations

from copy import deepcopy

import gymnasium as gym
import numpy as np

from agents.arpe import (
    ARPE_CANDIDATE_LABEL,
    ArpeCandidateArtifact,
    ARPE,
)
from agents.switcher_core import (
    SWITCHER_FEATURE_SCHEMA,
    SwitcherController,
    switcher_observation_space,
)
from agents.utils_agents import SUPPORTED_COLLISION_SYSTEMS
from planning.aoreplan_branch import AORePlanBranch
from pomapf_env.env import make_pomapf
from pomapf_env.switcher_env import SwitcherEnv


ARPE_SWITCHER_ENV_SCHEMA = "srslm_switcher_caar_candidate_env_v1"


class ArpeSwitcherEnv(SwitcherEnv):
    """Keep the established Switcher reward/state and replace branch zero."""

    controller_class = SwitcherController
    integration_schema = ARPE_SWITCHER_ENV_SCHEMA

    def __init__(
        self,
        *,
        grid_config,
        candidate_artifact: ArpeCandidateArtifact,
        candidate_device: str = "cuda",
        max_planning_steps: int = 10_000,
        team_reward_coefficient: float = 1.0,
        feature_schema: str = SWITCHER_FEATURE_SCHEMA,
        candidate_factory=ARPE.load,
        planner_factory=AORePlanBranch,
        base_env_factory=make_pomapf,
    ):
        # Initialise shared runtime fields here; the base is not directly
        # constructible. This environment owns the frozen candidate lifecycle.
        gym.Env.__init__(self)
        if feature_schema != SWITCHER_FEATURE_SCHEMA:
            raise ValueError(f"Unsupported Switcher feature schema {feature_schema!r}.")
        collision_system = getattr(grid_config, "collision_system", None)
        if collision_system not in SUPPORTED_COLLISION_SYSTEMS:
            raise ValueError(
                "Switcher training requires a supported collision "
                f"system; received {collision_system!r}, supported "
                f"{SUPPORTED_COLLISION_SYSTEMS}."
            )
        if not np.isfinite(team_reward_coefficient):
            raise ValueError("team_reward_coefficient must be finite.")

        self.base_env = base_env_factory(
            grid_config=deepcopy(grid_config), with_animations=False
        )
        candidate = candidate_factory(
            candidate_artifact,
            seed=int(grid_config.seed or 0),
            device=str(candidate_device),
        )
        planner = planner_factory(
            max_steps=int(max_planning_steps), seed=int(grid_config.seed or 0)
        )
        self.controller = self.controller_class(candidate, planner)
        self.candidate = candidate
        self.candidate_artifact = candidate_artifact
        self.candidate_label = ARPE_CANDIDATE_LABEL
        self.team_reward_coefficient = float(team_reward_coefficient)
        self.feature_schema = feature_schema
        self.observation_space = switcher_observation_space()
        self.action_space = gym.spaces.Discrete(2)
        self.num_agents = int(grid_config.num_agents)
        self.is_multiagent = True
        self._prepared = None

    def get_candidate_provenance(self) -> dict[str, object]:
        return deepcopy(self.candidate.get_model_provenance())


__all__ = [
    "ARPE_SWITCHER_ENV_SCHEMA",
    "ArpeSwitcherEnv",
    "switcher_observation_space",
]
