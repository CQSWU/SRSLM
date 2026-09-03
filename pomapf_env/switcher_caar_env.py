"""Switcher environments backed by one strictly pinned CAAR milestone."""

from __future__ import annotations

from copy import deepcopy

import gymnasium as gym
import numpy as np

from agents.switcher_caar_candidate import (
    CAAR_CANDIDATE_LABEL,
    CaarCandidateArtifact,
    CaarSwitcherCandidate,
)
from agents.switcher_core import (
    AllStateSwitcherController,
    NUM_PRIMITIVE_ACTIONS,
    SWITCHER_COORD_DIM,
    SWITCHER_FEATURE_SCHEMA,
    SWITCHER_SPATIAL_SHAPE,
    SwitcherController,
)
from planning.aoreplan_branch import AORePlanBranch
from pomapf_env.env import make_pomapf
from pomapf_env.switcher_env import SwitcherEnv


CAAR_SWITCHER_ENV_SCHEMA = "srslm_switcher_caar_candidate_env_v1"
CAAR_NOWAIT_ENV_SCHEMA = "srslm_switcher_caar_candidate_all_states_env_v1"


def switcher_observation_space() -> gym.spaces.Dict:
    """Return the unchanged five-field Switcher-v3 observation contract."""

    return gym.spaces.Dict(
        {
            "obs": gym.spaces.Box(
                0.0, 1.0, shape=SWITCHER_SPATIAL_SHAPE, dtype=np.float32
            ),
            "xy": gym.spaces.Box(
                -1024.0,
                1024.0,
                shape=(SWITCHER_COORD_DIM,),
                dtype=np.float32,
            ),
            "target_xy": gym.spaces.Box(
                -1024.0,
                1024.0,
                shape=(SWITCHER_COORD_DIM,),
                dtype=np.float32,
            ),
            # The public feature name remains caar_action because branch zero
            # is CAAR and changing it would alter the trained network state.
            "caar_action": gym.spaces.Box(
                0.0,
                1.0,
                shape=(NUM_PRIMITIVE_ACTIONS,),
                dtype=np.float32,
            ),
            "aoreplan_action": gym.spaces.Box(
                0.0,
                1.0,
                shape=(NUM_PRIMITIVE_ACTIONS,),
                dtype=np.float32,
            ),
        }
    )


class CaarSwitcherEnv(SwitcherEnv):
    """Keep the established Switcher reward/state and replace branch zero."""

    controller_class = SwitcherController
    integration_schema = CAAR_SWITCHER_ENV_SCHEMA

    def __init__(
        self,
        *,
        grid_config,
        candidate_artifact: CaarCandidateArtifact,
        candidate_device: str = "cuda",
        max_planning_steps: int = 10_000,
        team_reward_coefficient: float = 1.0,
        feature_schema: str = SWITCHER_FEATURE_SCHEMA,
        candidate_factory=CaarSwitcherCandidate.load,
        planner_factory=AORePlanBranch,
        base_env_factory=make_pomapf,
    ):
        # SwitcherEnv.__init__ constructs the legacy CAAR class.  Initialise
        # its shared runtime fields directly so this environment uses exactly
        # the milestone declared in candidate_artifact.
        gym.Env.__init__(self)
        if feature_schema != SWITCHER_FEATURE_SCHEMA:
            raise ValueError(f"Unsupported Switcher feature schema {feature_schema!r}.")
        if getattr(grid_config, "collision_system", None) != "block_both":
            raise ValueError("Switcher training requires collision_system='block_both'.")
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
        verification = candidate.verify_frozen()
        if verification.get("verified") is not True:
            raise RuntimeError("The frozen CAAR candidate failed verification.")
        planner = planner_factory(
            max_steps=int(max_planning_steps), seed=int(grid_config.seed or 0)
        )
        self.controller = self.controller_class(candidate, planner)
        self.candidate = candidate
        self.candidate_artifact = candidate_artifact
        self.candidate_label = CAAR_CANDIDATE_LABEL
        self.candidate_provenance = candidate.get_model_provenance()
        self.team_reward_coefficient = float(team_reward_coefficient)
        self.feature_schema = feature_schema
        self.observation_space = switcher_observation_space()
        self.action_space = gym.spaces.Discrete(2)
        self.num_agents = int(grid_config.num_agents)
        self.is_multiagent = True
        self._prepared = None
        self._last_rewards = None
        self._last_dones = None
        self._last_infos = None

    def get_candidate_provenance(self) -> dict[str, object]:
        current = self.candidate.get_model_provenance()
        if current != self.candidate_provenance:
            raise RuntimeError("Frozen CAAR provenance changed at runtime.")
        return deepcopy(current)


class CaarNoWaitSwitcherEnv(CaarSwitcherEnv):
    """Train the two-action Switcher on every state, including planner waits."""

    controller_class = AllStateSwitcherController
    integration_schema = CAAR_NOWAIT_ENV_SCHEMA


__all__ = [
    "CAAR_NOWAIT_ENV_SCHEMA",
    "CAAR_SWITCHER_ENV_SCHEMA",
    "CaarNoWaitSwitcherEnv",
    "CaarSwitcherEnv",
    "switcher_observation_space",
]
