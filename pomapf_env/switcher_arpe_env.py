"""Switcher environments using a frozen ARPE policy."""

from __future__ import annotations

from copy import deepcopy

import gymnasium as gym
import numpy as np

from agents.arpe import (
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


ARPE_SWITCHER_ENV_SCHEMA = "srslm_switcher_caar_candidate_env_v1"


class ArpeSwitcherEnv(gym.Env):
    """Train the Switcher with frozen ARPE and AORePlan branches."""

    metadata = {"render_modes": []}

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
        super().__init__()
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
        self.team_reward_coefficient = float(team_reward_coefficient)
        self.feature_schema = feature_schema
        self.observation_space = switcher_observation_space()
        self.action_space = gym.spaces.Discrete(2)
        self.num_agents = int(grid_config.num_agents)
        self.is_multiagent = True
        self._prepared = None

    @property
    def grid_config(self):
        return self.base_env.grid_config

    @property
    def grid(self):
        return self.base_env.grid

    def get_num_agents(self):
        return self.num_agents

    @staticmethod
    def _feature_observations(features):
        count = int(np.asarray(features["obs"]).shape[0])
        return [
            {
                key: np.asarray(value[index], dtype=np.float32).copy()
                for key, value in features.items()
            }
            for index in range(count)
        ]

    def _start_policy_episode(self, observations, infos):
        self.controller.after_reset()
        self.controller.set_grid_config(self.base_env.grid_config)
        self.controller.set_env(self.base_env)
        count = len(observations)
        self._prepared = self.controller.prepare_actions(
            observations,
            np.zeros(count, dtype=np.float32),
            np.zeros(count, dtype=np.bool_),
            infos,
        )
        return self._feature_observations(self._prepared.switcher_state)

    def reset(self, *, seed=None, options=None):
        del options
        if seed is not None:
            # POGEMA uses the seed stored in its grid config.  Sample Factory
            # normally leaves this unset, but accepting it keeps Gymnasium's
            # reset contract intact.
            self.base_env.grid_config.seed = int(seed)
        observations, infos = self.base_env.reset()
        return self._start_policy_episode(observations, infos), infos

    def _append_episode_stats(self, infos) -> None:
        stats = self.controller.get_stats()
        episode_stats = {
            "switch_selected_ao_rate": stats["selected_ao_rate"],
            "switch_executed_ao_rate": stats["executed_ao_rate"],
            "switch_wait_bypass_rate": stats["aoreplan_wait_bypass_rate"],
            "branch_action_agreement_rate": stats["branch_action_agreement_rate"],
        }
        for info in infos:
            target = info.setdefault("episode_extra_stats", {})
            target.update(episode_stats)

    def step(self, selector_actions):
        if self._prepared is None:
            raise RuntimeError("reset() must be called before step().")
        choices = np.asarray(selector_actions, dtype=np.int64).reshape(-1)
        if choices.shape != (self.num_agents,):
            raise ValueError("Switcher returned the wrong number of training actions.")
        switch_allowed = np.asarray(
            self._prepared.switch_allowed_mask,
            dtype=bool,
        )
        decision = self.controller.resolve_actions(choices[switch_allowed])
        (
            observations,
            rewards,
            terminated,
            truncated,
            infos,
        ) = self.base_env.step(list(decision.actions))
        done = np.logical_or(
            np.asarray(terminated, dtype=np.bool_),
            np.asarray(truncated, dtype=np.bool_),
        )
        self.controller.after_step(done)

        rewards_array = np.asarray(rewards, dtype=np.float32)
        team_mean = float(rewards_array.mean())
        training_rewards = rewards_array + (self.team_reward_coefficient * team_mean)

        episode_finished = bool(done.size and np.all(done))
        if episode_finished:
            self._append_episode_stats(infos)
            # make_pomapf includes AutoResetWrapper, so observations already
            # belong to the next episode here.
            next_features = self._start_policy_episode(observations, infos)
        else:
            self._prepared = self.controller.prepare_actions(
                observations,
                rewards_array,
                done,
                infos,
            )
            next_features = self._feature_observations(self._prepared.switcher_state)

        return (
            next_features,
            training_rewards.tolist(),
            list(terminated),
            list(truncated),
            infos,
        )

    def close(self):
        self._prepared = None
        return self.base_env.close()


__all__ = [
    "ARPE_SWITCHER_ENV_SCHEMA",
    "ArpeSwitcherEnv",
    "switcher_observation_space",
]
