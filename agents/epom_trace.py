"""Inference adapter for the frozen-EPOM shared-trace policy."""

from copy import deepcopy
from typing import Literal

import numpy as np
import torch
from pydantic import Extra
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.model_utils import get_rnn_size

from agents.caar import CAAR, CAARConfig
from learning.grid_memory import MultipleGridMemory
from pomapf_env.wrappers import MatrixObservationWrapper


class EPOMTraceConfig(CAARConfig, extra=Extra.forbid):
    name: Literal["EPOM-Trace"] = "EPOM-Trace"
    path_to_weights: str = (
        "weights/EPOM-Trace-gradient-long/seed0/"
        "EPOM-Trace-RawSmooth-Long-s0"
    )
    checkpoint_kind: Literal["auto", "latest", "best"] = "latest"


class EPOMTrace(CAAR):
    """Official EPOM v0 plus the trained shared-trace action correction."""

    def __init__(self, algo_cfg: EPOMTraceConfig):
        super().__init__(algo_cfg)
        self.grid_memory_radius = int(
            self.cfg.full_config["environment"]["grid_memory_obs_radius"]
        )
        self.grid_memory = MultipleGridMemory()

    def after_reset(self):
        super().after_reset()
        self.grid_memory.clear()

    def act(self, observations, rewards=None, dones=None, infos=None):
        del rewards, dones, infos
        observations = deepcopy(observations)
        num_agents = len(observations)

        self.grid_memory.update(observations)
        self.grid_memory.modify_observation(
            observations,
            self.grid_memory_radius,
        )
        observations = MatrixObservationWrapper.to_matrix(observations)

        if self.rnn_states is None or len(self.rnn_states) != num_agents:
            self.rnn_states = torch.zeros(
                [num_agents, get_rnn_size(self.cfg)],
                dtype=torch.float32,
                device=self.device,
            )
        if self.aco.tau is None:
            raise RuntimeError(
                "EPOM-Trace state is not initialized. Call set_env() after "
                "env.reset() and before act()."
            )
        self.aco.observe_for_inference(
            observations,
            positions=self._global_positions(),
            radius=self.tau_radius,
        )
        self._last_augmented_observations = deepcopy(observations)

        obs_torch = TensorDict(
            {
                key: torch.from_numpy(
                    np.stack([obs[key] for obs in observations])
                ).to(self.device).float()
                for key in observations[0]
            }
        )
        obs_torch = prepare_and_normalize_obs(self.ppo, obs_torch)
        policy_outputs = self.ppo(obs_torch, self.rnn_states)
        self.rnn_states = policy_outputs["new_rnn_states"]

        corrections = self.ppo.last_action_correction
        pressures = self.ppo.last_candidate_pressure
        if corrections is None or pressures is None:
            raise RuntimeError(
                "EPOM-Trace did not expose its action corrections and pressures."
            )
        self._action_correction_samples.append(
            corrections.float().cpu().numpy().reshape(-1)
        )
        self._candidate_pressure_samples.append(
            pressures.float().cpu().numpy().reshape(-1)
        )
        return policy_outputs["actions"].detach().cpu().numpy()

    def after_step(self, dones):
        super().after_step(dones)
        if all(dones):
            self.grid_memory.clear()
