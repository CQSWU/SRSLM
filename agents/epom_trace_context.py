from __future__ import annotations

from copy import deepcopy
from typing import Literal

import numpy as np
import torch
from pydantic import Extra
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.model_utils import get_rnn_size

from agents.policy_backbone import PolicyBackbone, PolicyBackboneConfig
from learning.grid_memory import MultipleGridMemory
from pomapf_env.trace_routing import TIE_KEY, bonus_rng, draw_tie_ranks
from pomapf_env.wrappers import MatrixObservationWrapper

TRACE_RADIUS = 5
TRACE_SIZE = 2 * TRACE_RADIUS + 1


class EPOMTraceContextConfig(PolicyBackboneConfig, extra=Extra.forbid):
    name: Literal["EPOM-TraceContext"] = "EPOM-TraceContext"


class EPOMTraceContext(PolicyBackbone):
    def __init__(self, algo_cfg: EPOMTraceContextConfig):
        super().__init__(algo_cfg)
        self.grid_memory_radius = int(
            self.cfg.full_config["environment"]["grid_memory_obs_radius"]
        )
        self.grid_memory = MultipleGridMemory()
        if int(self.tau_radius) != TRACE_RADIUS:
            raise RuntimeError(
                "Runtime tau_radius disagrees with checkpoint config: "
                f"{self.tau_radius} != {TRACE_RADIUS}"
            )
        self._numpy_rng = np.random.default_rng(algo_cfg.seed)
        self._bonus_rng = bonus_rng(algo_cfg.seed)

    def after_reset(self):
        super().after_reset()
        self.grid_memory.clear()
        self._numpy_rng = np.random.default_rng(self.algo_cfg.seed)
        self._bonus_rng = bonus_rng(self.algo_cfg.seed)

    def _add_exact_free_mask(self, observations, positions):

        for observation, (row, col) in zip(observations, positions):
            free = self.aco.extract_local_free_mask(int(row), int(col), TRACE_RADIUS)
            observation["tau_free_mask"] = free[np.newaxis, ...].astype(
                np.float32, copy=False
            )

    def act(self, observations, rewards=None, dones=None, infos=None):
        del rewards, dones, infos
        observations = deepcopy(observations)
        num_agents = len(observations)

        self.grid_memory.update(observations)
        self.grid_memory.modify_observation(observations, self.grid_memory_radius)
        observations = MatrixObservationWrapper.to_matrix(observations)

        if self.rnn_states is None or len(self.rnn_states) != num_agents:
            self.rnn_states = torch.zeros(
                [num_agents, get_rnn_size(self.cfg)],
                dtype=torch.float32,
                device=self.device,
            )
        if self.aco.tau is None:
            raise RuntimeError(
                "Context trace state is not initialised. Call set_env() after "
                "env.reset() and before act()."
            )

        positions = self._global_positions()
        self.aco.observe_for_inference(
            observations,
            positions=positions,
            radius=TRACE_RADIUS,
        )
        self._add_exact_free_mask(observations, positions)
        ranks = draw_tie_ranks(self._bonus_rng, num_agents)
        for observation, rank in zip(observations, ranks):
            observation[TIE_KEY] = rank

        with torch.no_grad():
            obs_torch = TensorDict(
                {
                    key: torch.from_numpy(np.stack([obs[key] for obs in observations]))
                    .to(self.device)
                    .float()
                    for key in observations[0]
                }
            )
            obs_torch = prepare_and_normalize_obs(self.ppo, obs_torch)
            policy_outputs = self.ppo(obs_torch, self.rnn_states)
            self.rnn_states = policy_outputs["new_rnn_states"]

        logits = policy_outputs["action_logits"].float().cpu().numpy()
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return np.asarray([self._numpy_rng.choice(5, p=row) for row in probabilities])

    def after_step(self, dones):
        super().after_step(dones)
        if all(dones):
            self.grid_memory.clear()


__all__ = [
    "EPOMTraceContext",
    "EPOMTraceContextConfig",
    "TRACE_RADIUS",
    "TRACE_SIZE",
]
