from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from pydantic import Extra, root_validator
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
    path_to_weights: str
    checkpoint_kind: Literal["latest", "best", "milestone"] = "latest"
    milestone_checkpoint: Optional[str] = None

    action_sampling: Literal["torch", "direct_numpy"] = "torch"

    @root_validator
    def explicit_checkpoint_selection(cls, values):
        milestone = values.get("milestone_checkpoint")
        if values.get("checkpoint_kind") == "milestone":
            if not isinstance(milestone, str) or not milestone.strip():
                raise ValueError(
                    "checkpoint_kind='milestone' requires milestone_checkpoint"
                )
        elif milestone is not None:
            raise ValueError(
                "milestone_checkpoint requires checkpoint_kind='milestone'"
            )
        return values


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

    def _load_checkpoint(self, checkpoint_dir, device, checkpoint_kind):
        if checkpoint_kind != "milestone":
            checkpoint = super()._load_checkpoint(
                checkpoint_dir, device, checkpoint_kind
            )
            return checkpoint

        candidate = Path(self.algo_cfg.milestone_checkpoint).expanduser()
        if not candidate.is_absolute():
            project_root = Path(__file__).resolve().parents[1]
            candidate = project_root / candidate
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Missing requested milestone checkpoint: {candidate}"
            )
        self.checkpoint_path = candidate
        checkpoint = self._load_checkpoint_path(candidate, device, "milestone")
        return checkpoint

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
            raw_tau=False,
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

        if self.algo_cfg.action_sampling == "torch":
            return policy_outputs["actions"].detach().cpu().numpy()
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
