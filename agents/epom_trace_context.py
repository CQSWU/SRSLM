"""Inference adapter for the contextual EPOM trace residual.

The actor-critic owns both the entropy-gated Direct rule and the learned
residual.  This adapter only reconstructs the two recurrent observations used
during training (EPOM grid memory and an 11x11 shared-trace crop), maintains
the exact 11x11 free-cell mask, and records model-side diagnostics. Radius 5
and the saved raw-versus-centred trace contract are read from the checkpoint
and enforced at inference, so evaluation cannot silently change the learned
representation.

Checkpoint selection is explicit.  ``latest`` and ``best`` use the normal
Sample Factory run directory.  ``milestone`` requires an exact checkpoint file
so a screen can never silently evaluate a newer checkpoint.

Registry note: ``train.register_custom_components`` must map
``encoder_custom=epom_trace_context`` to
``EPOMTraceContextActorCritic``.  The model/config integration owns that hook;
it deliberately does not live in this inference-only module.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from pydantic import Extra
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.model_utils import get_rnn_size

from agents.epom_trace import EPOMTrace, EPOMTraceConfig
from pomapf_env.wrappers import MatrixObservationWrapper


TRACE_RADIUS = 5
TRACE_SIZE = 2 * TRACE_RADIUS + 1


def _read_r5_trace_contract(config_path: Path) -> dict[str, object]:
    """Read and validate the observation contract saved with a checkpoint."""

    try:
        payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
        full_config = payload["full_config"]
        environment = full_config["environment"]
        architecture = full_config.get("experiment_settings", {}).get(
            "trace_context_architecture", "context"
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot read trace contract from checkpoint config {config_path}"
        ) from exc

    radius = environment.get("tau_radius")
    raw_tau = environment.get("tau_raw")
    if radius != TRACE_RADIUS:
        raise RuntimeError(
            "EPOM-TraceContext inference is fixed to the paper's 11x11 trace "
            f"crop (tau_radius=5), but checkpoint config has {radius!r}."
        )
    expected_raw_tau = architecture not in {
        "paper_entropy_multiplier",
        "paper_entropy_fusion",
    }
    if raw_tau is not expected_raw_tau:
        raise RuntimeError(
            "EPOM-TraceContext checkpoint has an invalid trace representation: "
            f"architecture={architecture!r} requires "
            f"tau_raw={expected_raw_tau!r}, got {raw_tau!r}."
        )
    return {
        "tau_radius": TRACE_RADIUS,
        "tau_size": TRACE_SIZE,
        "tau_raw": expected_raw_tau,
        "trace_context_architecture": architecture,
    }


class EPOMTraceContextConfig(EPOMTraceConfig, extra=Extra.forbid):
    name: Literal["EPOM-TraceContext"] = "EPOM-TraceContext"
    path_to_weights: str = (
        "weights/EPOM-TraceContext-R5/EPOM-TraceContext-R5"
    )
    checkpoint_kind: Literal[
        "auto", "latest", "best", "milestone"
    ] = "latest"
    milestone_checkpoint: Optional[str] = None
    # Evaluation-only ablation. ``checkpoint`` preserves the learned run's
    # entropy gate; ``all`` applies the same learned correction at every step.
    learned_gate_override: Literal["checkpoint", "all"] = "checkpoint"


class EPOMTraceContext(EPOMTrace):
    """Frozen EPOM-L, the Direct rule, and a contextual learned residual."""

    def __init__(self, algo_cfg: EPOMTraceContextConfig):
        super().__init__(algo_cfg)
        verifier = getattr(self.ppo, "verify_frozen_actor_backbone", None)
        if not callable(verifier):
            raise RuntimeError(
                "EPOM-TraceContext actor-critic does not expose the required "
                "frozen-backbone verification hook."
            )
        # ``super().__init__`` has already loaded the complete learned
        # checkpoint.  Compare it now with the digest captured immediately
        # after the external EPOM-L checkpoint was loaded by the model.
        self._actor_backbone_verification = verifier()
        # ``config_path`` is the immutable config stored next to this run.  The
        # assertion therefore verifies the actual checkpoint family rather
        # than trusting a command-line radius.
        self._trace_contract = _read_r5_trace_contract(self.config_path)
        gate_override = getattr(
            self.ppo, "set_inference_learned_gate_override", None
        )
        if not callable(gate_override):
            raise RuntimeError(
                "EPOM-TraceContext actor-critic does not expose the required "
                "inference gate-override hook."
            )
        gate_override(algo_cfg.learned_gate_override)
        if int(self.tau_radius) != TRACE_RADIUS:
            raise RuntimeError(
                "Runtime tau_radius disagrees with checkpoint config: "
                f"{self.tau_radius} != {TRACE_RADIUS}"
            )
        actor_radius = getattr(self.ppo, "trace_radius", None)
        if actor_radius != TRACE_RADIUS:
            raise RuntimeError(
                "Actor observation space is not the required 11x11 trace crop: "
                f"trace_radius={actor_radius!r}"
            )
        self._context_diagnostic_steps: list[dict[str, float]] = []

    # CAAR.__init__ dispatches to this method, so milestone selection happens
    # before the checkpoint is deserialised and cannot be changed afterwards.
    def _load_checkpoint(self, checkpoint_dir, device, checkpoint_kind):
        if checkpoint_kind != "milestone":
            checkpoint = super()._load_checkpoint(
                checkpoint_dir, device, checkpoint_kind
            )
            self.loaded_checkpoint_kind = checkpoint_kind
            return checkpoint

        configured = self.algo_cfg.milestone_checkpoint
        if not configured:
            raise ValueError(
                "checkpoint_kind='milestone' requires milestone_checkpoint"
            )
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            project_root = Path(__file__).resolve().parents[1]
            weights_dir = Path(checkpoint_dir).resolve().parent
            possibilities = (
                project_root / candidate,
                weights_dir / candidate,
                Path(checkpoint_dir).resolve() / candidate,
            )
            candidate = next(
                (path for path in possibilities if path.is_file()),
                possibilities[0],
            )
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(
                f"Missing requested milestone checkpoint: {candidate}"
            )
        if candidate.suffix != ".pth":
            raise ValueError(
                f"Milestone checkpoint must be a .pth file: {candidate}"
            )

        self.checkpoint_path = candidate
        checkpoint, self.checkpoint_sha256 = self._load_checkpoint_path(
            candidate, device, "milestone"
        )
        self.loaded_checkpoint_kind = "milestone"
        return checkpoint

    def after_reset(self):
        super().after_reset()
        self._context_diagnostic_steps = []

    def _add_exact_free_mask(self, observations, positions):
        """Add the exact 11x11 free-cell mask used during training.

        ``AcoState`` already owns the padded global obstacle mask, so using its
        crop routine also marks cells outside the map as unavailable.  This is
        intentionally exact: zero trace alone cannot distinguish an obstacle
        from a free cell that has not yet been visited.
        """

        for observation, (row, col) in zip(observations, positions):
            free = self.aco.extract_local_free_mask(
                int(row), int(col), TRACE_RADIUS
            )
            if free.shape != (TRACE_SIZE, TRACE_SIZE):
                raise RuntimeError(
                    "Trace mask escaped the radius-5 observation contract: "
                    f"shape={free.shape}"
                )
            observation["tau_free_mask"] = free[np.newaxis, ...].astype(
                np.float32, copy=False
            )

    def act(self, observations, rewards=None, dones=None, infos=None):
        del rewards, dones, infos
        observations = deepcopy(observations)
        num_agents = len(observations)

        self.grid_memory.update(observations)
        self.grid_memory.modify_observation(
            observations, self.grid_memory_radius
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
                "Context trace state is not initialised. Call set_env() after "
                "env.reset() and before act()."
            )

        positions = self._global_positions()
        self.aco.observe_for_inference(
            observations,
            positions=positions,
            raw_tau=bool(self._trace_contract["tau_raw"]),
            radius=TRACE_RADIUS,
        )
        self._add_exact_free_mask(observations, positions)
        for observation in observations:
            if observation["tau"].shape != (1, TRACE_SIZE, TRACE_SIZE):
                raise RuntimeError(
                    "Inference trace is not exactly [1,11,11]: "
                    f"{observation['tau'].shape}"
                )
            if observation["tau_free_mask"].shape != (
                1,
                TRACE_SIZE,
                TRACE_SIZE,
            ):
                raise RuntimeError(
                    "Inference free mask is not exactly [1,11,11]: "
                    f"{observation['tau_free_mask'].shape}"
                )
        self._last_augmented_observations = deepcopy(observations)

        with torch.no_grad():
            obs_torch = TensorDict(
                {
                    key: torch.from_numpy(
                        np.stack([obs[key] for obs in observations])
                    )
                    .to(self.device)
                    .float()
                    for key in observations[0]
                }
            )
            obs_torch = prepare_and_normalize_obs(self.ppo, obs_torch)
            policy_outputs = self.ppo(obs_torch, self.rnn_states)
            self.rnn_states = policy_outputs["new_rnn_states"]

            diagnostics = self.ppo.context_diagnostics()
            if diagnostics:
                self._context_diagnostic_steps.append(
                    {key: float(value) for key, value in diagnostics.items()}
                )

        return policy_outputs["actions"].detach().cpu().numpy()

    def get_action_correction_stats(self):
        if not self._context_diagnostic_steps:
            return {}
        keys = set.intersection(
            *(set(step) for step in self._context_diagnostic_steps)
        )
        return {
            f"context_{key}": float(
                np.mean([step[key] for step in self._context_diagnostic_steps])
            )
            for key in sorted(keys)
        }

    def get_model_provenance(self):
        model_provenance = {}
        provider = getattr(self.ppo, "checkpoint_provenance", None)
        if callable(provider):
            model_provenance = deepcopy(provider())
        return {
            "method": "EPOM-TraceContext",
            "weights_path": str(Path(self.algo_cfg.path_to_weights).resolve()),
            "checkpoint_kind": getattr(
                self, "loaded_checkpoint_kind", self.algo_cfg.checkpoint_kind
            ),
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_path": str(self.config_path),
            "config_sha256": self.config_sha256,
            "trace_contract": deepcopy(self._trace_contract),
            "model": model_provenance,
        }

    def last_switch_context(self):
        return None

    def get_name(self):
        return f"EPOM-TraceContext({self.checkpoint_path.name})"


__all__ = [
    "EPOMTraceContext",
    "EPOMTraceContextConfig",
    "TRACE_RADIUS",
    "TRACE_SIZE",
]
