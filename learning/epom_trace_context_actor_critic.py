"""Frozen EPOM-L support for the retained paper ARPE model.

The historical class name remains the parent of EPOMTraceMultiplierActorCritic.
It provides checkpoint loading, frozen-base training-mode
control and shared diagnostics; it no longer constructs a standalone contextual
residual model. The concrete paper architecture is defined in the multiplier
module. This support class cannot be selected as a second architecture.
"""

from __future__ import annotations

import json
from pathlib import Path
import torch
from sample_factory.model.actor_critic import ActorCriticSharedWeights
from torch import nn
from torch.nn.utils.rnn import PackedSequence


PRIMAL3_ENTROPY_THRESHOLD = 0.46371241
PRIMAL3_ENTROPY_EPS = 1e-10
MOVES = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


class EPOMTraceContextActorCritic(ActorCriticSharedWeights):
    """Shared frozen-base support; construct the concrete paper ARPE class."""

    NUM_ACTIONS = 5
    BASE_ARCH_KEYS = (
        "hidden_size",
        "pogema_encoder_num_filters",
        "pogema_encoder_num_res_blocks",
        "encoder_extra_fc_layers",
        "normalize_input",
        "normalize_input_keys",
    )
    # Concrete paper ARPE declares the exact permitted trainable prefixes.
    TRAINABLE_PREFIXES: tuple[str, ...] = ()

    def __init__(self, model_factory, obs_space, action_space, cfg):
        raise TypeError(
            "EPOMTraceContextActorCritic is frozen-base support only. "
            "Construct EPOMTraceMultiplierActorCritic for the paper ARPE model."
        )

    # --------------------------------------------------------- frozen base

    @staticmethod
    def _resolve_weights_dir(configured: str) -> Path:
        directory = Path(configured).expanduser()
        if not directory.is_absolute():
            directory = Path(__file__).resolve().parents[1] / directory
        return directory.resolve()

    @staticmethod
    def _latest_checkpoint(directory: Path) -> Path:
        checkpoints = sorted(
            path
            for path in (directory / "checkpoint_p0").glob("*.pth")
            if not path.name.startswith("best_")
        )
        if not checkpoints:
            raise FileNotFoundError(f"No non-best checkpoint under {directory}.")
        return checkpoints[-1]

    def _load_and_freeze_base(self, settings) -> None:
        directory = self._resolve_weights_dir(settings["epom_base_weights_path"])
        config_path = directory / "config.json"
        if not config_path.is_file():
            config_path = directory / "cfg.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"No EPOM-L config under {directory}.")
        declared_checkpoint = getattr(self.cfg, "base_checkpoint_path", None)
        checkpoint_path = (
            Path(declared_checkpoint).resolve()
            if declared_checkpoint
            else self._latest_checkpoint(directory)
        )

        serialized = json.loads(config_path.read_text(encoding="utf-8"))
        base_full = serialized.get("full_config", serialized)
        base_settings = base_full["experiment_settings"]
        mismatches = {
            key: {"training": settings.get(key), "base": base_settings.get(key)}
            for key in self.BASE_ARCH_KEYS
            if settings.get(key) != base_settings.get(key)
        }
        base_async = base_full.get("async_ppo", {})
        for key in ("use_rnn", "rnn_type", "rnn_num_layers"):
            current = getattr(self.cfg, key, None)
            expected = base_async.get(key)
            if current != expected:
                mismatches[key] = {"training": current, "base": expected}
        if mismatches:
            raise RuntimeError(
                f"EPOM-L backbone at {directory} is incompatible: {mismatches}"
            )

        grid = base_full.get("environment", {}).get("grid_config", {})
        if grid.get("obs_radius") != 5:
            raise ValueError("The EPOM base requires obs_radius=5.")
        memory_radius = base_full.get("environment", {}).get("grid_memory_obs_radius")
        if memory_radius != 7:
            raise RuntimeError(
                f"EPOM-L grid-memory radius must be 7, got {memory_radius}."
            )

        checkpoint = torch.load(
            str(checkpoint_path), map_location="cpu", weights_only=False
        )
        if "model" not in checkpoint:
            raise RuntimeError(f"Checkpoint {checkpoint_path} has no model state.")
        incompatible = self.load_state_dict(checkpoint["model"], strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                "EPOM-L checkpoint has unexpected keys: "
                f"{sorted(incompatible.unexpected_keys)}"
            )
        missing_context = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(self.TRAINABLE_PREFIXES)
        ]
        if missing_context:
            raise RuntimeError(
                "EPOM-L checkpoint left base parameters uninitialised: "
                f"{sorted(missing_context)}"
            )

        self._frozen_base_modules = tuple(
            module
            for module in (
                self.encoder,
                self.core,
                self.decoder,
                self.action_parameterization,
                self.critic_linear,
                getattr(self, "obs_normalizer", None),
                getattr(self, "returns_normalizer", None),
            )
            if module is not None
        )
        for module in self._frozen_base_modules:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def _verify_parameter_partition(self) -> None:
        trainable = [name for name, p in self.named_parameters() if p.requires_grad]
        unexpected = [
            name for name in trainable if not name.startswith(self.TRAINABLE_PREFIXES)
        ]
        if unexpected:
            raise RuntimeError(
                f"Frozen EPOM-L exposes trainable parameters: {sorted(unexpected)}"
            )
        absent = [
            prefix
            for prefix in self.TRAINABLE_PREFIXES
            if not any(name.startswith(prefix) for name in trainable)
        ]
        if absent:
            raise RuntimeError(
                f"Context modules have no trainable parameters: {absent}"
            )

    def train(self, mode: bool = True):
        super().train(mode)
        for module in getattr(self, "_frozen_base_modules", ()):
            module.eval()
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    # ------------------------------------------------------------- features

    @staticmethod
    def _base_entropy(logits: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=-1)
        return -(probabilities * torch.log(probabilities + PRIMAL3_ENTROPY_EPS)).sum(
            dim=-1
        )

    @staticmethod
    def _packed_like(reference: PackedSequence, data: torch.Tensor) -> PackedSequence:
        return PackedSequence(
            data,
            reference.batch_sizes,
            reference.sorted_indices,
            reference.unsorted_indices,
        )

    # ----------------------------------------------------------- diagnostics

    def context_diagnostics(self) -> dict[str, float]:
        if self.last_final_logits is None:
            return {}
        base = self.last_base_logits
        direct = self.last_direct_logits
        final = self.last_final_logits
        learned = self.last_learned_delta
        direct_log_p = torch.log_softmax(direct, dim=-1)
        final_log_p = torch.log_softmax(final, dim=-1)
        direct_p = direct_log_p.exp()
        kl_direct_final = (direct_p * (direct_log_p - final_log_p)).sum(dim=-1)
        gated = self.last_learned_gate.squeeze(-1) > 0.5
        if bool(gated.any()):
            gated_learned_norm = learned[gated].norm(dim=-1).mean()
        else:
            gated_learned_norm = learned.new_zeros(())
        return {
            "trace_radius": float(self.trace_radius),
            "gate_rate": float(self.last_gate.float().mean()),
            "learned_gate_rate": float(self.last_learned_gate.float().mean()),
            "base_entropy_mean": float(self.last_base_entropy.float().mean()),
            "rule_delta_norm": float(self.last_rule_delta.norm(dim=-1).mean()),
            "learned_delta_norm": float(learned.norm(dim=-1).mean()),
            "gated_learned_delta_norm": float(gated_learned_norm),
            "kl_direct_final": float(kl_direct_final.mean()),
            "argmax_flip_base_direct": float(
                (base.argmax(-1) != direct.argmax(-1)).float().mean()
            ),
            "argmax_flip_direct_final": float(
                (direct.argmax(-1) != final.argmax(-1)).float().mean()
            ),
            "residual_cap_fraction": float(
                (learned.abs() > 0.95 * self.residual_cap).float().mean()
            ),
            "candidate_trace_spread": float(
                (
                    self.last_candidate_trace.max(-1).values
                    - self.last_candidate_trace.min(-1).values
                ).mean()
            ),
            "free_candidate_fraction": float(self.last_legal_mask.float().mean()),
        }


__all__ = [
    "EPOMTraceContextActorCritic",
    "PRIMAL3_ENTROPY_THRESHOLD",
]
