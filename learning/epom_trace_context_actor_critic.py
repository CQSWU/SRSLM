"""Frozen EPOM-L support for the retained paper CAAR model.

The historical class name remains the parent of EPOMTraceMultiplierActorCritic.
It provides checkpoint validation, frozen-backbone identity, training-mode
control and shared diagnostics; it no longer constructs a standalone contextual
residual model. The concrete paper architecture is defined in the multiplier
module. Pure Direct-rule helpers remain available for isolated diagnostics.
"""

from __future__ import annotations

import hashlib
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
    """Shared frozen-base support; construct the concrete paper CAAR class."""

    NUM_ACTIONS = 5
    BASE_ARCH_KEYS = (
        "hidden_size",
        "pogema_encoder_num_filters",
        "pogema_encoder_num_res_blocks",
        "encoder_extra_fc_layers",
        "normalize_input",
        "normalize_input_keys",
    )
    # Concrete paper CAAR declares the exact permitted trainable prefixes.
    TRAINABLE_PREFIXES: tuple[str, ...] = ()
    ACTOR_BACKBONE_MODULE_NAMES = (
        "encoder",
        "core",
        "decoder",
        "action_parameterization",
    )

    def __init__(self, model_factory, obs_space, action_space, cfg):
        raise TypeError(
            "EPOMTraceContextActorCritic is frozen-base support only. "
            "Construct EPOMTraceMultiplierActorCritic for the paper CAAR model."
        )

    # --------------------------------------------------------- frozen base

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _update_digest_field(digest, value: str) -> None:
        """Add one unambiguous UTF-8 field to a deterministic digest."""

        payload = value.encode("utf-8")
        digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
        digest.update(payload)

    def _actor_backbone_modules(self) -> tuple[tuple[str, nn.Module], ...]:
        """Return only modules that can change the frozen actor policy.

        The critic and return normalizer are deliberately absent.  The
        observation normalizer is included when present because its buffers
        transform actor inputs before the encoder.
        """

        modules: list[tuple[str, nn.Module]] = []
        for name in self.ACTOR_BACKBONE_MODULE_NAMES:
            module = getattr(self, name, None)
            if not isinstance(module, nn.Module):
                raise RuntimeError(
                    f"Frozen actor backbone is missing module {name!r}."
                )
            modules.append((name, module))

        obs_normalizer = getattr(self, "obs_normalizer", None)
        if obs_normalizer is not None:
            if not isinstance(obs_normalizer, nn.Module):
                raise RuntimeError(
                    "obs_normalizer exists but is not a torch module; its "
                    "actor-input state cannot be verified."
                )
            modules.append(("obs_normalizer", obs_normalizer))
        return tuple(modules)

    def _actor_backbone_tensor_sha256(self) -> str:
        """Hash actor-backbone tensor names, metadata, and exact bytes.

        Length-prefixed fields make the stream unambiguous.  Tensor bytes are
        read from a contiguous CPU uint8 view, so the digest does not depend on
        the current accelerator or on ``torch.save`` serialization details.
        """

        digest = hashlib.sha256()
        self._update_digest_field(digest, "EPOM actor backbone tensor digest v1")
        for module_name, module in self._actor_backbone_modules():
            self._update_digest_field(digest, module_name)
            state = module.state_dict()
            self._update_digest_field(digest, str(len(state)))
            for tensor_name in sorted(state):
                tensor = state[tensor_name]
                if not isinstance(tensor, torch.Tensor):
                    raise RuntimeError(
                        "Actor backbone state contains a non-tensor entry: "
                        f"{module_name}.{tensor_name}"
                    )
                if tensor.layout != torch.strided:
                    raise RuntimeError(
                        "Actor backbone state contains an unsupported non-dense "
                        f"tensor: {module_name}.{tensor_name} ({tensor.layout})"
                    )
                self._update_digest_field(digest, tensor_name)
                self._update_digest_field(digest, str(tensor.dtype))
                self._update_digest_field(
                    digest, ",".join(str(value) for value in tensor.shape)
                )
                raw = (
                    tensor.detach()
                    .to(device="cpu")
                    .contiguous()
                    .reshape(-1)
                    .view(torch.uint8)
                    .numpy()
                    .tobytes(order="C")
                )
                digest.update(len(raw).to_bytes(8, byteorder="big", signed=False))
                digest.update(raw)
        return digest.hexdigest()

    def verify_frozen_actor_backbone(self) -> dict[str, object]:
        """Fail closed if a learned checkpoint replaced frozen actor tensors."""

        expected = getattr(
            self, "actor_backbone_tensor_sha256_expected", None
        )
        if not expected:
            raise RuntimeError(
                "No expected EPOM-L actor-backbone digest was recorded after "
                "loading the external base checkpoint."
            )
        current = self._actor_backbone_tensor_sha256()
        self.actor_backbone_tensor_sha256_current = current
        verified = current == expected
        self.actor_backbone_tensor_sha256_verified = verified
        if not verified:
            raise RuntimeError(
                "Learned checkpoint changed the frozen EPOM-L actor backbone: "
                f"expected tensor SHA256 {expected}, current {current}."
            )
        return {
            "expected": expected,
            "current": current,
            "verified": True,
        }

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
        checkpoint_path = self._latest_checkpoint(directory)

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
        training_grid = (
            getattr(self.cfg, "full_config", {})
            .get("environment", {})
            .get("grid_config", {})
        )
        training_collision = training_grid.get("collision_system")
        required_grid = {
            "on_target": "restart",
            "collision_system": training_collision,
            "obs_radius": 5,
        }
        bad_grid = {
            key: {"actual": grid.get(key), "required": value}
            for key, value in required_grid.items()
            if grid.get(key) != value
        }
        if bad_grid:
            raise RuntimeError(
                "The frozen base does not match the lifelong protocol being "
                f"trained ({training_collision!r} execution): {bad_grid}"
            )
        memory_radius = base_full.get("environment", {}).get(
            "grid_memory_obs_radius"
        )
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

        self.base_weights_dir = str(directory)
        self.base_checkpoint = checkpoint_path.name
        self.base_checkpoint_sha256 = self._sha256(checkpoint_path)
        self.base_config_sha256 = self._sha256(config_path)
        # This digest is captured immediately after the external EPOM-L state
        # is loaded.  It is a plain runtime attribute (not checkpoint state),
        # so loading the learned run cannot replace the expected value.
        self.actor_backbone_tensor_sha256_expected = (
            self._actor_backbone_tensor_sha256()
        )
        self.actor_backbone_tensor_sha256_current = (
            self.actor_backbone_tensor_sha256_expected
        )
        self.actor_backbone_tensor_sha256_verified = True

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
            raise RuntimeError(f"Context modules have no trainable parameters: {absent}")

    def train(self, mode: bool = True):
        super().train(mode)
        for module in getattr(self, "_frozen_base_modules", ()):
            module.eval()
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    # ------------------------------------------------------------- features

    @classmethod
    def center_candidate_trace(
        cls, tau: torch.Tensor, tau_free_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pure five-candidate Direct readout for diagnostics and tests.

        Args:
            tau: ``[B,1,S,S]`` raw or mean-shifted trace.  A spatially constant
                shift cancels when the five legal candidates are centred.
            tau_free_mask: exact aligned ``[B,1,S,S]`` mask, with one for
                free cells.

        Returns:
            The five legal-candidate-centred pressures and the five-cell legal
            mask, both ``[B,5]`` in POGEMA action order.
        """
        if tau.ndim != 4 or tau.shape[1] != 1:
            raise ValueError(f"Expected tau [B,1,S,S], got {tuple(tau.shape)}.")
        if tuple(tau_free_mask.shape) != tuple(tau.shape):
            raise ValueError(
                "tau_free_mask must match tau, got "
                f"{tuple(tau_free_mask.shape)} and "
                f"{tuple(tau.shape)}."
            )
        if tau.shape[-2] != tau.shape[-1] or tau.shape[-1] % 2 != 1:
            raise ValueError(f"Expected odd square tau, got {tuple(tau.shape)}.")
        centre = tau.shape[-1] // 2
        values = torch.stack(
            [tau[:, 0, centre + dx, centre + dy] for dx, dy in MOVES], dim=-1
        )
        legal = torch.stack(
            [
                tau_free_mask[:, 0, centre + dx, centre + dy]
                for dx, dy in MOVES
            ],
            dim=-1,
        ).to(tau.dtype)
        legal = (legal > 0.5).to(tau.dtype)
        count = legal.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = (values * legal).sum(dim=-1, keepdim=True) / count
        centred = (values - mean) * legal
        return centred, legal

    @staticmethod
    def _base_entropy(logits: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=-1)
        return -(
            probabilities
            * torch.log(probabilities + PRIMAL3_ENTROPY_EPS)
        ).sum(dim=-1)

    @classmethod
    def apply_direct_rule(
        cls,
        base_logits: torch.Tensor,
        centred_trace: torch.Tensor,
        entropy_threshold: float = PRIMAL3_ENTROPY_THRESHOLD,
        rule_scale: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pure entropy-gated Direct baseline for diagnostics and tests."""
        if base_logits.shape != centred_trace.shape:
            raise ValueError(
                "base_logits and centred_trace must match, got "
                f"{tuple(base_logits.shape)} and {tuple(centred_trace.shape)}."
            )
        entropy = cls._base_entropy(base_logits)
        gate = (entropy > float(entropy_threshold)).to(base_logits.dtype).unsqueeze(-1)
        direct_logits = base_logits - float(rule_scale) * gate * centred_trace
        return direct_logits, gate, entropy

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
        kl_direct_final = (
            direct_p * (direct_log_p - final_log_p)
        ).sum(dim=-1)
        gated = self.last_learned_gate.squeeze(-1) > 0.5
        if bool(gated.any()):
            gated_learned_norm = learned[gated].norm(dim=-1).mean()
        else:
            gated_learned_norm = learned.new_zeros(())
        return {
            "trace_radius": float(self.trace_radius),
            "gate_rate": float(self.last_gate.float().mean()),
            "learned_gate_rate": float(
                self.last_learned_gate.float().mean()
            ),
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

    def checkpoint_provenance(self) -> dict[str, object]:
        current_digest = self._actor_backbone_tensor_sha256()
        self.actor_backbone_tensor_sha256_current = current_digest
        expected_digest = self.actor_backbone_tensor_sha256_expected
        verified = current_digest == expected_digest
        self.actor_backbone_tensor_sha256_verified = verified
        return {
            "base_weights_dir": self.base_weights_dir,
            "base_checkpoint": self.base_checkpoint,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "base_config_sha256": self.base_config_sha256,
            "actor_backbone_tensor_sha256_expected": expected_digest,
            "actor_backbone_tensor_sha256_current": current_digest,
            "actor_backbone_tensor_sha256_verified": verified,
            "trace_size": self.trace_size,
            "trace_radius": self.trace_radius,
            "free_mask_source": self.free_mask_source,
            "rule_scale": self.rule_scale,
            "entropy_threshold": self.entropy_threshold,
            "learned_gate_mode": self.learned_gate_mode,
            "residual_cap": self.residual_cap,
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.trainable_parameters()
            ),
        }


__all__ = [
    "EPOMTraceContextActorCritic",
    "PRIMAL3_ENTROPY_THRESHOLD",
]
