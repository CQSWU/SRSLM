"""Contextual trace residual on a frozen lifelong EPOM policy.

This model deliberately starts from the parameter-free entropy-gated Direct
policy and only learns an *additional* bounded action-logit residual::

    z_direct = z_epom - 1[H(z_epom) > eta] * centre_legal(tau_candidates)
    z_final  = z_direct + g_learn * delta_phi

The fixed Direct term is always entropy-gated. ``g_learn`` is independently
configured as the same entropy gate or as one on every state. Both choices have
the identical Direct starting policy because the learned actor output is zero.

The last actor layer is exactly zero-initialised, therefore ``z_final`` equals
``z_direct`` bit for bit before the first optimiser update.  The lifelong EPOM
encoder, GRU, decoder, actor and critic are loaded from a verified checkpoint
and frozen.  The learned actor consumes the frozen 512-dimensional GRU feature
as well as a spatial trace context; the independent learned critic has its own
spatial encoder so critic gradients cannot suppress the actor's trace path.

The spatial branch uses two channels: scaled trace and a free-cell mask.  When
the trace crop and EPOM observation have the same size, the mask is constructed
from the obstacle channel of ``obs``.  A wider trace crop (for example 31x31
around EPOM's 15x15 observation) contains cells whose obstacle state cannot be
inferred from ``tau``: a free unvisited cell and an obstacle both have zero
trace.  In that case the environment must provide an exact ``tau_free_mask``
observation with shape ``[1, S, S]``.  The model fails closed rather than
silently treating unknown cells as free.  No trace radius is hard-coded; every
odd square crop is supported when an aligned free mask is available.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import torch
from sample_factory.algo.utils.action_distributions import get_action_distribution
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.actor_critic import ActorCriticSharedWeights
from torch import nn
from torch.nn.utils.rnn import PackedSequence


PRIMAL3_ENTROPY_THRESHOLD = 0.46371241
PRIMAL3_ENTROPY_EPS = 1e-10
MOVES = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


class _SpatialTraceEncoder(nn.Module):
    """Small radius-agnostic encoder that retains coarse spatial layout."""

    def __init__(self, filters: int, embedding_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(2, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(filters, filters, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(filters, filters, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(filters * 4 * 4, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.LeakyReLU(negative_slope=0.1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class EPOMTraceContextActorCritic(ActorCriticSharedWeights):
    """Frozen EPOM-L + exact Direct rule + contextual learned residual."""

    NUM_ACTIONS = 5
    BASE_ARCH_KEYS = (
        "hidden_size",
        "pogema_encoder_num_filters",
        "pogema_encoder_num_res_blocks",
        "encoder_extra_fc_layers",
        "normalize_input",
        "normalize_input_keys",
    )
    TRAINABLE_PREFIXES = (
        "actor_trace_encoder.",
        "critic_trace_encoder.",
        "actor_hidden_projection.",
        "critic_hidden_projection.",
        "context_actor.",
        "context_critic.",
    )
    ACTOR_BACKBONE_MODULE_NAMES = (
        "encoder",
        "core",
        "decoder",
        "action_parameterization",
    )

    def __init__(self, model_factory, obs_space, action_space, cfg):
        if not cfg.actor_critic_share_weights:
            raise ValueError("Contextual EPOM trace requires shared base weights.")
        if "tau" not in obs_space.spaces:
            raise ValueError("Contextual EPOM trace requires a tau observation.")
        if getattr(action_space, "n", None) != self.NUM_ACTIONS:
            raise ValueError(f"Expected five discrete actions, got {action_space}.")

        super().__init__(model_factory, obs_space, action_space, cfg)
        settings = cfg.full_config["experiment_settings"]
        environment = cfg.full_config["environment"]

        tau_shape = tuple(obs_space["tau"].shape)
        if (
            len(tau_shape) != 3
            or tau_shape[0] != 1
            or tau_shape[1] != tau_shape[2]
            or tau_shape[1] % 2 != 1
        ):
            raise ValueError(
                "Expected an odd square tau crop [1, S, S], got "
                f"{tau_shape}."
            )
        self.trace_size = int(tau_shape[1])
        self.trace_radius = self.trace_size // 2
        self.trace_centre = self.trace_radius

        obs_shape = tuple(obs_space["obs"].shape)
        if len(obs_shape) != 3 or obs_shape[0] < 1:
            raise ValueError(f"Expected EPOM matrix observation [C,H,W], got {obs_shape}.")
        self.obs_size = int(obs_shape[-1])
        if obs_shape[-2] != self.obs_size or self.obs_size % 2 != 1:
            raise ValueError(f"Expected an odd square EPOM observation, got {obs_shape}.")

        if "tau_free_mask" in obs_space.spaces:
            mask_shape = tuple(obs_space["tau_free_mask"].shape)
            if mask_shape != tau_shape:
                raise ValueError(
                    "tau_free_mask must have the same shape as tau, got "
                    f"{mask_shape} and {tau_shape}."
                )
            self.free_mask_source = "tau_free_mask"
        elif self.obs_size == self.trace_size:
            self.free_mask_source = "obs"
        else:
            raise ValueError(
                f"The {self.trace_size}x{self.trace_size} tau crop extends beyond "
                f"the {self.obs_size}x{self.obs_size} EPOM observation. The outer "
                "free-cell mask cannot be inferred from tau because both obstacles "
                "and unvisited free cells have zero trace. Add an exact observation "
                "named 'tau_free_mask' with shape [1,S,S]; guessing or padding the "
                "mask is intentionally unsupported."
            )

        self.core_out_size = int(self.core.get_out_size())
        self.trace_rho = float(environment.get("tau_rho", 0.1))
        if not 0.0 < self.trace_rho <= 1.0:
            raise ValueError("environment.tau_rho must be in (0,1].")
        self.rule_scale = float(settings.get("trace_rule_scale", 1.0))
        self.entropy_threshold = float(
            settings.get("trace_gate_threshold", PRIMAL3_ENTROPY_THRESHOLD)
        )
        self.learned_gate_mode = str(
            settings.get("trace_context_learned_gate", "entropy")
        ).lower()
        if self.learned_gate_mode not in ("entropy", "all"):
            raise ValueError(
                "trace_context_learned_gate must be 'entropy' or 'all', got "
                f"{self.learned_gate_mode!r}."
            )
        self.residual_cap = float(settings.get("trace_context_residual_cap", 2.0))
        if self.residual_cap <= 0.0:
            raise ValueError("trace_context_residual_cap must be positive.")

        filters = int(settings.get("trace_context_filters", 32))
        embedding_size = int(settings.get("trace_context_embedding_size", 128))
        projected_size = int(settings.get("trace_context_hidden_projection", 128))
        fusion_size = int(settings.get("trace_context_fusion_size", 256))
        head_size = int(settings.get("trace_context_head_size", 128))
        for name, value in (
            ("filters", filters),
            ("embedding_size", embedding_size),
            ("projected_size", projected_size),
            ("fusion_size", fusion_size),
            ("head_size", head_size),
        ):
            if value < 1:
                raise ValueError(f"{name} must be positive, got {value}.")

        # Actor and critic encoders are intentionally separate.  The old stage-1
        # experiment shared them and the actor never amplified the trace path.
        self.trace_embedding_size = embedding_size
        self.actor_trace_encoder = _SpatialTraceEncoder(filters, embedding_size)
        self.critic_trace_encoder = _SpatialTraceEncoder(filters, embedding_size)
        self.actor_hidden_projection = nn.Sequential(
            nn.Linear(self.core_out_size, projected_size),
            nn.LayerNorm(projected_size),
            nn.LeakyReLU(negative_slope=0.1),
        )
        self.critic_hidden_projection = nn.Sequential(
            nn.Linear(self.core_out_size, projected_size),
            nn.LayerNorm(projected_size),
            nn.LeakyReLU(negative_slope=0.1),
        )

        # Base centred logits (5), probabilities (5), entropy (1), top-two
        # margin (1), Direct candidate pressure (5), legality (5).
        policy_features = 2 * self.NUM_ACTIONS + 2 + 2 * self.NUM_ACTIONS
        fusion_input = projected_size + embedding_size + policy_features
        self.context_actor = nn.Sequential(
            nn.Linear(fusion_input, fusion_size),
            nn.LayerNorm(fusion_size),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(fusion_size, head_size),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(head_size, self.NUM_ACTIONS),
        )
        self.context_critic = nn.Sequential(
            nn.Linear(fusion_input, fusion_size),
            nn.LayerNorm(fusion_size),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(fusion_size, head_size),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(head_size, 1),
        )

        for module in self._context_modules():
            module.apply(self.initialize_weights)
        # This is the Direct-policy guarantee: no learned correction at step 0.
        nn.init.zeros_(self.context_actor[-1].weight)
        nn.init.zeros_(self.context_actor[-1].bias)

        self._load_and_freeze_base(settings)
        self._verify_parameter_partition()
        self._verify_zero_actor_output()

        self.head_extra_size = 2 * self.NUM_ACTIONS + 2 * embedding_size
        for name in (
            "last_base_logits",
            "last_direct_logits",
            "last_final_logits",
            "last_rule_delta",
            "last_learned_delta",
            "last_gate",
            "last_learned_gate",
            "last_base_entropy",
            "last_values",
            "last_candidate_trace",
            "last_legal_mask",
        ):
            setattr(self, name, None)

    # --------------------------------------------------------- construction

    def _context_modules(self) -> tuple[nn.Module, ...]:
        return (
            self.actor_trace_encoder,
            self.critic_trace_encoder,
            self.actor_hidden_projection,
            self.critic_hidden_projection,
            self.context_actor,
            self.context_critic,
        )

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
        required_grid = {
            "on_target": "restart",
            "collision_system": "block_both",
            "obs_radius": 5,
        }
        bad_grid = {
            key: {"actual": grid.get(key), "required": value}
            for key, value in required_grid.items()
            if grid.get(key) != value
        }
        if bad_grid:
            raise RuntimeError(
                "The base is not the required lifelong block_both EPOM-L: "
                f"{bad_grid}"
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

    def _verify_zero_actor_output(self) -> None:
        output = self.context_actor[-1]
        if not torch.count_nonzero(output.weight).item() == 0:
            raise RuntimeError("Context actor output weight is not exactly zero.")
        if output.bias is not None and not torch.count_nonzero(output.bias).item() == 0:
            raise RuntimeError("Context actor output bias is not exactly zero.")

    def train(self, mode: bool = True):
        super().train(mode)
        for module in getattr(self, "_frozen_base_modules", ()):
            module.eval()
        return self

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    # ------------------------------------------------------------- features

    def _aligned_free_mask(self, observations: TensorDict) -> torch.Tensor:
        if self.free_mask_source == "tau_free_mask":
            free = observations["tau_free_mask"]
        else:
            free = (observations["obs"][:, 0:1] <= 0.5).to(
                observations["tau"].dtype
            )
        if tuple(free.shape[1:]) != (1, self.trace_size, self.trace_size):
            raise RuntimeError(
                "Runtime free-mask shape differs from tau: "
                f"{tuple(free.shape)} versus {tuple(observations['tau'].shape)}."
            )
        # External masks are defined as 1=free, 0=obstacle/outside.
        return (free > 0.5).to(observations["tau"].dtype)

    @classmethod
    def center_candidate_trace(
        cls, tau: torch.Tensor, tau_free_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pure Direct-rule readout used by runtime and CPU-only tests.

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

    def _candidate_trace_and_legality(
        self, tau: torch.Tensor, free: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.center_candidate_trace(tau, free)

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
        """Pure entropy-gated Direct baseline used for exact-start tests."""
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

    @staticmethod
    def _centre_legal(
        values: torch.Tensor, legal: torch.Tensor
    ) -> torch.Tensor:
        count = legal.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = (values * legal).sum(dim=-1, keepdim=True) / count
        return (values - mean) * legal

    def _policy_features(
        self,
        base_logits: torch.Tensor,
        centred_trace: torch.Tensor,
        legal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        centred_logits = base_logits - base_logits.mean(dim=-1, keepdim=True)
        probabilities = torch.softmax(base_logits, dim=-1)
        entropy = self._base_entropy(base_logits)
        top_two = torch.topk(base_logits, k=2, dim=-1).values
        margin = top_two[:, :1] - top_two[:, 1:2]
        features = torch.cat(
            [
                centred_logits,
                probabilities,
                entropy.unsqueeze(-1),
                margin,
                centred_trace,
                legal,
            ],
            dim=-1,
        )
        return features, probabilities, entropy, margin

    # --------------------------------------------------------------- forward

    def forward_head(self, normalized_obs_dict):
        with torch.no_grad():
            base_context = self.encoder(normalized_obs_dict)
        tau = normalized_obs_dict["tau"].float()
        free = self._aligned_free_mask(normalized_obs_dict)
        centred_trace, legal = self._candidate_trace_and_legality(tau, free)
        spatial_inputs = torch.cat([self.trace_rho * tau, free], dim=1)
        actor_trace = self.actor_trace_encoder(spatial_inputs)
        critic_trace = self.critic_trace_encoder(spatial_inputs)
        return torch.cat(
            [
                base_context.detach(),
                centred_trace,
                legal,
                actor_trace,
                critic_trace,
            ],
            dim=-1,
        )

    def forward_core(self, head_output, rnn_states):
        if isinstance(head_output, PackedSequence):
            context = head_output.data[:, : -self.head_extra_size]
            extras = head_output.data[:, -self.head_extra_size :]
            with torch.no_grad():
                core_output, new_states = self.core(
                    self._packed_like(head_output, context), rnn_states
                )
            combined = torch.cat([core_output.data.detach(), extras], dim=-1)
            return self._packed_like(core_output, combined), new_states

        context = head_output[:, : -self.head_extra_size]
        extras = head_output[:, -self.head_extra_size :]
        with torch.no_grad():
            core_output, new_states = self.core(context, rnn_states)
        return torch.cat([core_output.detach(), extras], dim=-1), new_states

    def _split_core(self, core_output: torch.Tensor):
        offset = self.core_out_size
        hidden = core_output[:, :offset]
        centred_trace = core_output[:, offset : offset + self.NUM_ACTIONS]
        offset += self.NUM_ACTIONS
        legal = core_output[:, offset : offset + self.NUM_ACTIONS]
        offset += self.NUM_ACTIONS
        actor_trace = core_output[:, offset : offset + self.trace_embedding_size]
        offset += actor_trace.shape[-1]
        critic_trace = core_output[:, offset:]
        return hidden, centred_trace, legal, actor_trace, critic_trace

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool):
        if isinstance(core_output, PackedSequence):
            raise TypeError(
                "Sample Factory must unpack PackedSequence before forward_tail; "
                "the base actor-critic contract does not accept a packed tail."
            )
        hidden, centred_trace, legal, actor_trace, critic_trace = self._split_core(
            core_output
        )
        with torch.no_grad():
            decoder_output = self.decoder(hidden)
            base_logits, _ = self.action_parameterization(decoder_output)
        base_logits = base_logits.detach()
        policy_features, _, entropy, _ = self._policy_features(
            base_logits, centred_trace, legal
        )

        critic_input = torch.cat(
            [
                self.critic_hidden_projection(hidden.detach()),
                critic_trace,
                policy_features,
            ],
            dim=-1,
        )
        values = self.context_critic(critic_input).squeeze(-1)
        result = TensorDict(values=values)
        self.last_values = values.detach()
        if values_only:
            return result

        rule_delta = -self.rule_scale * centred_trace
        direct_logits, gate, entropy = self.apply_direct_rule(
            base_logits,
            centred_trace,
            entropy_threshold=self.entropy_threshold,
            rule_scale=self.rule_scale,
        )

        actor_input = torch.cat(
            [
                self.actor_hidden_projection(hidden.detach()),
                actor_trace,
                policy_features,
            ],
            dim=-1,
        )
        raw_delta = self.context_actor(actor_input)
        bounded_delta = self.residual_cap * torch.tanh(
            raw_delta / self.residual_cap
        )
        learned_delta = self._centre_legal(bounded_delta, legal)
        learned_gate = (
            gate
            if self.learned_gate_mode == "entropy"
            else torch.ones_like(gate)
        )
        final_logits = direct_logits + learned_gate * learned_delta

        self.last_base_logits = base_logits.detach()
        self.last_direct_logits = direct_logits.detach()
        self.last_final_logits = final_logits.detach()
        self.last_rule_delta = rule_delta.detach()
        self.last_learned_delta = learned_delta.detach()
        self.last_gate = gate.detach()
        self.last_learned_gate = learned_gate.detach()
        self.last_base_entropy = entropy.detach()
        self.last_candidate_trace = centred_trace.detach()
        self.last_legal_mask = legal.detach()

        self.last_action_distribution = get_action_distribution(
            self.action_space, final_logits
        )
        result["action_logits"] = final_logits
        self._maybe_sample_actions(sample_actions, result)
        return result

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
