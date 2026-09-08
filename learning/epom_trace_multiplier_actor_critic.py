"""Paper CAAR: a learned logit correction on the frozen EPOM-L policy.

The actor encodes the full, free-cell-mean-centred 11x11 shared trace with
Conv32, two residual blocks and FC32. It fuses that feature with the detached
EPOM-L hidden state (512) and logits (5), then predicts five corrections p.
The policy is z - g*p: g is the frozen policy's entropy gate, or one when the
checkpoint was trained without the gate. No action mask, pressure multiplier,
output clipping, or second centring operation is applied to the learned p.

An independent linear critic reads only the detached EPOM-L hidden state.
Historical module and parameter names are retained to load the selected paper
checkpoints without rewriting their tensors or configuration files.
"""

from __future__ import annotations

import math

import torch
from sample_factory.algo.utils.action_distributions import get_action_distribution
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.actor_critic import ActorCriticSharedWeights
from sample_factory.model.encoder import ResBlock
from torch import nn
from torch.nn.utils.rnn import PackedSequence

from learning.epom_trace_context_actor_critic import (
    EPOMTraceContextActorCritic,
    MOVES,
    PRIMAL3_ENTROPY_THRESHOLD,
)


PAPER_ENTROPY_FUSION_ARCHITECTURE = "paper_entropy_fusion"
HLINEAR_CRITIC_KIND = "frozen_epom_hidden_linear_512_to_1"


class _PaperTraceEncoder(nn.Module):
    """Conv32, two residual blocks, then a 32D trace embedding."""

    OUTPUT_SIZE = 32
    TRACE_SIZE = 11

    def __init__(self, cfg):
        super().__init__()
        channels = 32
        self.network = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ResBlock(cfg, channels, channels),
            ResBlock(cfg, channels, channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(channels * self.TRACE_SIZE * self.TRACE_SIZE, self.OUTPUT_SIZE),
            nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if tuple(inputs.shape[1:]) != (1, self.TRACE_SIZE, self.TRACE_SIZE):
            raise ValueError(
                "Paper Trace Encoder requires [B,1,11,11], got "
                f"{tuple(inputs.shape)}."
            )
        return self.network(inputs)


class EPOMTraceMultiplierActorCritic(EPOMTraceContextActorCritic):
    """Frozen EPOM-L with the paper's trace-fusion actor and linear critic."""

    EXPECTED_TRAINABLE_PARAMETERS = 303_846
    TRAINABLE_PREFIXES = (
        "actor_trace_encoder.",
        "trace_fusion_head.",
        "trace_multiplier_head.",
        "trace_value_head.",
    )

    @staticmethod
    def _validate_spatial_contract(
        obs_shape: tuple[int, ...],
        tau_shape: tuple[int, ...],
        tau_free_mask_shape: tuple[int, ...] | None,
    ) -> tuple[int, str]:
        """Keep the saved observation schema; the actor never reads the mask.

        Frozen EPOM-L expands its grid-memory input to 15x15. The trace stays
        11x11. The separate mask belongs to the shared observation interface,
        not to the learned correction's feature vector.
        """

        if tau_shape != (1, 11, 11):
            raise ValueError(
                "Paper CAAR requires the [1,11,11] trace crop, "
                f"got {tau_shape}."
            )
        if (
            len(obs_shape) != 3
            or obs_shape[-2] != obs_shape[-1]
            or obs_shape[-1] % 2 != 1
            or obs_shape[-1] < 11
        ):
            raise ValueError(
                "Paper CAAR requires an odd square EPOM observation of at "
                f"least 11x11, got {obs_shape}."
            )
        obs_size = int(obs_shape[-1])
        if tau_free_mask_shape is not None:
            if tau_free_mask_shape != tau_shape:
                raise ValueError("tau_free_mask must match the 11x11 trace crop.")
            return obs_size, "tau_free_mask"
        if obs_size != 11:
            raise ValueError(
                "A separate 11x11 tau_free_mask is required when EPOM's internal "
                f"observation is {obs_size}x{obs_size}."
            )
        return obs_size, "obs"

    def __init__(self, model_factory, obs_space, action_space, cfg):
        if not cfg.actor_critic_share_weights:
            raise ValueError("Paper CAAR requires shared base weights.")
        if "tau" not in obs_space.spaces:
            raise ValueError("Paper CAAR requires a tau observation.")
        if getattr(action_space, "n", None) != self.NUM_ACTIONS:
            raise ValueError(f"Expected five discrete actions, got {action_space}.")

        # Retain the audited EPOM base integration without constructing the
        # discarded contextual-residual actor and critic from the parent.
        ActorCriticSharedWeights.__init__(
            self, model_factory, obs_space, action_space, cfg
        )
        settings = cfg.full_config["experiment_settings"]
        environment = cfg.full_config["environment"]
        architecture = str(getattr(cfg, "trace_context_architecture", ""))
        if architecture != PAPER_ENTROPY_FUSION_ARCHITECTURE:
            raise ValueError(
                "Only the retained paper_entropy_fusion architecture is supported, "
                f"got {architecture!r}."
            )
        self.reweight_mode = architecture
        self.trace_size = 11
        self.trace_radius = 5
        self.trace_centre = 5
        mask_shape = (
            tuple(obs_space["tau_free_mask"].shape)
            if "tau_free_mask" in obs_space.spaces
            else None
        )
        self.obs_size, self.free_mask_source = self._validate_spatial_contract(
            tuple(obs_space["obs"].shape), tuple(obs_space["tau"].shape), mask_shape
        )
        self.core_out_size = int(self.core.get_out_size())
        self.trace_rho = float(environment.get("tau_rho", 0.1))
        if self.trace_rho != 0.1:
            raise ValueError("Paper CAAR fixes tau_rho=0.1 for reproducibility.")
        self.rule_scale = float(settings.get("trace_rule_scale", 1.0))
        self.entropy_threshold = float(
            settings.get("trace_gate_threshold", PRIMAL3_ENTROPY_THRESHOLD)
        )
        self.checkpoint_entropy_threshold = self.entropy_threshold
        self.learned_gate_mode = str(
            settings.get("trace_context_learned_gate", "entropy")
        )
        if self.learned_gate_mode not in {"entropy", "all"}:
            raise ValueError("Paper fusion training gate must be entropy or all.")
        self.inference_learned_gate_override = "checkpoint"
        self.inference_entropy_threshold_override: float | None = None
        self.critic_kind = HLINEAR_CRITIC_KIND
        self.critic_uses_trace = False
        # This legacy diagnostic threshold is not a bound on the learned p.
        self.residual_cap = 2.0

        # Keep construction order and subsequent .apply order unchanged so
        # identical seeds consume identical RNG streams before training.
        self.actor_trace_encoder = _PaperTraceEncoder(cfg)
        self._build_critic_modules()
        self.trace_fusion_head = nn.Sequential(
            nn.Linear(
                _PaperTraceEncoder.OUTPUT_SIZE
                + self.core_out_size
                + self.NUM_ACTIONS,
                256,
            ),
            nn.ReLU(),
        )
        self.trace_multiplier_head = nn.Sequential(nn.Linear(256, self.NUM_ACTIONS))
        for module in self._context_modules():
            module.apply(self.initialize_weights)
        nn.init.zeros_(self.trace_multiplier_head[-1].weight)
        nn.init.zeros_(self.trace_multiplier_head[-1].bias)

        self._load_and_freeze_base(settings)
        self._verify_parameter_partition()
        self.expected_trainable_parameters = self.EXPECTED_TRAINABLE_PARAMETERS
        trainable_count = sum(p.numel() for p in self.trainable_parameters())
        if trainable_count != self.expected_trainable_parameters:
            raise RuntimeError(
                "Unexpected paper CAAR trainable parameter count: "
                f"{trainable_count} != {self.expected_trainable_parameters}."
            )
        self._verify_zero_actor_output()

        self.actor_trace_embedding_size = self.actor_trace_encoder.OUTPUT_SIZE
        self.critic_trace_embedding_size = 0
        self.trace_embedding_size = self.actor_trace_embedding_size
        # The three five-action diagnostic slots retain the saved recurrent
        # training interface; only actor_trace enters the learned trace head.
        self.head_extra_size = 3 * self.NUM_ACTIONS + self.actor_trace_embedding_size
        for name in (
            "last_base_logits",
            "last_direct_logits",
            "last_final_logits",
            "last_rule_delta",
            "last_learned_delta",
            "last_gate",
            "last_learned_gate",
            "last_base_entropy",
            "last_candidate_trace",
            "last_learned_trace",
            "last_legal_mask",
            "last_values",
            "last_multipliers",
        ):
            setattr(self, name, None)

    def set_inference_learned_gate_override(self, mode: str) -> None:
        """Select the effective gate without changing saved model weights."""

        if mode not in {"checkpoint", "all"}:
            raise ValueError(
                "Inference learned-gate override must be 'checkpoint' or 'all'."
            )
        self.inference_learned_gate_override = mode

    def set_inference_entropy_threshold_override(
        self, threshold: float | None
    ) -> None:
        """Override only the inference threshold, never checkpoint weights."""

        if threshold is None:
            self.inference_entropy_threshold_override = None
            return
        threshold = float(threshold)
        if not math.isfinite(threshold) or not 0.0 <= threshold <= math.log(
            self.NUM_ACTIONS
        ):
            raise ValueError(
                "Entropy-threshold override must be finite and in [0, log(5)]."
            )
        self.inference_entropy_threshold_override = threshold

    def effective_entropy_threshold(self) -> float:
        override = self.inference_entropy_threshold_override
        return self.entropy_threshold if override is None else float(override)

    def apply_effective_paper_correction(
        self,
        base_logits: torch.Tensor,
        raw_correction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Use the training gate unless the inference ablation opens it."""

        if (
            self.learned_gate_mode != "all"
            and self.inference_learned_gate_override != "all"
        ):
            return self.apply_paper_entropy_correction_rule(
                base_logits,
                raw_correction,
                entropy_threshold=self.effective_entropy_threshold(),
            )
        entropy = self._base_entropy(base_logits)
        gate = torch.ones_like(entropy).unsqueeze(-1)
        learned_delta = -raw_correction
        return base_logits + learned_delta, learned_delta, gate, entropy

    def _resolved_critic_kind(self) -> str:
        return HLINEAR_CRITIC_KIND

    def _critic_reads_trace(self) -> bool:
        return False

    def _build_critic_modules(self) -> None:
        if self.core_out_size != 512:
            raise ValueError(
                "The hlinear critic requires the official 512D EPOM hidden."
            )
        self.trace_value_head = nn.Linear(self.core_out_size, 1)

    def _context_modules(self) -> tuple[nn.Module, ...]:
        # This order is part of reproducible initialisation, not forward order.
        return (
            self.actor_trace_encoder,
            self.trace_multiplier_head,
            self.trace_value_head,
            self.trace_fusion_head,
        )

    def _verify_zero_actor_output(self) -> None:
        output = self.trace_multiplier_head[-1]
        if torch.count_nonzero(output.weight).item() != 0:
            raise RuntimeError("Correction output weight is not exactly zero.")
        if output.bias is not None and torch.count_nonzero(output.bias).item() != 0:
            raise RuntimeError("Correction output bias is not exactly zero.")

    def forward_head(self, normalized_obs_dict):
        with torch.no_grad():
            base_context = self.encoder(normalized_obs_dict)
        # AcoState has already centred all free cells in the complete crop.
        # Obstacles and padding are zero; the learned path never reads a mask.
        tau = normalized_obs_dict["tau"].float()
        centred_trace = self.centered_trace_candidates(tau)
        legal = torch.zeros_like(centred_trace)
        actor_trace = self.actor_trace_encoder(tau)
        return torch.cat(
            [base_context.detach(), centred_trace, legal, centred_trace, actor_trace],
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
        learned_trace = core_output[:, offset : offset + self.NUM_ACTIONS]
        offset += self.NUM_ACTIONS
        actor_trace = core_output[:, offset : offset + self.actor_trace_embedding_size]
        offset += self.actor_trace_embedding_size
        critic_trace = core_output[:, offset : offset + self.critic_trace_embedding_size]
        offset += self.critic_trace_embedding_size
        if offset != core_output.shape[-1]:
            raise RuntimeError(
                "Unexpected paper CAAR core width: "
                f"consumed {offset}, got {core_output.shape[-1]}."
            )
        return hidden, centred_trace, legal, learned_trace, actor_trace, critic_trace

    def _critic_values(
        self, hidden: torch.Tensor, critic_trace: torch.Tensor
    ) -> torch.Tensor:
        """Predict return without sending gradients to the frozen EPOM base."""

        if critic_trace.shape != (hidden.shape[0], 0):
            raise RuntimeError(
                f"The {HLINEAR_CRITIC_KIND} critic must not receive trace features."
            )
        return self.trace_value_head(hidden.detach()).squeeze(-1)

    @staticmethod
    def centered_trace_candidates(tau: torch.Tensor) -> torch.Tensor:
        """Sample diagnostic candidates without centring the crop a second time."""

        if tau.ndim != 4 or tuple(tau.shape[1:]) != (1, 11, 11):
            raise ValueError(
                "Centered Shared Trace Memory must have shape [B,1,11,11], "
                f"got {tuple(tau.shape)}."
            )
        centre = 5
        return torch.stack(
            [tau[:, 0, centre + dx, centre + dy] for dx, dy in MOVES], dim=-1
        )

    @staticmethod
    def compose_paper_entropy_fusion_input(
        trace_feature: torch.Tensor,
        frozen_hidden: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse exactly trace32, detached EPOM-L h512, and detached logits z5."""

        batch = trace_feature.shape[0]
        expected = ((batch, 32), (batch, 512), (batch, 5))
        actual = (
            tuple(trace_feature.shape),
            tuple(frozen_hidden.shape),
            tuple(base_logits.shape),
        )
        if actual != expected:
            raise ValueError(
                "Paper entropy fusion expects trace32+h512+z5, got "
                f"{actual}."
            )
        return torch.cat(
            [trace_feature, frozen_hidden.detach(), base_logits.detach()], dim=-1
        )

    @classmethod
    def apply_paper_entropy_correction_rule(
        cls,
        base_logits: torch.Tensor,
        raw_correction: torch.Tensor,
        entropy_threshold: float = PRIMAL3_ENTROPY_THRESHOLD,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply z' = z - g*p without clipping, masking, or output centring."""

        if (
            base_logits.ndim != 2
            or base_logits.shape[-1] != cls.NUM_ACTIONS
            or raw_correction.shape != base_logits.shape
        ):
            raise ValueError(
                "base_logits and raw_correction must both have shape [B,5]."
            )
        entropy = cls._base_entropy(base_logits)
        gate = (entropy > float(entropy_threshold)).to(
            base_logits.dtype
        ).unsqueeze(-1)
        correction = gate * raw_correction
        final_logits = base_logits - correction
        return final_logits, -correction, gate, entropy

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool):
        if isinstance(core_output, PackedSequence):
            raise TypeError("Sample Factory must unpack PackedSequence before tail.")
        (
            hidden,
            centred_trace,
            legal,
            learned_trace,
            actor_trace,
            critic_trace,
        ) = self._split_core(core_output)
        with torch.no_grad():
            decoder_output = self.decoder(hidden)
            base_logits, _ = self.action_parameterization(decoder_output)
        base_logits = base_logits.detach()
        values = self._critic_values(hidden, critic_trace)
        result = TensorDict(values=values)
        self.last_values = values.detach()
        if values_only:
            return result

        fusion_input = self.compose_paper_entropy_fusion_input(
            actor_trace, hidden, base_logits
        )
        actor_input = self.trace_fusion_head(fusion_input)
        raw_correction = self.trace_multiplier_head(actor_input)
        final_logits, learned_delta, gate, entropy = self.apply_effective_paper_correction(
            base_logits, raw_correction
        )

        # For this architecture the learned correction is the entire
        # reweighting operation. Existing diagnostics call the unchanged base
        # output direct_logits, and the applied p last_multipliers.
        self.last_base_logits = base_logits.detach()
        self.last_direct_logits = base_logits.detach()
        self.last_final_logits = final_logits.detach()
        self.last_rule_delta = (base_logits - base_logits).detach()
        self.last_learned_delta = learned_delta.detach()
        self.last_gate = gate.detach()
        self.last_learned_gate = gate.detach()
        self.last_base_entropy = entropy.detach()
        self.last_candidate_trace = centred_trace.detach()
        self.last_learned_trace = learned_trace.detach()
        self.last_legal_mask = legal.detach()
        self.last_multipliers = (gate * raw_correction).detach()
        self.last_action_distribution = get_action_distribution(
            self.action_space, final_logits
        )
        result["action_logits"] = final_logits
        self._maybe_sample_actions(sample_actions, result)
        return result

    def context_diagnostics(self) -> dict[str, float]:
        result = super().context_diagnostics()
        # The paper actor never consumes a candidate mask. Do not report a
        # fabricated legal-action fraction from its unused diagnostic slot.
        result.pop("free_candidate_fraction", None)
        if self.last_multipliers is not None:
            correction = self.last_multipliers.float()
            result.update(
                {
                    "logit_correction_mean": float(correction.mean()),
                    "logit_correction_abs_mean": float(correction.abs().mean()),
                    "logit_correction_min": float(correction.min()),
                    "logit_correction_max": float(correction.max()),
                    "architecture_trainable_parameters": float(
                        sum(p.numel() for p in self.trainable_parameters())
                    ),
                }
            )
        return result

    def checkpoint_provenance(self) -> dict[str, object]:
        result = super().checkpoint_provenance()
        trace_parameters = sum(
            p.numel() for p in self.actor_trace_encoder.parameters() if p.requires_grad
        )
        fusion_parameters = sum(
            p.numel() for p in self.trace_fusion_head.parameters() if p.requires_grad
        )
        actor_parameters = sum(
            p.numel() for p in self.trace_multiplier_head.parameters() if p.requires_grad
        )
        critic_parameters = sum(
            p.numel() for p in self.trace_value_head.parameters() if p.requires_grad
        )
        all_state_override = self.inference_learned_gate_override == "all"
        gate_disabled = all_state_override or self.learned_gate_mode == "all"
        result.update(
            {
                "trace_architecture": (
                    "paper_entropy_conv_direct_correction_centered_P_h_z_v3"
                ),
                "actor_inputs": [
                    "full_crop_centered_trace_1x11x11",
                    "frozen_epom_recurrent_hidden_512",
                    "frozen_epom_base_logits_5",
                ],
                "actor_output": "five_direct_logit_corrections_p",
                "actor_uses_epom_hidden": True,
                "actor_uses_base_logits": True,
                "actor_uses_raw_base_logits": True,
                "actor_uses_base_policy_probabilities": False,
                "actor_uses_base_policy_entropy": False,
                "actor_input_features": 549,
                "actor_hidden_features": 256,
                "actor_trace_encoder": (
                    "conv32_3x3_two_residual_blocks_flatten_fc32_relu"
                ),
                "actor_trace_embedding_features": 32,
                "actor_receives_candidate_pressure": False,
                "actor_receives_free_mask_tensor": False,
                "actor_critic_share_trace_trunk": False,
                "feature_fusion": "linear549_256_relu",
                "trace_encoder_parameters": trace_parameters,
                "feature_fusion_parameters": fusion_parameters,
                "actor_head_parameters": actor_parameters,
                "critic_head_parameters": critic_parameters,
                "actor_trainable_parameters": (
                    trace_parameters + fusion_parameters + actor_parameters
                ),
                "critic_trainable_parameters": critic_parameters,
                "critic_architecture": HLINEAR_CRITIC_KIND,
                "critic_inputs": ["frozen_epom_hidden_512"],
                "critic_uses_epom_hidden": True,
                "critic_uses_trace": False,
                "critic_backpropagates_to_epom": False,
                "learned_gate_mode": (
                    "all_states_trained"
                    if self.learned_gate_mode == "all"
                    else "all_states_inference_override"
                    if all_state_override
                    else "raw_shannon_entropy_hard_gate"
                ),
                "checkpoint_learned_gate_mode": self.learned_gate_mode,
                "inference_learned_gate_override": self.inference_learned_gate_override,
                "entropy_threshold": self.effective_entropy_threshold(),
                "checkpoint_entropy_threshold": self.checkpoint_entropy_threshold,
                "inference_entropy_threshold_override": (
                    self.inference_entropy_threshold_override
                ),
                "base_policy_entropy_normalization": "none",
                "learned_network_uses_free_mask": False,
                "learned_residual_uses_free_mask": False,
                "correction_action_order": ["wait", "up", "down", "left", "right"],
                "logit_rule": (
                    "z_prime_equals_z_minus_p"
                    if gate_disabled
                    else "z_prime_equals_z_minus_entropy_gated_p"
                ),
                "correction_source": "feature_fusion_linear_output_5",
                "correction_centering": "none",
                "correction_bound": "unbounded_logit_space",
                "zero_initial_correction": True,
                "entropy_gate_applies_to_correction": not gate_disabled,
                "trace_input": "full_11x11_free_cell_mean_centered_trace",
                "trace_obstacles_and_padding": "zero",
                # Retain these legacy report fields for existing audit files.
                # The correction_source/bound fields above describe the actor.
                "multiplier_gate": "same_entropy_gate_as_direct",
                "learned_pressure": "plain_five_action_centered_raw_trace",
                "learned_residual_centering": "not_applicable",
                "learned_residual_bound": "not_applicable",
                "trace_flatten_order": "not_applicable",
            }
        )
        return result


__all__ = [
    "EPOMTraceMultiplierActorCritic",
    "HLINEAR_CRITIC_KIND",
    "PAPER_ENTROPY_FUSION_ARCHITECTURE",
    "_PaperTraceEncoder",
]
