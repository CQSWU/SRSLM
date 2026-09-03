"""A small, mask-free learned refinement of entropy-gated Direct.

The frozen EPOM-L policy supplies the goal-directed logits ``z``.  The fixed
Direct baseline may use its ordinary legality-aware pressure ``p_D``.  The
learned residual is deliberately separate.  Every compact actor uses only the
raw 11x11 trace ``P`` plus selected frozen-policy summaries.  The v4 spatial
residual contract parameterizes whether the actor receives ``q`` (five frozen
action probabilities), ``H`` (normalized entropy), both, or neither.  Its
trace view is either all 121 row-major cells or the five action-centred cells.
It emits a bounded five-action logit redistribution directly.  Earlier
coefficient actors remain available for reproducibility.  No obstacle/free-cell
mask participates in either learned path::

    z_direct = z - entropy_gate * p_D
    u = actor(selected(0.1 * P, softmax(z), entropy(z) / log(5)))
    delta = centre_5(2 * tanh(u / 2))
    z_final = z_direct + delta

The actor's last layer is zero-initialised, so the initial policy is exactly
Direct.  Only the independent critic may read the frozen EPOM GRU feature.
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
from learning.spatial_residual_contract import (
    SPATIAL_RESIDUAL_ARCHITECTURES,
    SpatialResidualContract,
    resolve_spatial_residual_contract,
)


CONV_RESIDUAL_SPECS: dict[str, dict[str, object]] = {
    "conv_residual64": {
        "provenance_name": (
            "conv_residual64_unmasked_conv32_relative_preference_v5"
        ),
        "uses_relative_action_preference": True,
        "input_features": 37,
        "hidden_features": 64,
        "head_parameters": 2_757,
        "actor_parameters": 9_877,
        "total_trainable_parameters": 37_638,
    },
    "conv_residual_linear": {
        "provenance_name": (
            "conv_residual_linear_unmasked_conv32_relative_preference_v5"
        ),
        "uses_relative_action_preference": True,
        "input_features": 37,
        "hidden_features": 0,
        "head_parameters": 190,
        "actor_parameters": 7_310,
        "total_trainable_parameters": 35_071,
    },
    "conv_residual32": {
        "provenance_name": (
            "conv_residual32_unmasked_conv32_relative_preference_v6"
        ),
        "uses_relative_action_preference": True,
        "input_features": 37,
        "hidden_features": 32,
        "head_parameters": 1_381,
        "actor_parameters": 8_501,
        "total_trainable_parameters": 36_262,
    },
    "conv_residual64_p_only": {
        "provenance_name": "conv_residual64_p_only_unmasked_conv32_raw_p_v6",
        "uses_relative_action_preference": False,
        "input_features": 32,
        "hidden_features": 64,
        "head_parameters": 2_437,
        "actor_parameters": 9_557,
        "total_trainable_parameters": 37_318,
    },
    "conv_residual64_hlinear_critic": {
        "provenance_name": (
            "conv_residual64_unmasked_conv32_relative_preference_"
            "hlinear_critic_v7"
        ),
        "uses_relative_action_preference": True,
        "input_features": 37,
        "hidden_features": 64,
        "head_parameters": 2_757,
        "actor_parameters": 9_877,
        "critic_kind": "frozen_epom_hidden_linear_512_to_1",
        "critic_parameters": 513,
        "total_trainable_parameters": 10_390,
    },
    "conv_residual64_hmlp_critic": {
        "provenance_name": (
            "conv_residual64_unmasked_conv32_relative_preference_"
            "hmlp_critic_v8"
        ),
        "uses_relative_action_preference": True,
        "input_features": 37,
        "hidden_features": 64,
        "head_parameters": 2_757,
        "actor_parameters": 9_877,
        "critic_kind": "frozen_epom_hidden_projection32_mlp64_to_1",
        "critic_parameters": 18_593,
        "total_trainable_parameters": 28_470,
    },
    "conv_residual64_linear_value_critic": {
        "provenance_name": (
            "conv_residual64_unmasked_conv32_relative_preference_"
            "linear_value_critic_v8"
        ),
        "uses_relative_action_preference": True,
        "input_features": 37,
        "hidden_features": 64,
        "head_parameters": 2_757,
        "actor_parameters": 9_877,
        "critic_kind": "trace_conv32_plus_hidden_projection32_linear64_to_1",
        "critic_parameters": 23_601,
        "total_trainable_parameters": 33_478,
    },
}
CONV_RESIDUAL_ARCHITECTURES = tuple(CONV_RESIDUAL_SPECS)
LINEAR_GAIN_ARCHITECTURE = "linear_gain"
PAPER_ENTROPY_MULTIPLIER_ARCHITECTURE = "paper_entropy_multiplier"
PAPER_ENTROPY_FUSION_ARCHITECTURE = "paper_entropy_fusion"
LINEAR_GAIN_RELATIVE_PREFERENCE_SCALE = 0.1
LINEAR_GAIN_SPEC: dict[str, object] = {
    "provenance_name": (
        "linear_gain_p5_direct_centered_pressure_relative_preference_"
        "linear_value_critic_v9"
    ),
    "actor_inputs": [
        "direct_static_legal_centered_pressure_5",
        "base_relative_action_preference_mean_centered_5",
    ],
    "input_contract": "direct_centered_pressure_5_plus_relative_preference_5",
    "input_features": 10,
    "hidden_features": 0,
    "head_parameters": 55,
    "actor_parameters": 55,
    "critic_kind": "trace_conv32_plus_hidden_projection32_linear64_to_1",
    "critic_parameters": 23_601,
    "total_trainable_parameters": 23_656,
}
LEGACY_TRACE_CRITIC_KIND = "trace_conv32_plus_hidden_projection32_mlp64_to_1"
HLINEAR_CRITIC_KIND = "frozen_epom_hidden_linear_512_to_1"
HMLP_CRITIC_KIND = "frozen_epom_hidden_projection32_mlp64_to_1"
LINEAR_VALUE_CRITIC_KIND = (
    "trace_conv32_plus_hidden_projection32_linear64_to_1"
)
HLINEAR_CRITIC_ARCHITECTURES = frozenset(
    name
    for name, spec in CONV_RESIDUAL_SPECS.items()
    if spec.get("critic_kind") == HLINEAR_CRITIC_KIND
)
HIDDEN_ONLY_CRITIC_ARCHITECTURES = frozenset(
    name
    for name, spec in CONV_RESIDUAL_SPECS.items()
    if spec.get("critic_kind") in {HLINEAR_CRITIC_KIND, HMLP_CRITIC_KIND}
)


class _LightTraceEncoder(nn.Module):
    """Two convolutions and one projection; spatial layout remains explicit."""

    OUTPUT_SIZE = 32

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Flatten(),
            nn.Linear(16 * 3 * 3, self.OUTPUT_SIZE),
            nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class _PaperTraceEncoder(nn.Module):
    """Paper branch: Conv32, two residual blocks, then a 32D embedding."""

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


class _EntropyTraceEncoder(nn.Module):
    """Tiny spatial encoder used by the entropy-conditioned actors.

    Two convolutions retain enough local structure to distinguish a queue from
    a diffuse trace, while global average pooling makes the actor independent
    of any hand-picked larger crop.  The encoder reads only the raw 11x11 trace
    tensor; it has no mask, base-policy feature, or recurrent-state input.
    """

    OUTPUT_SIZE = 8

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class _FlattenTraceEncoder(nn.Module):
    """Parameter-free, row-major view of the paper's raw 11x11 trace.

    Unlike global pooling, flattening preserves which side of the agent each
    trace value came from.  The actor therefore retains directional evidence
    without adding a convolutional feature extractor or a larger trace crop.
    """

    OUTPUT_SIZE = 11 * 11

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (1, 11, 11):
            raise ValueError(
                "Spatial residual actor expects raw trace shape [B,1,11,11], "
                f"got {tuple(inputs.shape)}."
            )
        return inputs.flatten(start_dim=1)


class _CenterTraceEncoder(nn.Module):
    """Parameter-free P5 view in policy action order, without a mask."""

    OUTPUT_SIZE = 5

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (1, 11, 11):
            raise ValueError(
                "Center-P5 actor expects raw trace shape [B,1,11,11], "
                f"got {tuple(inputs.shape)}."
            )
        centre = 5
        return torch.stack(
            [inputs[:, 0, centre + dx, centre + dy] for dx, dy in MOVES],
            dim=-1,
        )


class _EmptyActorTraceEncoder(nn.Module):
    """Zero-width placeholder when Direct already supplies every actor input."""

    OUTPUT_SIZE = 0

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (1, 11, 11):
            raise ValueError(
                "Linear-Gain expects raw trace transport [B,1,11,11], "
                f"got {tuple(inputs.shape)}."
            )
        return inputs.new_empty((inputs.shape[0], 0))


def _make_spatial_residual_head(
    contract: SpatialResidualContract,
) -> nn.Sequential:
    """Build the single v4 head implementation for hidden=16 or linear=0."""

    if contract.hidden_dim == 0:
        return nn.Sequential(nn.Linear(contract.actor_input_features, 5))
    return nn.Sequential(
        nn.Linear(contract.actor_input_features, contract.hidden_dim),
        nn.ReLU(),
        nn.Linear(contract.hidden_dim, 5),
    )


def _make_conv_residual_head(architecture: str) -> nn.Sequential:
    """Build one audited v5/v6 convolutional residual head."""

    try:
        spec = CONV_RESIDUAL_SPECS[architecture]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported convolutional residual: {architecture!r}."
        ) from exc
    input_features = int(spec["input_features"])
    hidden_features = int(spec["hidden_features"])
    if hidden_features == 0:
        return nn.Sequential(nn.Linear(input_features, 5))
    return nn.Sequential(
        nn.Linear(input_features, hidden_features),
        nn.ReLU(),
        nn.Linear(hidden_features, 5),
    )


def _make_linear_gain_head() -> nn.Sequential:
    """Map Direct p_D and relative base preferences to five gains."""

    return nn.Sequential(
        nn.Linear(
            int(LINEAR_GAIN_SPEC["input_features"]),
            EPOMTraceContextActorCritic.NUM_ACTIONS,
        )
    )


class EPOMTraceMultiplierActorCritic(EPOMTraceContextActorCritic):
    """Frozen EPOM-L + Direct + a five-action trace multiplier."""

    EXPECTED_TRAINABLE_PARAMETERS = 37_638

    TRAINABLE_PREFIXES = (
        "actor_trace_encoder.",
        "critic_trace_encoder.",
        "critic_hidden_projection.",
        "trace_multiplier_head.",
        "trace_value_head.",
    )

    @staticmethod
    def _validate_spatial_contract(
        obs_shape: tuple[int, ...],
        tau_shape: tuple[int, ...],
        tau_free_mask_shape: tuple[int, ...] | None,
    ) -> tuple[int, str]:
        """Validate the frozen EPOM input and the learned 11x11 trace input.

        EPOM-L internally expands its grid-memory observation to 15x15.  That
        frozen observation is not an input to the learned multiplier.  The
        multiplier always receives only the paper's aligned 11x11 trace.  A
        separate ``tau_free_mask`` is still mandatory when spatial sizes differ
        because the unchanged fixed Direct rule needs exact candidate legality;
        the mask is never passed to a learned encoder or head.
        """

        if tau_shape != (1, 11, 11):
            raise ValueError(
                "Trace Multiplier is fixed to the paper's [1,11,11] trace crop, "
                f"got {tau_shape}."
            )
        if (
            len(obs_shape) != 3
            or obs_shape[-2] != obs_shape[-1]
            or obs_shape[-1] % 2 != 1
            or obs_shape[-1] < 11
        ):
            raise ValueError(
                "Trace Multiplier requires an odd square EPOM observation of at "
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
            raise ValueError("Trace Multiplier requires shared base weights.")
        if "tau" not in obs_space.spaces:
            raise ValueError("Trace Multiplier requires a tau observation.")
        if getattr(action_space, "n", None) != self.NUM_ACTIONS:
            raise ValueError(f"Expected five discrete actions, got {action_space}.")

        # Skip the large TraceContext constructor while retaining its audited
        # frozen-backbone and Direct-rule helpers.
        ActorCriticSharedWeights.__init__(
            self, model_factory, obs_space, action_space, cfg
        )
        settings = cfg.full_config["experiment_settings"]
        environment = cfg.full_config["environment"]

        tau_shape = tuple(obs_space["tau"].shape)
        self.trace_size = 11
        self.trace_radius = 5
        self.trace_centre = 5

        obs_shape = tuple(obs_space["obs"].shape)
        mask_shape = (
            tuple(obs_space["tau_free_mask"].shape)
            if "tau_free_mask" in obs_space.spaces
            else None
        )
        self.obs_size, self.free_mask_source = self._validate_spatial_contract(
            obs_shape, tau_shape, mask_shape
        )

        self.core_out_size = int(self.core.get_out_size())
        self.trace_rho = float(environment.get("tau_rho", 0.1))
        if self.trace_rho != 0.1:
            raise ValueError("Trace Multiplier fixes tau_rho=0.1 for reproducibility.")
        self.rule_scale = float(settings.get("trace_rule_scale", 1.0))
        self.entropy_threshold = float(
            settings.get("trace_gate_threshold", PRIMAL3_ENTROPY_THRESHOLD)
        )
        architecture = str(getattr(cfg, "trace_context_architecture", "multiplier"))
        if architecture not in {
            "multiplier",
            "coefficient",
            "scalar_gate",
            "factorized_gate",
            "entropy_scalar",
            "entropy_direction",
            "tiny_residual16",
            "linear_spatial_residual",
            LINEAR_GAIN_ARCHITECTURE,
            PAPER_ENTROPY_MULTIPLIER_ARCHITECTURE,
            PAPER_ENTROPY_FUSION_ARCHITECTURE,
            *CONV_RESIDUAL_ARCHITECTURES,
        }:
            raise ValueError(f"Unsupported light trace architecture: {architecture!r}.")
        self.reweight_mode = architecture
        architecture_spec = (
            LINEAR_GAIN_SPEC
            if architecture == LINEAR_GAIN_ARCHITECTURE
            else CONV_RESIDUAL_SPECS.get(architecture, {})
        )
        self.critic_kind = str(
            HLINEAR_CRITIC_KIND
            if architecture == PAPER_ENTROPY_FUSION_ARCHITECTURE
            else architecture_spec.get("critic_kind", LEGACY_TRACE_CRITIC_KIND)
        )
        self.uses_hlinear_critic = architecture in HLINEAR_CRITIC_ARCHITECTURES
        self.critic_uses_trace = (
            architecture not in HIDDEN_ONLY_CRITIC_ARCHITECTURES
            and architecture != PAPER_ENTROPY_FUSION_ARCHITECTURE
        )
        if not self.critic_uses_trace:
            # These critic ablations change only the critic.  Their Actor keeps
            # the exact v5 module names, shapes, input order, and residual rule.
            self.TRAINABLE_PREFIXES = tuple(
                prefix
                for prefix in self.TRAINABLE_PREFIXES
                if prefix != "critic_trace_encoder."
            )
        if self.critic_kind == HLINEAR_CRITIC_KIND:
            self.TRAINABLE_PREFIXES = tuple(
                prefix
                for prefix in self.TRAINABLE_PREFIXES
                if prefix != "critic_hidden_projection."
            )
        if architecture == PAPER_ENTROPY_FUSION_ARCHITECTURE:
            self.TRAINABLE_PREFIXES = (
                "actor_trace_encoder.",
                "trace_fusion_head.",
                "trace_multiplier_head.",
                "trace_value_head.",
            )
        self.learned_gate_mode = (
            "entropy"
            if architecture
            in {
                "multiplier",
                LINEAR_GAIN_ARCHITECTURE,
                PAPER_ENTROPY_MULTIPLIER_ARCHITECTURE,
                PAPER_ENTROPY_FUSION_ARCHITECTURE,
            }
            else "all"
        )
        # This flag is deliberately absent from the checkpoint state.  It is
        # an inference-only ablation and defaults to the checkpoint behavior.
        self.inference_learned_gate_override = "checkpoint"
        # Multipliers are bounded in (0,2).  Keep this field for shared
        # diagnostics/provenance.
        self.residual_cap = 2.0

        entropy_conditioned = architecture in {
            "entropy_scalar",
            "entropy_direction",
        }
        spatial_residual = architecture in SPATIAL_RESIDUAL_ARCHITECTURES
        self.spatial_residual_contract: SpatialResidualContract | None = (
            resolve_spatial_residual_contract(architecture, settings)
            if spatial_residual
            else None
        )
        if architecture == PAPER_ENTROPY_FUSION_ARCHITECTURE:
            self.actor_trace_encoder = _PaperTraceEncoder(cfg)
        elif architecture == PAPER_ENTROPY_MULTIPLIER_ARCHITECTURE:
            self.actor_trace_encoder = _PaperTraceEncoder(cfg)
        elif architecture == LINEAR_GAIN_ARCHITECTURE:
            self.actor_trace_encoder = _EmptyActorTraceEncoder()
            # Direct preprocessing already supplies the five pressure values.
            # The legality tensor itself is never passed to the learned head.
            self.TRAINABLE_PREFIXES = tuple(
                prefix
                for prefix in self.TRAINABLE_PREFIXES
                if prefix != "actor_trace_encoder."
            )
        elif spatial_residual:
            self.actor_trace_encoder = (
                _FlattenTraceEncoder()
                if self.spatial_residual_contract.trace_view == "P121"
                else _CenterTraceEncoder()
            )
            # This encoder is deliberately parameter-free, so it must not be
            # listed among modules that the partition audit expects to expose
            # trainable tensors.
            self.TRAINABLE_PREFIXES = tuple(
                prefix
                for prefix in self.TRAINABLE_PREFIXES
                if prefix != "actor_trace_encoder."
            )
        elif entropy_conditioned:
            self.actor_trace_encoder = _EntropyTraceEncoder()
        else:
            self.actor_trace_encoder = _LightTraceEncoder()
        if architecture == PAPER_ENTROPY_FUSION_ARCHITECTURE:
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
        else:
            self._build_critic_modules()
        actor_output_size = (
            1
            if architecture in {"scalar_gate", "entropy_scalar"}
            else self.NUM_ACTIONS
        )
        if architecture == PAPER_ENTROPY_FUSION_ARCHITECTURE:
            self.trace_multiplier_head = nn.Sequential(
                nn.Linear(256, self.NUM_ACTIONS)
            )
        elif architecture == PAPER_ENTROPY_MULTIPLIER_ARCHITECTURE:
            # h is the frozen 512D recurrent EPOM-L feature and z is its raw
            # five-action logit vector. Candidate pressure p is intentionally
            # absent: it is used only by the terminal analytic correction.
            paper_fusion_size = (
                _PaperTraceEncoder.OUTPUT_SIZE
                + self.core_out_size
                + self.NUM_ACTIONS
            )
            self.trace_multiplier_head = nn.Sequential(
                nn.Linear(paper_fusion_size, 128),
                nn.ReLU(),
                nn.Linear(128, self.NUM_ACTIONS),
            )
        elif architecture == LINEAR_GAIN_ARCHITECTURE:
            self.trace_multiplier_head = _make_linear_gain_head()
        elif spatial_residual:
            contract = self.spatial_residual_contract
            self.trace_multiplier_head = _make_spatial_residual_head(contract)
        elif architecture in CONV_RESIDUAL_ARCHITECTURES:
            self.trace_multiplier_head = _make_conv_residual_head(architecture)
        elif entropy_conditioned:
            # Exactly nine actor inputs: eight raw-trace features and one
            # normalized entropy value from the frozen base policy.
            self.trace_multiplier_head = nn.Sequential(
                nn.Linear(_EntropyTraceEncoder.OUTPUT_SIZE + 1, 16),
                nn.ReLU(),
                nn.Linear(16, actor_output_size),
            )
        else:
            self.trace_multiplier_head = nn.Sequential(
                nn.Linear(_LightTraceEncoder.OUTPUT_SIZE + self.NUM_ACTIONS, 64),
                nn.ReLU(),
                nn.Linear(64, actor_output_size),
            )
        self.trace_amplitude_head = (
            nn.Linear(_LightTraceEncoder.OUTPUT_SIZE, 1)
            if architecture == "factorized_gate"
            else None
        )
        if self.trace_amplitude_head is not None:
            self.TRAINABLE_PREFIXES = self.TRAINABLE_PREFIXES + (
                "trace_amplitude_head.",
            )
        for module in self._context_modules():
            module.apply(self.initialize_weights)
        nn.init.zeros_(self.trace_multiplier_head[-1].weight)
        nn.init.zeros_(self.trace_multiplier_head[-1].bias)
        if self.trace_amplitude_head is not None:
            nn.init.zeros_(self.trace_amplitude_head.weight)
            nn.init.zeros_(self.trace_amplitude_head.bias)

        self._load_and_freeze_base(settings)
        self._verify_parameter_partition()
        trainable_count = sum(p.numel() for p in self.trainable_parameters())
        expected_trainable = {
            "multiplier": 37_638,
            "coefficient": 37_638,
            "scalar_gate": 37_378,
            "factorized_gate": 37_671,
            "entropy_scalar": 28_602,
            "entropy_direction": 28_670,
            PAPER_ENTROPY_MULTIPLIER_ARCHITECTURE: 260_054,
            PAPER_ENTROPY_FUSION_ARCHITECTURE: 303_846,
            LINEAR_GAIN_ARCHITECTURE: int(
                LINEAR_GAIN_SPEC["total_trainable_parameters"]
            ),
            **{
                name: int(spec["total_trainable_parameters"])
                for name, spec in CONV_RESIDUAL_SPECS.items()
            },
        }.get(architecture)
        if spatial_residual:
            expected_trainable = (
                self.spatial_residual_contract.total_trainable_parameters
            )
        self.expected_trainable_parameters = expected_trainable
        if trainable_count != expected_trainable:
            raise RuntimeError(
                "Unexpected Trace Multiplier trainable parameter count: "
                f"{trainable_count} != {expected_trainable}."
            )
        self._verify_zero_actor_output()

        self.actor_trace_embedding_size = int(self.actor_trace_encoder.OUTPUT_SIZE)
        self.critic_trace_embedding_size = (
            _LightTraceEncoder.OUTPUT_SIZE if self.critic_uses_trace else 0
        )
        # Kept for older diagnostics that used one shared embedding size.
        self.trace_embedding_size = self.actor_trace_embedding_size
        self.head_extra_size = (
            3 * self.NUM_ACTIONS
            + self.actor_trace_embedding_size
            + self.critic_trace_embedding_size
        )
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
        """Select the effective learned gate without changing model weights."""

        if mode not in {"checkpoint", "all"}:
            raise ValueError(
                "Inference learned-gate override must be 'checkpoint' or 'all'."
            )
        if mode == "all" and self.reweight_mode != PAPER_ENTROPY_FUSION_ARCHITECTURE:
            raise ValueError(
                "The all-state inference ablation is defined only for the "
                "paper entropy-fusion CAAR architecture."
            )
        self.inference_learned_gate_override = mode

    def apply_effective_paper_correction(
        self,
        base_logits: torch.Tensor,
        raw_correction: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Apply either the checkpoint entropy gate or the all-state ablation."""

        if self.inference_learned_gate_override != "all":
            return self.apply_paper_entropy_correction_rule(
                base_logits,
                raw_correction,
                entropy_threshold=self.entropy_threshold,
            )
        entropy = self._base_entropy(base_logits)
        gate = torch.ones_like(entropy).unsqueeze(-1)
        learned_delta = -raw_correction
        return base_logits + learned_delta, learned_delta, gate, entropy

    def _resolved_critic_kind(self) -> str:
        """Return the explicit critic contract, with v7-test compatibility."""

        critic_kind = getattr(self, "critic_kind", None)
        if critic_kind is not None:
            return str(critic_kind)
        if getattr(self, "uses_hlinear_critic", False):
            return HLINEAR_CRITIC_KIND
        return LEGACY_TRACE_CRITIC_KIND

    def _critic_reads_trace(self) -> bool:
        return self._resolved_critic_kind() not in {
            HLINEAR_CRITIC_KIND,
            HMLP_CRITIC_KIND,
        }

    def _build_critic_modules(self) -> None:
        """Create the selected critic while leaving the v5 Actor untouched."""

        critic_kind = self._resolved_critic_kind()
        if critic_kind == HLINEAR_CRITIC_KIND:
            if self.core_out_size != 512:
                raise ValueError(
                    "The hlinear critic requires the official 512D EPOM hidden."
                )
            self.trace_value_head = nn.Linear(self.core_out_size, 1)
            return
        if self.core_out_size != 512:
            raise ValueError(
                "The compact critics require the official 512D EPOM hidden."
            )
        trace_encoder = _LightTraceEncoder()
        hidden_projection = nn.Sequential(
            nn.Linear(self.core_out_size, 32),
            nn.ReLU(),
        )
        legacy_value_head = nn.Sequential(
            nn.Linear(_LightTraceEncoder.OUTPUT_SIZE + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        if critic_kind in {HMLP_CRITIC_KIND, LINEAR_VALUE_CRITIC_KIND}:
            # Construct the complete legacy v5 critic first so global RNG has
            # advanced by exactly the same amount before the Actor head is
            # built and re-initialised.  The replacement value head is built
            # under a restored RNG state; therefore A/B have bitwise-identical
            # Actor initialisation to v5 for the same seed, while changing only
            # their registered critic modules.
            rng_state = torch.random.get_rng_state()
            try:
                replacement_value_head: nn.Module = (
                    nn.Sequential(
                        nn.Linear(32, 64),
                        nn.ReLU(),
                        nn.Linear(64, 1),
                    )
                    if critic_kind == HMLP_CRITIC_KIND
                    else nn.Linear(
                        _LightTraceEncoder.OUTPUT_SIZE + 32, 1
                    )
                )
            finally:
                torch.random.set_rng_state(rng_state)
            self.critic_hidden_projection = hidden_projection
            self.trace_value_head = replacement_value_head
            if critic_kind == LINEAR_VALUE_CRITIC_KIND:
                self.critic_trace_encoder = trace_encoder
            return
        self.critic_hidden_projection = hidden_projection
        self.critic_trace_encoder = trace_encoder
        if critic_kind != LEGACY_TRACE_CRITIC_KIND:
            raise ValueError(f"Unsupported critic kind: {critic_kind!r}.")
        self.trace_value_head = legacy_value_head

    def _context_modules(self) -> tuple[nn.Module, ...]:
        modules = [
            self.actor_trace_encoder,
            self.trace_multiplier_head,
            self.trace_value_head,
        ]
        if hasattr(self, "trace_fusion_head"):
            modules.append(self.trace_fusion_head)
        if hasattr(self, "critic_trace_encoder"):
            modules.append(self.critic_trace_encoder)
        if hasattr(self, "critic_hidden_projection"):
            modules.append(self.critic_hidden_projection)
        if self.trace_amplitude_head is not None:
            modules.append(self.trace_amplitude_head)
        return tuple(modules)

    def _verify_zero_actor_output(self) -> None:
        output = self.trace_multiplier_head[-1]
        if torch.count_nonzero(output.weight).item() != 0:
            raise RuntimeError("Multiplier output weight is not exactly zero.")
        if output.bias is not None and torch.count_nonzero(output.bias).item() != 0:
            raise RuntimeError("Multiplier output bias is not exactly zero.")

    def forward_head(self, normalized_obs_dict):
        with torch.no_grad():
            base_context = self.encoder(normalized_obs_dict)
        tau = normalized_obs_dict["tau"].float()
        if self.reweight_mode in {
            PAPER_ENTROPY_MULTIPLIER_ARCHITECTURE,
            PAPER_ENTROPY_FUSION_ARCHITECTURE,
        }:
            # AcoState has already subtracted the mean over all free cells in
            # the complete 11x11 crop. Obstacles and padding are zero. Neither
            # the actor nor the terminal rule reads a legality/free-cell mask.
            centred_trace = self.centered_trace_candidates(tau)
            learned_trace = centred_trace
            legal = torch.zeros_like(centred_trace)
            spatial_inputs = tau
        else:
            free = self._aligned_free_mask(normalized_obs_dict)
            centred_trace, legal = self._candidate_trace_and_legality(tau, free)
            learned_trace = self.unmasked_candidate_trace(tau)
            # Older learned encoders retain their historical 0.1 input scale.
            spatial_inputs = self.trace_rho * tau
        actor_trace = self.actor_trace_encoder(spatial_inputs)
        extras = [
            base_context.detach(),
            centred_trace,
            legal,
            learned_trace,
            actor_trace,
        ]
        if self._critic_reads_trace():
            extras.append(self.critic_trace_encoder(spatial_inputs))
        return torch.cat(extras, dim=-1)

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
        actor_trace = core_output[
            :, offset : offset + self.actor_trace_embedding_size
        ]
        offset += self.actor_trace_embedding_size
        critic_trace = core_output[
            :, offset : offset + self.critic_trace_embedding_size
        ]
        offset += self.critic_trace_embedding_size
        if offset != core_output.shape[-1]:
            raise RuntimeError(
                "Unexpected light trace core width: "
                f"consumed {offset}, got {core_output.shape[-1]}."
            )
        return hidden, centred_trace, legal, learned_trace, actor_trace, critic_trace

    def _critic_values(
        self,
        hidden: torch.Tensor,
        critic_trace: torch.Tensor,
    ) -> torch.Tensor:
        """Predict return without allowing critic gradients into frozen EPOM.

        Hidden-only ablations deliberately consume no trace feature.  Every
        variant detaches the 512D EPOM hidden before its own trainable layers.
        """

        frozen_hidden = hidden.detach()
        critic_kind = self._resolved_critic_kind()
        critic_uses_trace = self._critic_reads_trace()
        expected_trace_width = (
            _LightTraceEncoder.OUTPUT_SIZE if critic_uses_trace else 0
        )
        if critic_trace.shape != (hidden.shape[0], expected_trace_width):
            if expected_trace_width == 0:
                raise RuntimeError(
                    f"The {critic_kind} critic must not receive trace features."
                )
            raise RuntimeError(
                f"The {critic_kind} critic expects trace width "
                f"{expected_trace_width}, got {tuple(critic_trace.shape)}."
            )
        if critic_kind == HLINEAR_CRITIC_KIND:
            return self.trace_value_head(frozen_hidden).squeeze(-1)
        hidden_projection = self.critic_hidden_projection(frozen_hidden)
        if critic_kind == HMLP_CRITIC_KIND:
            return self.trace_value_head(hidden_projection).squeeze(-1)
        critic_input = torch.cat(
            [hidden_projection, critic_trace], dim=-1
        )
        return self.trace_value_head(critic_input).squeeze(-1)

    @staticmethod
    def unmasked_candidate_trace(tau: torch.Tensor) -> torch.Tensor:
        """Return plain five-action-centred pressure without reading a mask."""

        if tau.ndim != 4 or tau.shape[1] != 1:
            raise ValueError(f"Expected tau shape [B,1,H,W], got {tuple(tau.shape)}.")
        if tau.shape[-2] != tau.shape[-1] or tau.shape[-1] % 2 != 1:
            raise ValueError(f"Expected odd square tau, got {tuple(tau.shape)}.")
        centre = tau.shape[-1] // 2
        candidates = torch.stack(
            [tau[:, 0, centre + dx, centre + dy] for dx, dy in MOVES], dim=-1
        )
        return candidates - candidates.mean(dim=-1, keepdim=True)

    @staticmethod
    def centered_trace_candidates(tau: torch.Tensor) -> torch.Tensor:
        """Sample p from full-crop-centred P without a second centring step."""

        if tau.ndim != 4 or tuple(tau.shape[1:]) != (1, 11, 11):
            raise ValueError(
                "Centered Shared Trace Memory must have shape [B,1,11,11], "
                f"got {tuple(tau.shape)}."
            )
        centre = 5
        return torch.stack(
            [tau[:, 0, centre + dx, centre + dy] for dx, dy in MOVES],
            dim=-1,
        )

    @staticmethod
    def compose_paper_entropy_multiplier_input(
        trace_feature: torch.Tensor,
        frozen_hidden: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse exactly trace32, frozen EPOM-L h512, and base logits z5."""

        batch = trace_feature.shape[0]
        expected = ((batch, 32), (batch, 512), (batch, 5))
        actual = (
            tuple(trace_feature.shape),
            tuple(frozen_hidden.shape),
            tuple(base_logits.shape),
        )
        if actual != expected:
            raise ValueError(
                "Paper multiplier expects trace32+h512+z5, got "
                f"{actual}."
            )
        return torch.cat(
            [trace_feature, frozen_hidden.detach(), base_logits.detach()],
            dim=-1,
        )

    @staticmethod
    def compose_paper_entropy_fusion_input(
        trace_feature: torch.Tensor,
        frozen_hidden: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse exactly trace32, frozen EPOM-L h512, and base logits z5."""

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
            [
                trace_feature,
                frozen_hidden.detach(),
                base_logits.detach(),
            ],
            dim=-1,
        )

    @classmethod
    def apply_paper_entropy_multiplier_rule(
        cls,
        base_logits: torch.Tensor,
        candidate_pressure: torch.Tensor,
        raw_scores: torch.Tensor,
        entropy_threshold: float = PRIMAL3_ENTROPY_THRESHOLD,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Apply z'=z-g*(p*m), with p used only in this terminal operation."""

        if (
            base_logits.ndim != 2
            or base_logits.shape[-1] != cls.NUM_ACTIONS
            or candidate_pressure.shape != base_logits.shape
            or raw_scores.shape != base_logits.shape
        ):
            raise ValueError(
                "base_logits, candidate_pressure, and raw_scores must all "
                "have shape [B,5]."
            )
        entropy = cls._base_entropy(base_logits)
        gate = (entropy > float(entropy_threshold)).to(
            base_logits.dtype
        ).unsqueeze(-1)
        probabilities = torch.softmax(raw_scores, dim=-1)
        multipliers = 1.0 + probabilities - (1.0 / cls.NUM_ACTIONS)
        correction = gate * (candidate_pressure * multipliers)
        final_logits = base_logits - correction
        return final_logits, -correction, multipliers, gate, entropy

    @classmethod
    def apply_paper_entropy_correction_rule(
        cls,
        base_logits: torch.Tensor,
        raw_correction: torch.Tensor,
        entropy_threshold: float = PRIMAL3_ENTROPY_THRESHOLD,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Apply the figure's direct rule ``z' = z - g*p``.

        ``raw_correction`` is the five-dimensional output of Feature Fusion.
        The entropy gate is computed only from the frozen EPOM-L logits.  No
        candidate-pressure multiplication, mask, clipping, or second
        centring is hidden after the learned output.
        """

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

    @classmethod
    def apply_multiplier_rule(
        cls,
        direct_logits: torch.Tensor,
        centred_trace: torch.Tensor,
        legal: torch.Tensor,
        raw_multiplier: torch.Tensor,
        gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply a true entropy-gated multiplier to Direct's trace pressure.

        Zero ``raw_multiplier`` yields multipliers of exactly one and therefore
        returns ``direct_logits`` bit for bit.  Constant candidate pressure also
        yields no correction after legal centring.
        """

        expected = direct_logits.shape
        if any(tensor.shape != expected for tensor in (centred_trace, legal, raw_multiplier)):
            raise ValueError("All multiplier-rule tensors must have shape [B,5].")
        if gate.shape != (direct_logits.shape[0], 1):
            raise ValueError("Multiplier gate must have shape [B,1].")
        multipliers = 1.0 + torch.tanh(raw_multiplier)
        pressure_adjustment = (multipliers - 1.0) * centred_trace
        learned_delta = -gate * cls._centre_legal(pressure_adjustment, legal)
        return direct_logits + learned_delta, learned_delta, multipliers

    @classmethod
    def apply_coefficient_rule(
        cls,
        direct_logits: torch.Tensor,
        centred_trace: torch.Tensor,
        legal: torch.Tensor,
        raw_coefficient: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add a bounded, action-specific trace coefficient to Direct.

        ``tanh`` maps each output to ``(-1,1)``.  Positive values strengthen
        avoidance, negative values relax it, and zero returns Direct exactly.
        The correction is still structurally zero without local trace contrast.
        """

        expected = direct_logits.shape
        if any(
            tensor.shape != expected
            for tensor in (centred_trace, legal, raw_coefficient)
        ):
            raise ValueError("All coefficient-rule tensors must have shape [B,5].")
        coefficients = torch.tanh(raw_coefficient)
        pressure_adjustment = coefficients * centred_trace
        learned_delta = -cls._centre_legal(pressure_adjustment, legal)
        return direct_logits + learned_delta, learned_delta, coefficients

    @classmethod
    def apply_scalar_gate_rule(
        cls,
        direct_logits: torch.Tensor,
        learned_trace: torch.Tensor,
        raw_gate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Learn one signed residual strength around the fixed Direct rule.

        The scalar is conditioned on the complete 11x11 trace and the frozen
        policy logits.  It answers only *how much* the known trace-pressure
        direction should change in this state.  Zero returns Direct exactly;
        positive values strengthen avoidance and negative values relax it.
        """

        if direct_logits.shape != learned_trace.shape:
            raise ValueError("direct_logits and learned_trace must have shape [B,5].")
        if raw_gate.shape != (direct_logits.shape[0], 1):
            raise ValueError("The scalar gate must have shape [B,1].")
        coefficient = torch.tanh(raw_gate)
        learned_delta = -coefficient * learned_trace
        return direct_logits + learned_delta, learned_delta, coefficient

    @classmethod
    def apply_entropy_direction_rule(
        cls,
        direct_logits: torch.Tensor,
        learned_trace: torch.Tensor,
        raw_direction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply five bounded pressure coefficients without any action mask.

        The actor may relax or strengthen each component of the same raw
        five-action pressure vector.  It cannot add an unconstrained logit
        residual, and the last layer's zero initialization returns Direct
        exactly.  Obstacles are ordinary raw-trace zeros in ``learned_trace``;
        no legal/free tensor is accepted by this function.
        """

        expected = direct_logits.shape
        if any(tensor.shape != expected for tensor in (learned_trace, raw_direction)):
            raise ValueError(
                "direct_logits, learned_trace, and raw_direction must be [B,5]."
            )
        coefficients = torch.tanh(raw_direction)
        learned_delta = -(coefficients * learned_trace)
        return direct_logits + learned_delta, learned_delta, coefficients

    @classmethod
    def apply_spatial_residual_rule(
        cls,
        direct_logits: torch.Tensor,
        raw_residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add a bounded five-action residual without accepting any mask.

        Each raw output is smoothly bounded before the ordinary five-action
        mean is removed.  Mean removal discards the softmax-invariant common
        offset and makes every learned change an explicit redistribution among
        stay/up/down/left/right.  A zero output returns Direct bit for bit.
        """

        if direct_logits.ndim != 2 or direct_logits.shape[-1] != cls.NUM_ACTIONS:
            raise ValueError("direct_logits must have shape [B,5].")
        if raw_residual.shape != direct_logits.shape:
            raise ValueError("raw_residual must have the same [B,5] shape.")
        bounded = 2.0 * torch.tanh(raw_residual / 2.0)
        learned_delta = bounded - bounded.mean(dim=-1, keepdim=True)
        return direct_logits + learned_delta, learned_delta, bounded

    @staticmethod
    def compose_spatial_actor_input(
        trace_features: torch.Tensor,
        policy_probabilities: torch.Tensor | None,
        normalized_entropy: torch.Tensor | None,
        input_order: tuple[str, ...],
    ) -> torch.Tensor:
        """Compose only the declared P/q/H features, in audited order."""

        components = {
            "P": trace_features,
            "q": policy_probabilities,
            "H": normalized_entropy,
        }
        selected: list[torch.Tensor] = []
        for name in input_order:
            tensor = components.get(name)
            if tensor is None:
                raise ValueError(f"Missing declared spatial actor input {name!r}.")
            if tensor.ndim != 2 or tensor.shape[0] != trace_features.shape[0]:
                raise ValueError(
                    f"Spatial actor input {name!r} must have shape [B,D]."
                )
            selected.append(tensor)
        if not selected or input_order[0] != "P":
            raise ValueError("Spatial actor input order must start with P.")
        return torch.cat(selected, dim=-1)

    @staticmethod
    def compose_conv_residual_actor_input(
        trace_features: torch.Tensor,
        relative_action_preference: torch.Tensor,
    ) -> torch.Tensor:
        """Compose conv32(P) and mean-centred action preference only."""

        if trace_features.ndim != 2 or trace_features.shape[-1] != 32:
            raise ValueError("trace_features must have shape [B,32].")
        if relative_action_preference.shape != (trace_features.shape[0], 5):
            raise ValueError(
                "relative_action_preference must have shape [B,5]."
            )
        return torch.cat(
            [trace_features, relative_action_preference], dim=-1
        )

    @staticmethod
    def compose_conv_residual_p_only_actor_input(
        trace_features: torch.Tensor,
    ) -> torch.Tensor:
        """Expose conv32(P) directly, with no policy-summary input."""

        if trace_features.ndim != 2 or trace_features.shape[-1] != 32:
            raise ValueError("trace_features must have shape [B,32].")
        return trace_features

    @staticmethod
    def compose_linear_gain_actor_input(
        direct_centered_pressure: torch.Tensor,
        relative_action_preference: torch.Tensor,
        trace_scale: float = 0.1,
        preference_scale: float = LINEAR_GAIN_RELATIVE_PREFERENCE_SCALE,
    ) -> torch.Tensor:
        """Build the mask-free 10D input used by Linear-Gain CAAR.

        ``direct_centered_pressure`` is the existing fixed Direct readout: its
        mean was computed over static-legal actions and illegal entries are
        zero.  The learned head receives these five values, not the legality
        tensor that produced them.
        """

        if (
            direct_centered_pressure.ndim != 2
            or direct_centered_pressure.shape[-1] != 5
        ):
            raise ValueError("direct_centered_pressure must have shape [B,5].")
        if relative_action_preference.shape != direct_centered_pressure.shape:
            raise ValueError(
                "relative_action_preference must match pressure shape [B,5]."
            )
        if not math.isfinite(float(trace_scale)) or float(trace_scale) <= 0.0:
            raise ValueError("trace_scale must be finite and positive.")
        if (
            not math.isfinite(float(preference_scale))
            or float(preference_scale) <= 0.0
        ):
            raise ValueError("preference_scale must be finite and positive.")
        return torch.cat(
            [
                float(trace_scale) * direct_centered_pressure,
                float(preference_scale) * relative_action_preference,
            ],
            dim=-1,
        )

    @classmethod
    def apply_factorized_gate_rule(
        cls,
        direct_logits: torch.Tensor,
        learned_trace: torch.Tensor,
        raw_direction: torch.Tensor,
        raw_amplitude: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Factor state-level strength from five action-level coefficients.

        The shared amplitude sees only the trace embedding and decides whether
        the current trace pattern deserves a weak or strong correction.  The
        five bounded coefficients additionally see the base logits and decide
        which primitive actions to relax or penalize.  The pressure vector is
        centred over all five actions without a mask.  Zero action
        coefficients return Direct exactly.
        """

        expected = direct_logits.shape
        if any(tensor.shape != expected for tensor in (learned_trace, raw_direction)):
            raise ValueError("All factorized direction tensors must have shape [B,5].")
        if raw_amplitude.shape != (direct_logits.shape[0], 1):
            raise ValueError("The factorized amplitude must have shape [B,1].")
        directions = torch.tanh(raw_direction)
        amplitude = torch.sigmoid(raw_amplitude)
        learned_delta = -(amplitude * directions * learned_trace)
        action_weights = amplitude * directions
        return direct_logits + learned_delta, learned_delta, action_weights

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
        entropy = self._base_entropy(base_logits)

        values = self._critic_values(hidden, critic_trace)
        result = TensorDict(values=values)
        self.last_values = values.detach()
        if values_only:
            return result

        direct_logits, gate, entropy = self.apply_direct_rule(
            base_logits,
            centred_trace,
            entropy_threshold=self.entropy_threshold,
            rule_scale=self.rule_scale,
        )
        if self.reweight_mode == PAPER_ENTROPY_FUSION_ARCHITECTURE:
            fusion_input = self.compose_paper_entropy_fusion_input(
                actor_trace, hidden, base_logits
            )
            actor_input = self.trace_fusion_head(fusion_input)
        elif self.reweight_mode == PAPER_ENTROPY_MULTIPLIER_ARCHITECTURE:
            actor_input = self.compose_paper_entropy_multiplier_input(
                actor_trace, hidden, base_logits
            )
        elif self.reweight_mode in {"entropy_scalar", "entropy_direction"}:
            normalized_entropy = (
                entropy / math.log(float(self.NUM_ACTIONS))
            ).unsqueeze(-1)
            actor_input = torch.cat([actor_trace, normalized_entropy], dim=-1)
        elif self.reweight_mode in SPATIAL_RESIDUAL_ARCHITECTURES:
            contract = self.spatial_residual_contract
            normalized_entropy = (
                (entropy / math.log(float(self.NUM_ACTIONS))).unsqueeze(-1)
                if contract.uses_entropy
                else None
            )
            base_probabilities = (
                torch.softmax(base_logits, dim=-1)
                if contract.uses_probabilities
                else None
            )
            actor_input = self.compose_spatial_actor_input(
                actor_trace,
                base_probabilities,
                normalized_entropy,
                contract.input_order,
            )
            if actor_input.shape[-1] != contract.actor_input_features:
                raise RuntimeError(
                    "Spatial actor input width drift: "
                    f"{actor_input.shape[-1]} != {contract.actor_input_features}."
                )
        elif self.reweight_mode == LINEAR_GAIN_ARCHITECTURE:
            relative_action_preference = (
                base_logits - base_logits.mean(dim=-1, keepdim=True)
            )
            actor_input = self.compose_linear_gain_actor_input(
                centred_trace,
                relative_action_preference,
                trace_scale=self.trace_rho,
            )
        elif self.reweight_mode in CONV_RESIDUAL_ARCHITECTURES:
            if bool(
                CONV_RESIDUAL_SPECS[self.reweight_mode][
                    "uses_relative_action_preference"
                ]
            ):
                relative_action_preference = (
                    base_logits - base_logits.mean(dim=-1, keepdim=True)
                )
                actor_input = self.compose_conv_residual_actor_input(
                    actor_trace, relative_action_preference
                )
            else:
                actor_input = self.compose_conv_residual_p_only_actor_input(
                    actor_trace
                )
        else:
            centred_logits = base_logits - base_logits.mean(dim=-1, keepdim=True)
            actor_input = torch.cat([actor_trace, centred_logits], dim=-1)
        raw_multiplier = self.trace_multiplier_head(actor_input)
        if self.reweight_mode == PAPER_ENTROPY_FUSION_ARCHITECTURE:
            (
                final_logits,
                learned_delta,
                gate,
                entropy,
            ) = self.apply_effective_paper_correction(
                base_logits, raw_multiplier
            )
            # For this architecture the learned correction is the entire
            # reweighting operation.  ``direct_logits`` therefore denotes the
            # unchanged frozen EPOM-L output in shared diagnostics.
            direct_logits = base_logits
            action_weights = gate * raw_multiplier
            learned_gate = gate
        elif self.reweight_mode == PAPER_ENTROPY_MULTIPLIER_ARCHITECTURE:
            (
                final_logits,
                _,
                action_weights,
                gate,
                entropy,
            ) = self.apply_paper_entropy_multiplier_rule(
                base_logits,
                centred_trace,
                raw_multiplier,
                entropy_threshold=self.entropy_threshold,
            )
            # Keep the established diagnostics split for the older multiplier
            # architecture: Direct is m=1 and the learned part is only the
            # multiplier's difference from Direct.
            direct_logits = base_logits - gate * centred_trace
            learned_delta = final_logits - direct_logits
            learned_gate = gate
        elif self.reweight_mode in {"multiplier", LINEAR_GAIN_ARCHITECTURE}:
            final_logits, learned_delta, action_weights = self.apply_multiplier_rule(
                direct_logits, centred_trace, legal, raw_multiplier, gate
            )
            learned_gate = gate
        elif self.reweight_mode == "coefficient":
            final_logits, learned_delta, action_weights = self.apply_coefficient_rule(
                direct_logits, centred_trace, legal, raw_multiplier
            )
            learned_gate = torch.ones_like(gate)
        elif self.reweight_mode in {"scalar_gate", "entropy_scalar"}:
            final_logits, learned_delta, action_weights = self.apply_scalar_gate_rule(
                direct_logits, learned_trace, raw_multiplier
            )
            learned_gate = action_weights
        elif self.reweight_mode == "factorized_gate":
            raw_amplitude = self.trace_amplitude_head(actor_trace)
            final_logits, learned_delta, action_weights = self.apply_factorized_gate_rule(
                direct_logits,
                learned_trace,
                raw_multiplier,
                raw_amplitude,
            )
            learned_gate = torch.sigmoid(raw_amplitude)
        elif self.reweight_mode == "entropy_direction":
            final_logits, learned_delta, action_weights = (
                self.apply_entropy_direction_rule(
                    direct_logits, learned_trace, raw_multiplier
                )
            )
            learned_gate = action_weights.abs().mean(dim=-1, keepdim=True)
        elif self.reweight_mode in {
            *SPATIAL_RESIDUAL_ARCHITECTURES,
            *CONV_RESIDUAL_ARCHITECTURES,
        }:
            final_logits, learned_delta, action_weights = (
                self.apply_spatial_residual_rule(direct_logits, raw_multiplier)
            )
            learned_gate = action_weights.abs().mean(dim=-1, keepdim=True)
        else:
            raise RuntimeError(f"Unsupported reweight mode: {self.reweight_mode!r}.")

        self.last_base_logits = base_logits.detach()
        self.last_direct_logits = direct_logits.detach()
        self.last_final_logits = final_logits.detach()
        self.last_rule_delta = (direct_logits - base_logits).detach()
        self.last_learned_delta = learned_delta.detach()
        self.last_gate = gate.detach()
        self.last_learned_gate = learned_gate.detach()
        self.last_base_entropy = entropy.detach()
        self.last_candidate_trace = centred_trace.detach()
        self.last_learned_trace = learned_trace.detach()
        self.last_legal_mask = legal.detach()
        self.last_multipliers = action_weights.detach()

        self.last_action_distribution = get_action_distribution(
            self.action_space, final_logits
        )
        result["action_logits"] = final_logits
        self._maybe_sample_actions(sample_actions, result)
        return result

    def context_diagnostics(self) -> dict[str, float]:
        result = super().context_diagnostics()
        if self.reweight_mode in {
            PAPER_ENTROPY_MULTIPLIER_ARCHITECTURE,
            PAPER_ENTROPY_FUSION_ARCHITECTURE,
        }:
            # This architecture deliberately never consumes a candidate mask;
            # reporting a fabricated legal fraction would be misleading.
            result.pop("free_candidate_fraction", None)
        if (
            self.last_multipliers is not None
            and self.reweight_mode == PAPER_ENTROPY_FUSION_ARCHITECTURE
        ):
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
        elif self.last_multipliers is not None:
            action_weight = self.last_multipliers.float()
            result.update(
                {
                    "action_weight_mean": float(action_weight.mean()),
                    "action_weight_min": float(action_weight.min()),
                    "action_weight_max": float(action_weight.max()),
                    "architecture_trainable_parameters": float(
                        sum(p.numel() for p in self.trainable_parameters())
                    ),
                }
            )
            if self.reweight_mode in {
                *SPATIAL_RESIDUAL_ARCHITECTURES,
                *CONV_RESIDUAL_ARCHITECTURES,
            }:
                result.update(
                    {
                        "bounded_residual_abs_mean": float(
                            action_weight.abs().mean()
                        ),
                        "bounded_residual_saturation_fraction": float(
                            (action_weight.abs() > 1.9).float().mean()
                        ),
                        "centered_residual_sum_abs_max": float(
                            self.last_learned_delta.sum(dim=-1).abs().max()
                        ),
                    }
                )
        return result

    def checkpoint_provenance(self) -> dict[str, object]:
        result = super().checkpoint_provenance()
        critic_kind = self._resolved_critic_kind()
        critic_uses_trace = self._critic_reads_trace()
        critic_inputs = (
            ["frozen_epom_hidden_512"]
            if critic_kind == HLINEAR_CRITIC_KIND
            else (
                ["frozen_epom_hidden_projection_32"]
                if critic_kind == HMLP_CRITIC_KIND
                else [
                    "frozen_epom_hidden_projection_32",
                    "trace_11x11_conv_adaptive3x3_fc32",
                ]
            )
        )
        result.update(
            {
                "trace_architecture": {
                    "multiplier": "true_multiplier_p_logits_v1",
                    "coefficient": "bounded_coefficient_p_logits_v1",
                    "scalar_gate": "scalar_gate_unmasked_p_logits_v2",
                    "factorized_gate": "factorized_gate_unmasked_p_logits_v2",
                    "entropy_scalar": "entropy_scalar_unmasked_p_v3",
                    "entropy_direction": "entropy_direction_unmasked_p_v3",
                    "tiny_residual16": "tiny_residual16_unmasked_spatial_p_probs_entropy_v4",
                    "linear_spatial_residual": "linear_spatial_residual_unmasked_spatial_p_probs_entropy_v4",
                    LINEAR_GAIN_ARCHITECTURE: str(
                        LINEAR_GAIN_SPEC["provenance_name"]
                    ),
                    PAPER_ENTROPY_MULTIPLIER_ARCHITECTURE: (
                        "paper_entropy_multiplier_centered_P_h_z_v1"
                    ),
                    PAPER_ENTROPY_FUSION_ARCHITECTURE: (
                        "paper_entropy_conv_direct_correction_centered_P_h_z_v3"
                    ),
                    **{
                        name: str(spec["provenance_name"])
                        for name, spec in CONV_RESIDUAL_SPECS.items()
                    },
                }[self.reweight_mode],
                "actor_inputs": (
                    [
                        "trace_11x11_flatten_row_major_121",
                        "base_policy_probabilities_5",
                        "base_policy_entropy_1",
                    ]
                    if self.reweight_mode
                    in {"tiny_residual16", "linear_spatial_residual"}
                    else (
                        list(LINEAR_GAIN_SPEC["actor_inputs"])
                        if self.reweight_mode == LINEAR_GAIN_ARCHITECTURE
                        else
                        (
                            [
                                "trace_11x11_conv_adaptive3x3_fc32",
                                "base_relative_action_preference_mean_centered_5",
                            ]
                            if bool(
                                CONV_RESIDUAL_SPECS[self.reweight_mode][
                                    "uses_relative_action_preference"
                                ]
                            )
                            else ["trace_11x11_conv_adaptive3x3_fc32"]
                        )
                        if self.reweight_mode in CONV_RESIDUAL_ARCHITECTURES
                        else
                        ["trace_11x11", "base_policy_entropy_1"]
                        if self.reweight_mode
                        in {"entropy_scalar", "entropy_direction"}
                        else ["trace_11x11", "base_logits_5"]
                    )
                ),
                "actor_output": {
                    "multiplier": "five_trace_multipliers_in_0_2",
                    "coefficient": "five_trace_coefficients_in_minus1_1",
                    "scalar_gate": "one_signed_trace_coefficient_in_minus1_1",
                    "factorized_gate": "one_amplitude_and_five_signed_coefficients",
                    "entropy_scalar": "one_signed_trace_coefficient_in_minus1_1",
                    "entropy_direction": "five_signed_trace_coefficients_in_minus1_1",
                    "tiny_residual16": "five_bounded_then_mean_centered_logit_residuals",
                    "linear_spatial_residual": "five_bounded_then_mean_centered_logit_residuals",
                    LINEAR_GAIN_ARCHITECTURE: (
                        "five_trace_multipliers_in_0_2_on_static_legal_direct_pressure"
                    ),
                    PAPER_ENTROPY_MULTIPLIER_ARCHITECTURE: (
                        "five_mean_one_softmax_pressure_multipliers"
                    ),
                    PAPER_ENTROPY_FUSION_ARCHITECTURE: (
                        "five_direct_logit_corrections"
                    ),
                    **{
                        name: "five_bounded_then_mean_centered_logit_residuals"
                        for name in CONV_RESIDUAL_ARCHITECTURES
                    },
                }[self.reweight_mode],
                "multiplier_gate": (
                    "same_entropy_gate_as_direct"
                    if self.reweight_mode
                    in {
                        "multiplier",
                        LINEAR_GAIN_ARCHITECTURE,
                        PAPER_ENTROPY_MULTIPLIER_ARCHITECTURE,
                        PAPER_ENTROPY_FUSION_ARCHITECTURE,
                    }
                    else "not_applicable"
                ),
                "actor_uses_epom_hidden": False,
                "actor_uses_base_logits": self.reweight_mode
                not in {
                    "entropy_scalar",
                    "entropy_direction",
                    "tiny_residual16",
                    "linear_spatial_residual",
                    LINEAR_GAIN_ARCHITECTURE,
                },
                "actor_uses_base_policy_probabilities": self.reweight_mode
                in {"tiny_residual16", "linear_spatial_residual"},
                "actor_input_features": (
                    127
                    if self.reweight_mode
                    in {"tiny_residual16", "linear_spatial_residual"}
                    else "not_applicable"
                ),
                "actor_hidden_features": (
                    16
                    if self.reweight_mode == "tiny_residual16"
                    else (
                        0
                        if self.reweight_mode == "linear_spatial_residual"
                        else "not_applicable"
                    )
                ),
                "base_policy_entropy_normalization": (
                    "divide_by_log_5"
                    if self.reweight_mode
                    in {
                        "entropy_scalar",
                        "entropy_direction",
                        "tiny_residual16",
                        "linear_spatial_residual",
                    }
                    else "not_applicable"
                ),
                "learned_network_uses_free_mask": False,
                "learned_residual_uses_free_mask": False,
                "learned_pressure": (
                    "not_applicable_free_five_logit_residual"
                    if self.reweight_mode
                    in {"tiny_residual16", "linear_spatial_residual"}
                    else "plain_five_action_centered_raw_trace"
                ),
                "learned_residual_centering": (
                    "ordinary_mean_over_all_five_actions_no_mask"
                    if self.reweight_mode
                    in {"tiny_residual16", "linear_spatial_residual"}
                    else "not_applicable"
                ),
                "learned_residual_bound": (
                    "2*tanh(raw/2)_before_five_action_centering"
                    if self.reweight_mode
                    in {"tiny_residual16", "linear_spatial_residual"}
                    else "not_applicable"
                ),
                "trace_flatten_order": (
                    "row_major_y_then_x"
                    if self.reweight_mode
                    in {"tiny_residual16", "linear_spatial_residual"}
                    else "not_applicable"
                ),
                "actor_trainable_parameters": sum(
                    parameter.numel()
                    for module in (
                        self.actor_trace_encoder,
                        self.trace_multiplier_head,
                    )
                    for parameter in module.parameters()
                    if parameter.requires_grad
                ),
                "critic_uses_epom_hidden": True,
                "critic_uses_trace": critic_uses_trace,
                "critic_backpropagates_to_epom": False,
                "critic_architecture": critic_kind,
                "critic_inputs": critic_inputs,
                "critic_trainable_parameters": sum(
                    parameter.numel()
                    for name, parameter in self.named_parameters()
                    if parameter.requires_grad
                    and name.startswith(
                        (
                            "critic_trace_encoder.",
                            "critic_hidden_projection.",
                            "trace_value_head.",
                        )
                    )
                ),
            }
        )
        if self.spatial_residual_contract is not None:
            contract = self.spatial_residual_contract
            result.update(contract.checkpoint_fields())
            result.update(
                {
                    "trace_architecture": contract.provenance_name,
                    "actor_uses_epom_hidden": False,
                    "actor_uses_base_logits": False,
                    "actor_uses_base_policy_probabilities": (
                        contract.uses_probabilities
                    ),
                    "actor_uses_base_policy_entropy": contract.uses_entropy,
                    "base_policy_entropy_normalization": (
                        "divide_by_log_5"
                        if contract.uses_entropy
                        else "not_applicable"
                    ),
                    "trace_flatten_order": contract.trace_order,
                    "learned_network_uses_free_mask": False,
                    "learned_residual_uses_free_mask": False,
                    "direct_trace_source": "real_P_with_exact_free_mask",
                    "direct_uses_real_trace": True,
                }
            )
        if self.reweight_mode == LINEAR_GAIN_ARCHITECTURE:
            # This legacy field cannot distinguish receiving a mask tensor from
            # consuming a pressure vector produced by legal preprocessing.
            result.pop("learned_residual_uses_free_mask", None)
            result.update(
                {
                    "actor_uses_epom_hidden": False,
                    "actor_uses_base_logits": False,
                    "actor_uses_raw_base_logits": False,
                    "actor_uses_relative_action_preference": True,
                    "relative_action_preference": (
                        "base_logits_minus_all_five_action_mean"
                    ),
                    "actor_uses_base_policy_probabilities": False,
                    "actor_uses_base_policy_entropy": False,
                    "actor_receives_free_mask_tensor": False,
                    "actor_pressure_preprocessing_uses_static_legality": True,
                    "actor_input_features": int(
                        LINEAR_GAIN_SPEC["input_features"]
                    ),
                    "actor_hidden_features": int(
                        LINEAR_GAIN_SPEC["hidden_features"]
                    ),
                    "linear_gain_input_contract": str(
                        LINEAR_GAIN_SPEC["input_contract"]
                    ),
                    "input_scaling": {
                        "direct_centered_pressure_5": self.trace_rho,
                        "base_relative_action_preference_5": (
                            LINEAR_GAIN_RELATIVE_PREFERENCE_SCALE
                        ),
                    },
                    "actor_trace_encoder": (
                        "none_direct_pressure_precomputed_parameter_free"
                    ),
                    "actor_trace_embedding_features": 0,
                    "learned_gate_mode": "entropy",
                    "learned_network_uses_free_mask": False,
                    "learned_residual_uses_static_legality": True,
                    "learned_pressure": "direct_static_legal_centered_pressure",
                    "learned_residual_centering": (
                        "static_legal_mean_after_actionwise_gain"
                    ),
                    "learned_residual_bound": (
                        "pressure_multiplier_1_plus_tanh_in_0_2"
                    ),
                    "gain_application_uses_static_legality": True,
                    "direct_trace_source": "real_P_with_exact_free_mask",
                    "direct_uses_real_trace": True,
                }
            )
        if self.reweight_mode == PAPER_ENTROPY_MULTIPLIER_ARCHITECTURE:
            result.update(
                {
                    "actor_inputs": [
                        "centered_trace_11x11_conv32_resblock2_fc32",
                        "frozen_epom_recurrent_hidden_512",
                        "frozen_epom_base_logits_5",
                    ],
                    "actor_output": (
                        "softmax_scores_to_five_multipliers_"
                        "m_equals_1_plus_q_minus_one_fifth"
                    ),
                    "actor_uses_epom_hidden": True,
                    "actor_uses_base_logits": True,
                    "actor_uses_raw_base_logits": True,
                    "actor_uses_base_policy_probabilities": False,
                    "actor_uses_base_policy_entropy": False,
                    "actor_input_features": 549,
                    "actor_hidden_features": 128,
                    "actor_trace_encoder": (
                        "conv32_3x3_resblock32x2_flatten_fc32"
                    ),
                    "actor_trace_embedding_features": 32,
                    "learned_gate_mode": "raw_shannon_entropy_hard_gate",
                    "entropy_threshold": self.entropy_threshold,
                    "base_policy_entropy_normalization": "none",
                    "learned_network_uses_free_mask": False,
                    "learned_residual_uses_free_mask": False,
                    "actor_receives_candidate_pressure": False,
                    "candidate_pressure_order": [
                        "wait",
                        "up",
                        "down",
                        "left",
                        "right",
                    ],
                    "candidate_pressure_source": (
                        "five_cells_of_free_cell_mean_centered_11x11_trace"
                    ),
                    "candidate_pressure_is_terminal_operand_only": True,
                    "logit_rule": "z_prime_equals_z_minus_g_times_p_times_m",
                    "multiplier_range": [0.8, 1.8],
                    "multiplier_mean": 1.0,
                    "learned_pressure": (
                        "five_candidates_sampled_from_full_11x11_"
                        "free_cell_mean_centered_trace"
                    ),
                    "learned_residual_centering": "none_after_full_crop_centering",
                    "learned_residual_bound": (
                        "positive_mean_one_softmax_multiplier_in_0.8_to_1.8"
                    ),
                    "direct_trace_source": "mean_centered_P_no_mask",
                    "direct_uses_real_trace": True,
                }
            )
        if self.reweight_mode == PAPER_ENTROPY_FUSION_ARCHITECTURE:
            trace_encoder_parameters = sum(
                parameter.numel()
                for parameter in self.actor_trace_encoder.parameters()
                if parameter.requires_grad
            )
            fusion_parameters = sum(
                parameter.numel()
                for parameter in self.trace_fusion_head.parameters()
                if parameter.requires_grad
            )
            actor_head_parameters = sum(
                parameter.numel()
                for parameter in self.trace_multiplier_head.parameters()
                if parameter.requires_grad
            )
            critic_head_parameters = sum(
                parameter.numel()
                for parameter in self.trace_value_head.parameters()
                if parameter.requires_grad
            )
            all_state_override = self.inference_learned_gate_override == "all"
            result.update(
                {
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
                    "trace_encoder_parameters": trace_encoder_parameters,
                    "feature_fusion_parameters": fusion_parameters,
                    "actor_head_parameters": actor_head_parameters,
                    "critic_head_parameters": critic_head_parameters,
                    "actor_trainable_parameters": (
                        trace_encoder_parameters
                        + fusion_parameters
                        + actor_head_parameters
                    ),
                    "critic_trainable_parameters": critic_head_parameters,
                    "critic_architecture": HLINEAR_CRITIC_KIND,
                    "critic_inputs": critic_inputs,
                    "critic_uses_trace": False,
                    "critic_backpropagates_to_epom": False,
                    "learned_gate_mode": (
                        "all_states_inference_override"
                        if all_state_override
                        else "raw_shannon_entropy_hard_gate"
                    ),
                    "checkpoint_learned_gate_mode": "entropy",
                    "inference_learned_gate_override": (
                        self.inference_learned_gate_override
                    ),
                    "entropy_threshold": self.entropy_threshold,
                    "base_policy_entropy_normalization": "none",
                    "learned_network_uses_free_mask": False,
                    "learned_residual_uses_free_mask": False,
                    "correction_action_order": [
                        "wait",
                        "up",
                        "down",
                        "left",
                        "right",
                    ],
                    "logit_rule": (
                        "z_prime_equals_z_minus_p"
                        if all_state_override
                        else "z_prime_equals_z_minus_entropy_gated_p"
                    ),
                    "correction_source": "feature_fusion_linear_output_5",
                    "correction_centering": "none",
                    "correction_bound": "unbounded_logit_space",
                    "zero_initial_correction": True,
                    "entropy_gate_applies_to_correction": (
                        not all_state_override
                    ),
                    "trace_input": (
                        "full_11x11_free_cell_mean_centered_trace"
                    ),
                    "trace_obstacles_and_padding": "zero",
                }
            )
        if self.reweight_mode in CONV_RESIDUAL_ARCHITECTURES:
            spec = CONV_RESIDUAL_SPECS[self.reweight_mode]
            uses_relative_preference = bool(
                spec["uses_relative_action_preference"]
            )
            result.update(
                {
                    "actor_uses_epom_hidden": False,
                    "actor_uses_base_logits": False,
                    "actor_uses_raw_base_logits": False,
                    "actor_uses_relative_action_preference": (
                        uses_relative_preference
                    ),
                    "relative_action_preference": (
                        "base_logits_minus_all_five_action_mean"
                        if uses_relative_preference
                        else "not_applicable"
                    ),
                    "actor_uses_base_policy_probabilities": False,
                    "actor_uses_base_policy_entropy": False,
                    "actor_input_features": int(spec["input_features"]),
                    "actor_hidden_features": int(spec["hidden_features"]),
                    "conv_residual_input_contract": (
                        "conv32_P_plus_relative_preference_5"
                        if uses_relative_preference
                        else "conv32_P_only"
                    ),
                    "base_policy_entropy_normalization": "not_applicable",
                    "actor_trace_encoder": (
                        "two_conv_adaptive3x3_fc32_raw_P_only"
                    ),
                    "actor_trace_embedding_features": 32,
                    "learned_gate_mode": "all",
                    "learned_network_uses_free_mask": False,
                    "learned_residual_uses_free_mask": False,
                    "learned_pressure": (
                        "not_applicable_free_five_logit_residual"
                    ),
                    "learned_residual_centering": (
                        "ordinary_mean_over_all_five_actions_no_mask"
                    ),
                    "learned_residual_bound": (
                        "2*tanh(raw/2)_before_five_action_centering"
                    ),
                    "trace_flatten_order": "not_applicable_conv32",
                    "direct_trace_source": "real_P_with_exact_free_mask",
                    "direct_uses_real_trace": True,
                }
            )
        return result


__all__ = [
    "EPOMTraceMultiplierActorCritic",
    "CONV_RESIDUAL_ARCHITECTURES",
    "CONV_RESIDUAL_SPECS",
    "HIDDEN_ONLY_CRITIC_ARCHITECTURES",
    "HLINEAR_CRITIC_ARCHITECTURES",
    "HLINEAR_CRITIC_KIND",
    "HMLP_CRITIC_KIND",
    "LEGACY_TRACE_CRITIC_KIND",
    "LINEAR_VALUE_CRITIC_KIND",
    "LINEAR_GAIN_ARCHITECTURE",
    "PAPER_ENTROPY_MULTIPLIER_ARCHITECTURE",
    "PAPER_ENTROPY_FUSION_ARCHITECTURE",
    "LINEAR_GAIN_RELATIVE_PREFERENCE_SCALE",
    "LINEAR_GAIN_SPEC",
    "_CenterTraceEncoder",
    "_EmptyActorTraceEncoder",
    "_EntropyTraceEncoder",
    "_FlattenTraceEncoder",
    "_LightTraceEncoder",
    "_PaperTraceEncoder",
    "_make_spatial_residual_head",
    "_make_conv_residual_head",
    "_make_linear_gain_head",
]
