from __future__ import annotations

import math
from pathlib import Path
import torch
from sample_factory.algo.utils.action_distributions import get_action_distribution
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.actor_critic import ActorCriticSharedWeights
from sample_factory.model.encoder import ResBlock
from torch import nn
from torch.nn.utils.rnn import PackedSequence

PRIMAL3_ENTROPY_THRESHOLD = 0.46371241
PRIMAL3_ENTROPY_EPS = 1e-10
MOVES = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
PAPER_ENTROPY_FUSION_ARCHITECTURE = "paper_entropy_fusion"
TIE_KEY = "bonus_tie_ranks"


def select_top2_low_pressure(logits, pressure, legal, tie_ranks):
    if logits.ndim != 2 or logits.shape[-1] != 5:
        raise ValueError("logits must have shape [B,5]")
    if pressure.shape != logits.shape or legal.shape != logits.shape:
        raise ValueError("pressure and legal must match logits [B,5]")
    if tie_ranks.shape != (logits.shape[0], 2, 5):
        raise ValueError("bonus_tie_ranks must have shape [B,2,5]")
    with torch.no_grad():
        z, p, ranks = logits.detach(), pressure.detach(), tie_ranks.detach()
        moves = (legal.detach() > 0.5).clone()
        moves[:, 0] = False
        scores = torch.where(moves, z, -torch.inf)
        ties = moves & (scores == scores.max(dim=-1, keepdim=True).values)
        first = torch.where(ties, ranks[:, 0], -1.0).argmax(dim=-1, keepdim=True)
        remaining = moves.scatter(1, first, False)
        scores = torch.where(remaining, z, -torch.inf)
        ties = remaining & (scores == scores.max(dim=-1, keepdim=True).values)
        second = torch.where(ties, ranks[:, 0], -1.0).argmax(dim=-1, keepdim=True)
        top2 = torch.zeros_like(moves).scatter(1, first, True).scatter(1, second, True)
        top2 = top2 & moves
        pressures = torch.where(top2, p, torch.inf)
        ties = top2 & (pressures == pressures.min(dim=-1, keepdim=True).values)
        winner = torch.where(ties, ranks[:, 1], -1.0).argmax(dim=-1, keepdim=True)
        route = torch.zeros_like(z).scatter(1, winner, 1.0)
        return route * (moves.sum(dim=-1, keepdim=True) >= 2).to(z.dtype)


class _PaperTraceEncoder(nn.Module):
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
                f"Paper Trace Encoder requires [B,1,11,11], got {tuple(inputs.shape)}."
            )
        return self.network(inputs)


class _FrozenEPOMActorCritic(ActorCriticSharedWeights):
    NUM_ACTIONS = 5
    TRAINABLE_PREFIXES = ()

    def _load_and_freeze_base(self, settings):
        directory = Path(settings["epom_base_weights_path"]).expanduser()
        if not directory.is_absolute():
            directory = Path(__file__).resolve().parents[1] / directory
        checkpoint_path = getattr(self.cfg, "base_checkpoint_path", None)
        if checkpoint_path is None:
            checkpoints = sorted(
                path
                for path in (directory / "checkpoint_p0").glob("*.pth")
                if not path.name.startswith("best_")
            )
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoint under {directory}.")
            checkpoint_path = checkpoints[-1]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        incompatible = self.load_state_dict(checkpoint["model"], strict=False)
        missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(self.TRAINABLE_PREFIXES)
        ]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"EPOM checkpoint mismatch: missing={missing}, "
                f"unexpected={incompatible.unexpected_keys}"
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

    def train(self, mode=True):
        super().train(mode)
        for module in getattr(self, "_frozen_base_modules", ()):
            module.eval()
        return self

    @staticmethod
    def _base_entropy(logits):
        probabilities = torch.softmax(logits, dim=-1)
        return -(probabilities * torch.log(probabilities + PRIMAL3_ENTROPY_EPS)).sum(-1)

    @staticmethod
    def _packed_like(reference, data):
        return PackedSequence(
            data,
            reference.batch_sizes,
            reference.sorted_indices,
            reference.unsorted_indices,
        )


class EPOMTraceMultiplierActorCritic(_FrozenEPOMActorCritic):
    TRAINABLE_PREFIXES = (
        "actor_trace_encoder.",
        "trace_fusion_head.",
        "trace_multiplier_head.",
        "critic_trace_encoder.",
        "critic_fusion_head.",
        "trace_value_head.",
    )

    @staticmethod
    def _validate_spatial_contract(
        obs_shape: tuple[int, ...],
        tau_shape: tuple[int, ...],
        tau_free_mask_shape: tuple[int, ...] | None,
    ) -> tuple[int, str]:

        if tau_shape != (1, 11, 11):
            raise ValueError(
                f"Paper ARPE requires the [1,11,11] trace crop, got {tau_shape}."
            )
        if (
            len(obs_shape) != 3
            or obs_shape[-2] != obs_shape[-1]
            or obs_shape[-1] % 2 != 1
            or obs_shape[-1] < 11
        ):
            raise ValueError(
                "Paper ARPE requires an odd square EPOM observation of at "
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
            raise ValueError("Paper ARPE requires shared base weights.")
        if "tau" not in obs_space.spaces:
            raise ValueError("Paper ARPE requires a tau observation.")
        if TIE_KEY not in obs_space.spaces or tuple(obs_space[TIE_KEY].shape) != (2, 5):
            raise ValueError("ARPE requires stored bonus_tie_ranks with shape [2,5].")
        if getattr(action_space, "n", None) != self.NUM_ACTIONS:
            raise ValueError(f"Expected five discrete actions, got {action_space}.")

        ActorCriticSharedWeights.__init__(
            self, model_factory, obs_space, action_space, cfg
        )
        settings = cfg.full_config["experiment_settings"]
        mask_shape = (
            tuple(obs_space["tau_free_mask"].shape)
            if "tau_free_mask" in obs_space.spaces
            else None
        )
        _, self.free_mask_source = self._validate_spatial_contract(
            tuple(obs_space["obs"].shape), tuple(obs_space["tau"].shape), mask_shape
        )
        self.core_out_size = int(self.core.get_out_size())
        self.rule_scale = float(settings.get("trace_rule_scale", 1.0))
        self.entropy_threshold = float(
            settings.get("trace_gate_threshold", PRIMAL3_ENTROPY_THRESHOLD)
        )
        self.learned_gate_mode = str(
            settings.get("trace_context_learned_gate", "entropy")
        )
        if self.learned_gate_mode not in {"entropy", "always"}:
            raise ValueError("ARPE training gate must be entropy or always.")

        self.actor_trace_encoder = _PaperTraceEncoder(cfg)
        if self.core_out_size != 512:
            raise ValueError("ARPE requires the 512D EPOM hidden state.")
        self.critic_trace_encoder = _PaperTraceEncoder(cfg)
        self.critic_fusion_head = nn.Sequential(nn.Linear(549, 256), nn.ReLU())
        self.trace_value_head = nn.Linear(256, 1)
        self.trace_fusion_head = nn.Sequential(
            nn.Linear(
                _PaperTraceEncoder.OUTPUT_SIZE + self.core_out_size + self.NUM_ACTIONS,
                256,
            ),
            nn.ReLU(),
        )
        self.trace_multiplier_head = nn.Sequential(nn.Linear(256, self.NUM_ACTIONS))
        for module in (
            self.actor_trace_encoder,
            self.trace_multiplier_head,
            self.trace_value_head,
            self.trace_fusion_head,
            self.critic_trace_encoder,
            self.critic_fusion_head,
        ):
            module.apply(self.initialize_weights)
        nn.init.zeros_(self.trace_multiplier_head[-1].weight)
        nn.init.zeros_(self.trace_multiplier_head[-1].bias)

        self._load_and_freeze_base(settings)
        self.register_buffer(
            "fixed_entropy_threshold",
            torch.tensor(self.entropy_threshold, dtype=torch.float64),
        )
        self.register_buffer(
            "paper_entropy_gate_version",
            torch.tensor(
                1 if self.learned_gate_mode == "entropy" else 0, dtype=torch.int64
            ),
        )
        self.register_buffer(
            "independent_critic_version", torch.tensor(1, dtype=torch.int64)
        )
        self.register_buffer(
            "allaction_residual_version", torch.tensor(2, dtype=torch.int64)
        )
        self.actor_trace_embedding_size = self.actor_trace_encoder.OUTPUT_SIZE
        self.critic_trace_embedding_size = self.critic_trace_encoder.OUTPUT_SIZE
        self.head_extra_size = (
            3 * self.NUM_ACTIONS
            + self.actor_trace_embedding_size
            + self.critic_trace_embedding_size
            + 10
        )

    def forward_head(self, normalized_obs_dict):
        with torch.no_grad():
            base_context = self.encoder(normalized_obs_dict)
        tau = normalized_obs_dict["tau"].float()
        centred_trace = self.centered_trace_candidates(tau)
        if self.free_mask_source == "tau_free_mask":
            free = normalized_obs_dict["tau_free_mask"].detach()
        else:
            free = (normalized_obs_dict["obs"][:, 0:1].detach() < 0.5).to(tau.dtype)
        legal = (self.centered_trace_candidates(free) > 0.5).to(tau.dtype)
        actor_trace = self.actor_trace_encoder(tau)
        critic_trace = self.critic_trace_encoder(tau)
        ranks = normalized_obs_dict[TIE_KEY].detach().to(tau.dtype)
        if ranks.shape != (tau.shape[0], 2, 5):
            raise ValueError("Stored routing rankings must be [B,2,5]")
        return torch.cat(
            [
                base_context.detach(),
                centred_trace,
                legal,
                centred_trace,
                actor_trace,
                critic_trace,
                ranks.flatten(1),
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

    def _split_core(self, core_output):
        hidden, pressure, legal, _pressure_copy, actor_trace, critic_trace = (
            core_output.split(
                (
                    self.core_out_size,
                    self.NUM_ACTIONS,
                    self.NUM_ACTIONS,
                    self.NUM_ACTIONS,
                    self.actor_trace_embedding_size,
                    self.critic_trace_embedding_size,
                ),
                dim=-1,
            )
        )
        return hidden, pressure, legal, actor_trace, critic_trace

    def _critic_values(
        self,
        hidden: torch.Tensor,
        critic_trace: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:

        features = self.compose_paper_entropy_fusion_input(
            critic_trace, hidden, base_logits
        )
        return self.trace_value_head(self.critic_fusion_head(features)).squeeze(-1)

    @staticmethod
    def centered_trace_candidates(tau: torch.Tensor) -> torch.Tensor:

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

        batch = trace_feature.shape[0]
        expected = ((batch, 32), (batch, 512), (batch, 5))
        actual = (
            tuple(trace_feature.shape),
            tuple(frozen_hidden.shape),
            tuple(base_logits.shape),
        )
        if actual != expected:
            raise ValueError(
                f"Paper entropy fusion expects trace32+h512+z5, got {actual}."
            )
        return torch.cat(
            [trace_feature, frozen_hidden.detach(), base_logits.detach()], dim=-1
        )

    @classmethod
    def apply_paper_entropy_correction_rule(
        cls,
        base_logits: torch.Tensor,
        raw_correction: torch.Tensor,
        pressure: torch.Tensor,
        legal: torch.Tensor,
        tie_ranks: torch.Tensor,
        *,
        direct_bonus: float = 1.0,
        entropy_threshold: float = PRIMAL3_ENTROPY_THRESHOLD,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        if (
            base_logits.ndim != 2
            or base_logits.shape[-1] != cls.NUM_ACTIONS
            or raw_correction.shape != base_logits.shape
        ):
            raise ValueError(
                "base_logits and raw_correction must both have shape [B,5]."
            )
        if not math.isfinite(direct_bonus) or direct_bonus < 0:
            raise ValueError("direct_bonus must be finite and non-negative")
        if not math.isfinite(entropy_threshold) or entropy_threshold < 0:
            raise ValueError("entropy_threshold must be finite and non-negative")
        base_logits = base_logits.detach()
        entropy = cls._base_entropy(base_logits)
        gate = (entropy > entropy_threshold).to(base_logits.dtype).unsqueeze(-1)
        route = (
            select_top2_low_pressure(base_logits, pressure, legal, tie_ranks)
            if direct_bonus
            else torch.zeros_like(base_logits)
        )
        learned_delta = gate * raw_correction
        final_logits = base_logits + direct_bonus * gate * route + learned_delta
        return final_logits, learned_delta, gate, entropy

    def _apply_configured_correction_rule(
        self,
        base_logits,
        raw_correction,
        pressure,
        legal,
        tie_ranks,
    ):
        inference = getattr(self, "inference_correction", None)
        if inference is not None:
            if self.training:
                raise RuntimeError(
                    "Inference correction must not be used for training."
                )
            base_logits = base_logits.detach()
            return inference.apply(
                base_logits,
                raw_correction,
                self._base_entropy(base_logits),
            )
        if self.learned_gate_mode == "entropy":
            return self.apply_paper_entropy_correction_rule(
                base_logits,
                raw_correction,
                pressure,
                legal,
                tie_ranks,
                direct_bonus=self.rule_scale,
                entropy_threshold=getattr(
                    self, "entropy_threshold", PRIMAL3_ENTROPY_THRESHOLD
                ),
            )
        base_logits = base_logits.detach()
        entropy = self._base_entropy(base_logits)
        gate = torch.ones_like(base_logits[:, :1])
        route = (
            select_top2_low_pressure(base_logits, pressure, legal, tie_ranks)
            if self.rule_scale
            else torch.zeros_like(base_logits)
        )
        return (
            base_logits + self.rule_scale * route + raw_correction,
            raw_correction,
            gate,
            entropy,
        )

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool):
        if isinstance(core_output, PackedSequence):
            raise TypeError("Sample Factory must unpack PackedSequence before tail.")
        ranks = core_output[:, -10:].reshape(-1, 2, 5)
        (
            hidden,
            centred_trace,
            legal,
            actor_trace,
            critic_trace,
        ) = self._split_core(core_output[:, :-10])
        with torch.no_grad():
            decoder_output = self.decoder(hidden)
            base_logits, _ = self.action_parameterization(decoder_output)
        base_logits = base_logits.detach()
        values = self._critic_values(hidden, critic_trace, base_logits)
        result = TensorDict(values=values)
        if values_only:
            return result

        fusion_input = self.compose_paper_entropy_fusion_input(
            actor_trace, hidden, base_logits
        )
        actor_input = self.trace_fusion_head(fusion_input)
        raw_correction = self.trace_multiplier_head(actor_input)
        final_logits, _, _, _ = self._apply_configured_correction_rule(
            base_logits,
            raw_correction,
            centred_trace,
            legal,
            ranks,
        )
        self.last_action_distribution = get_action_distribution(
            self.action_space, final_logits
        )
        result["action_logits"] = final_logits
        self._maybe_sample_actions(sample_actions, result)
        return result
