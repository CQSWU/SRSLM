"""Sample Factory actor-critic with shared-trace action pressure."""

import torch
from sample_factory.algo.utils.action_distributions import get_action_distribution
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.actor_critic import ActorCriticSharedWeights
from sample_factory.model.encoder import ResBlock
from torch import nn
from torch.nn.utils.rnn import PackedSequence


class CAARActorCritic(ActorCriticSharedWeights):
    """Subtract either observed or learned pressure from five action logits."""

    NUM_ACTIONS = 5
    MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))
    ACTIONS = ((0, 0),) + MOVES
    def __init__(self, model_factory, obs_space, action_space, cfg):
        if not cfg.actor_critic_share_weights:
            raise ValueError("CAAR requires shared actor-critic weights.")
        if "tau" not in obs_space.spaces:
            raise ValueError("CAAR requires a separate tau observation.")

        super().__init__(model_factory, obs_space, action_space, cfg)
        if getattr(action_space, "n", None) != self.NUM_ACTIONS:
            raise ValueError(f"CAAR expects five discrete actions, got {action_space}.")

        self.core_out_size = self.core.get_out_size()
        tau_shape = obs_space["tau"].shape
        if len(tau_shape) != 3 or tau_shape[0] != 1:
            raise ValueError(f"Expected tau shape [1, H, W], got {tau_shape}.")
        self.learn_residual = bool(getattr(cfg, "caar_learn_residual", True))
        self.contextual_pressure = bool(
            getattr(cfg, "caar_contextual_pressure", False)
        )
        if self.learn_residual != self.contextual_pressure:
            raise ValueError(
                "CAAR supports only Direct (both flags false) or Context "
                "(both flags true); the legacy residual mode is retired."
            )
        self.pressure_head_mode = str(
            getattr(cfg, "caar_pressure_head_mode", "legacy_multiplier")
        )
        if self.pressure_head_mode not in (
            "legacy_multiplier",
            "direct_pressure",
        ):
            raise ValueError(
                "caar_pressure_head_mode must be 'legacy_multiplier' or "
                "'direct_pressure'."
            )
        self.pressure_output_transform = str(
            getattr(cfg, "caar_pressure_output_transform", "clipped_relu")
        )
        if self.pressure_output_transform not in ("clipped_relu", "identity"):
            raise ValueError(
                "caar_pressure_output_transform must be 'clipped_relu' or "
                "'identity'."
            )
        self.pressure_cap = float(getattr(cfg, "caar_pressure_cap", 2.0))
        self.pressure_init = float(getattr(cfg, "caar_pressure_init", 0.1))
        if self.pressure_cap <= 0.0:
            raise ValueError("caar_pressure_cap must be positive.")
        if self.pressure_init < 0.0:
            raise ValueError("caar_pressure_init must be non-negative.")
        reweight_wait_action = getattr(cfg, "caar_reweight_wait_action", None)
        if reweight_wait_action is None:
            # Direct has no learned pressure head, so its existing checkpoint is
            # shape-compatible with five-action reweighting. Legacy contextual
            # checkpoints retain their original four-direction head.
            reweight_wait_action = not self.contextual_pressure
        self.reweight_wait_action = bool(reweight_wait_action)
        if self.contextual_pressure and self.pressure_head_mode == "direct_pressure":
            if not self.reweight_wait_action:
                raise ValueError(
                    f"{self.pressure_head_mode} CAAR must reweight all five actions, "
                    "including wait."
                )
        self.pressure_offsets = (
            self.ACTIONS if self.reweight_wait_action else self.MOVES
        )
        self.pressure_size = len(self.pressure_offsets)
        tau_filters = int(getattr(cfg, "caar_tau_num_filters", 8))
        tau_conv_layers = int(getattr(cfg, "caar_tau_num_conv_layers", 1))
        tau_res_blocks = int(getattr(cfg, "caar_tau_num_res_blocks", 0))
        tau_hidden_size = int(getattr(cfg, "caar_tau_hidden_size", 0))
        if (
            tau_filters < 1
            or tau_conv_layers < 1
            or tau_res_blocks < 0
            or tau_hidden_size < 0
        ):
            raise ValueError(
                "CAAR tau network requires positive filters and convolution layers, "
                "and non-negative residual-block and hidden sizes."
            )

        if self.contextual_pressure:
            tau_layers = [
                nn.Conv2d(1, tau_filters, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1),
            ]
            for _ in range(1, tau_conv_layers):
                tau_layers.extend(
                    [
                        nn.Conv2d(tau_filters, tau_filters, kernel_size=3, padding=1),
                        nn.LeakyReLU(negative_slope=0.1),
                    ]
                )
            for _ in range(tau_res_blocks):
                tau_layers.append(ResBlock(cfg, tau_filters, tau_filters))
            if tau_res_blocks:
                tau_layers.append(nn.LeakyReLU(negative_slope=0.1))

            flattened_size = tau_filters * tau_shape[1] * tau_shape[2]
            tau_layers.append(nn.Flatten())
            if tau_hidden_size < 1:
                raise ValueError(
                    "Contextual CAAR requires a positive tau hidden size."
                )
            tau_layers.extend(
                [
                    nn.Linear(flattened_size, tau_hidden_size),
                    nn.LeakyReLU(negative_slope=0.1),
                ]
            )
            self.tau_feature_size = tau_hidden_size

            self.tau_action_net = nn.Sequential(*tau_layers)
            pressure_hidden_size = max(self.pressure_size, tau_hidden_size)
            if self.pressure_head_mode == "direct_pressure":
                pressure_input_size = (
                    self.decoder.get_out_size() + self.tau_feature_size
                )
            else:
                pressure_input_size = (
                    self.decoder.get_out_size()
                    + self.NUM_ACTIONS
                    + self.tau_feature_size
                    + self.pressure_size
                )
            pressure_head = nn.Sequential(
                nn.Linear(
                    pressure_input_size,
                    pressure_hidden_size,
                ),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Linear(pressure_hidden_size, self.pressure_size),
            )
            if self.pressure_head_mode == "legacy_multiplier":
                self.pressure_scale_head = pressure_head
                self.pressure_head = None
                nn.init.zeros_(self.pressure_scale_head[-1].weight)
                nn.init.zeros_(self.pressure_scale_head[-1].bias)
            else:
                self.pressure_scale_head = None
                self.pressure_head = pressure_head
                # A zero logit followed by ReLU has no useful gradient.  Start
                # just above zero so PPO initially makes only a small change,
                # while gradients can immediately reach the new pressure head.
                nn.init.normal_(self.pressure_head[-1].weight, mean=0.0, std=1e-3)
                nn.init.constant_(self.pressure_head[-1].bias, self.pressure_init)
            self.aux_size = self.tau_feature_size
            if self.pressure_head_mode == "legacy_multiplier":
                self.aux_size += self.pressure_size
        else:
            self.tau_action_net = None
            self.pressure_scale_head = None
            self.pressure_head = None
            self.tau_feature_size = 0
            self.aux_size = self.pressure_size
        self.last_action_correction = None
        self.last_candidate_pressure = None
        self.last_raw_predicted_pressure = None
        self.last_predicted_pressure = None
        self.last_tau_residual = None
        self.last_movement_adjustment = None
        self.last_pressure_multiplier = None
        self.last_decoder_output = None
        self.last_base_logits = None
        self.last_adjusted_logits = None
        self.last_values = None

    def candidate_pressures(self, tau):
        if tau.ndim != 4 or tau.shape[1] != 1:
            raise ValueError(f"Expected tau shape [B, 1, H, W], got {tuple(tau.shape)}.")
        center_x = tau.shape[2] // 2
        center_y = tau.shape[3] // 2
        return torch.stack(
            [
                tau[:, 0, center_x + dx, center_y + dy]
                for dx, dy in self.pressure_offsets
            ],
            dim=-1,
        )

    def movement_adjustments(self, pressures, residuals):
        if pressures.shape[-1] != self.pressure_size:
            raise ValueError(
                f"Expected {self.pressure_size} movement pressures, "
                f"got shape {tuple(pressures.shape)}."
            )
        if residuals.shape != pressures.shape:
            raise ValueError(
                "Tau residuals must match candidate pressures, got "
                f"{tuple(residuals.shape)} and {tuple(pressures.shape)}."
            )
        return pressures + residuals

    def transform_predicted_pressure(self, raw_pressure):
        """Map the learned five-action output to the P used in ``z - P``."""
        if raw_pressure.shape[-1] != self.NUM_ACTIONS:
            raise ValueError(
                f"Expected five predicted pressures, got {tuple(raw_pressure.shape)}."
            )
        if self.pressure_output_transform == "identity":
            return raw_pressure
        return torch.clamp(torch.relu(raw_pressure), max=self.pressure_cap)

    def forward_head(self, normalized_obs_dict):
        context_features = self.encoder(normalized_obs_dict)
        tau = normalized_obs_dict["tau"]
        if self.contextual_pressure and self.pressure_head_mode == "direct_pressure":
            # The learned P head sees the policy state and an encoding of the
            # complete trace crop.  Raw five-cell trace values and base logits
            # are deliberately not fed back into this branch.
            parts = [context_features, self.tau_action_net(tau)]
        else:
            pressures = self.candidate_pressures(tau)
            parts = [context_features, pressures]
            if self.contextual_pressure:
                parts.append(self.tau_action_net(tau))
        return torch.cat(parts, dim=-1)

    @staticmethod
    def _packed_like(reference, data):
        return PackedSequence(
            data,
            reference.batch_sizes,
            reference.sorted_indices,
            reference.unsorted_indices,
        )

    def forward_core(self, head_output, rnn_states):
        if isinstance(head_output, PackedSequence):
            context = head_output.data[:, :-self.aux_size]
            auxiliary = head_output.data[:, -self.aux_size:]
            context_sequence = self._packed_like(head_output, context)
            core_output, new_rnn_states = self.core(context_sequence, rnn_states)
            combined = torch.cat([core_output.data, auxiliary], dim=-1)
            return self._packed_like(core_output, combined), new_rnn_states

        context = head_output[:, :-self.aux_size]
        auxiliary = head_output[:, -self.aux_size:]
        core_output, new_rnn_states = self.core(context, rnn_states)
        return torch.cat([core_output, auxiliary], dim=-1), new_rnn_states

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool):
        context = core_output[:, : self.core_out_size]
        if self.contextual_pressure and self.pressure_head_mode == "direct_pressure":
            pressures = None
            learned_tau = core_output[:, self.core_out_size :]
        else:
            pressures = core_output[
                :, self.core_out_size : self.core_out_size + self.pressure_size
            ]
            learned_tau = core_output[:, self.core_out_size + self.pressure_size :]
        decoder_output = self.decoder(context)
        values = self.critic_linear(decoder_output).squeeze(-1)
        self.last_decoder_output = decoder_output.detach()
        self.last_values = values.detach()

        result = TensorDict(values=values)
        if values_only:
            return result

        base_logits, _ = self.action_parameterization(decoder_output)
        raw_predicted_pressure = None
        predicted_pressure = None
        if self.contextual_pressure and self.pressure_head_mode == "direct_pressure":
            raw_predicted_pressure = self.pressure_head(
                torch.cat([decoder_output, learned_tau], dim=-1)
            )
            predicted_pressure = self.transform_predicted_pressure(
                raw_predicted_pressure
            )
            adjustments = predicted_pressure
            multipliers = None
            residuals = None
        elif self.contextual_pressure:
            raw_scales = self.pressure_scale_head(
                torch.cat(
                    [decoder_output, base_logits, pressures, learned_tau],
                    dim=-1,
                )
            )
            direction_weights = torch.softmax(raw_scales, dim=-1)
            uniform_weight = 1.0 / self.pressure_size
            multipliers = 1.0 + direction_weights - uniform_weight
            residuals = pressures * (multipliers - 1.0)
            adjustments = self.movement_adjustments(pressures, residuals)
        else:
            multipliers = torch.ones_like(pressures)
            residuals = torch.zeros_like(pressures)
            adjustments = self.movement_adjustments(pressures, residuals)
        if self.reweight_wait_action:
            corrections = -adjustments
        else:
            corrections = torch.cat(
                [torch.zeros_like(base_logits[..., :1]), -adjustments], dim=-1
            )
        adjusted_logits = base_logits + corrections

        self.last_action_correction = corrections.detach()
        self.last_candidate_pressure = (
            pressures.detach() if pressures is not None else None
        )
        self.last_raw_predicted_pressure = (
            raw_predicted_pressure.detach()
            if raw_predicted_pressure is not None
            else None
        )
        self.last_predicted_pressure = (
            predicted_pressure.detach() if predicted_pressure is not None else None
        )
        self.last_tau_residual = residuals.detach() if residuals is not None else None
        self.last_movement_adjustment = adjustments.detach()
        self.last_pressure_multiplier = (
            multipliers.detach() if multipliers is not None else None
        )
        self.last_base_logits = base_logits.detach()
        self.last_adjusted_logits = adjusted_logits.detach()
        self.last_action_distribution = get_action_distribution(
            self.action_space,
            adjusted_logits,
        )
        result["action_logits"] = adjusted_logits
        self._maybe_sample_actions(sample_actions, result)
        return result
