"""Frozen NoReweight policy with fixed shared-trace logit subtraction."""

from copy import deepcopy
import os
from typing import Literal

import numpy as np
import torch
from pydantic import Extra
from sample_factory.algo.utils.action_distributions import (
    get_action_distribution,
    sample_actions_log_probs,
)
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.model_utils import get_rnn_size

from agents.caar import NoReweight, NoReweightConfig
from pomapf_env.stigmergic import AcoState
from pomapf_env.wrappers import MatrixObservationWrapper


class DirectConfig(NoReweightConfig, extra=Extra.forbid):
    name: Literal["Direct", "Direct-0P"] = "Direct"
    tau_rho: float = 0.1
    tau_radius: int = 5
    pressure_scale: float = 1.0


class Direct(NoReweight):
    """Keep NoReweight frozen and subtract five shared-trace pressures."""

    ACTION_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
    ACTION_NAMES = ("wait", "up", "down", "left", "right")
    PRESSURE_CAP = 2.0

    def __init__(self, cfg: DirectConfig):
        super().__init__(cfg)
        self.tau_radius = int(cfg.tau_radius)
        self.pressure_scale = float(cfg.pressure_scale)
        self.pressure_cap = self.PRESSURE_CAP
        self.obstacle_trace_value = 0.0
        if self.pressure_scale < 0.0:
            raise ValueError("pressure_scale must be non-negative.")
        self.aco = AcoState(rho=cfg.tau_rho)
        self.env = None
        self._correction_samples = []
        self._pressure_samples = []
        self._base_logit_range_samples = []
        self._correction_range_samples = []
        self._argmax_change_samples = []
        self._policy_kl_samples = []
        self._base_logits_by_action = []
        self._adjusted_logits_by_action = []
        self._corrections_by_action = []
        self._pressures_by_action = []
        self._base_probabilities_by_action = []
        self._adjusted_probabilities_by_action = []
        self._sampled_actions_by_step = []
        self._positions_by_step = []
        self._snapshot_step = int(
            os.environ.get("DIRECT_SNAPSHOT_STEP", "-1")
        )
        self._snapshot_agent = int(
            os.environ.get("DIRECT_SNAPSHOT_AGENT", "-1")
        )
        self._targeted_trace_snapshot = None

    def set_grid_config(self, grid_config):
        self.aco.configure_from_grid_config(grid_config, clear=True)

    def set_env(self, env):
        self.env = env
        obstacles = getattr(getattr(env, "grid", None), "obstacles", None)
        if obstacles is not None:
            self.aco.configure_from_obstacle_mask(
                np.asarray(obstacles, dtype=bool),
                clear=True,
            )

    def after_reset(self):
        super().after_reset()
        self.aco.clear()
        self._correction_samples = []
        self._pressure_samples = []
        self._base_logit_range_samples = []
        self._correction_range_samples = []
        self._argmax_change_samples = []
        self._policy_kl_samples = []
        self._base_logits_by_action = []
        self._adjusted_logits_by_action = []
        self._corrections_by_action = []
        self._pressures_by_action = []
        self._base_probabilities_by_action = []
        self._adjusted_probabilities_by_action = []
        self._sampled_actions_by_step = []
        self._positions_by_step = []
        self._targeted_trace_snapshot = None

    def _global_positions(self):
        grid = getattr(self.env, "grid", None) if self.env is not None else None
        positions = getattr(grid, "positions_xy", None) if grid is not None else None
        if positions is None and grid is not None and hasattr(grid, "get_agents_xy"):
            positions = grid.get_agents_xy()
        if positions is None:
            raise RuntimeError(
                "Direct requires global positions; call set_env() after reset."
            )
        return np.asarray(positions, dtype=np.int64)

    def _candidate_free_mask(self, positions):
        height, width = self.aco.tau.shape
        obstacle_mask = self.aco._obstacle_mask
        free = np.ones((len(positions), len(self.ACTION_OFFSETS)), dtype=bool)
        for agent, (x, y) in enumerate(positions):
            for action, (dx, dy) in enumerate(self.ACTION_OFFSETS):
                candidate_x = int(x) + dx
                candidate_y = int(y) + dy
                in_bounds = (
                    0 <= candidate_x < height
                    and 0 <= candidate_y < width
                )
                free[agent, action] = bool(
                    in_bounds
                    and (
                        obstacle_mask is None
                        or not obstacle_mask[candidate_x, candidate_y]
                    )
                )
        return torch.from_numpy(free).to(self.device)

    @classmethod
    def candidate_pressures(cls, tau):
        if tau.ndim != 4 or tau.shape[1] != 1:
            raise ValueError(f"Expected tau shape [B, 1, H, W], got {tau.shape}.")
        center_x = tau.shape[2] // 2
        center_y = tau.shape[3] // 2
        return torch.stack(
            [
                tau[:, 0, center_x + dx, center_y + dy]
                for dx, dy in cls.ACTION_OFFSETS
            ],
            dim=-1,
        )

    @staticmethod
    def transform_pressures(centered_pressures):
        """Apply the fixed capped ReLU to centered shared-trace pressure."""
        return torch.clamp(
            torch.relu(centered_pressures),
            max=Direct.PRESSURE_CAP,
        )

    def adjust_logits(self, base_logits, pressures):
        if base_logits.shape != pressures.shape:
            raise ValueError(
                "NoReweight logits and Direct pressures must have the same shape, got "
                f"{tuple(base_logits.shape)} and {tuple(pressures.shape)}."
            )
        return base_logits - self.pressure_scale * pressures

    def act(self, observations, rewards=None, dones=None, infos=None):
        del rewards, dones, infos
        observations = MatrixObservationWrapper.to_matrix(deepcopy(observations))
        num_agents = len(observations)
        if self.rnn_states is None or len(self.rnn_states) != num_agents:
            self.rnn_states = torch.zeros(
                (num_agents, get_rnn_size(self.cfg)),
                dtype=torch.float32,
                device=self.device,
            )
        if self.aco.tau is None:
            raise RuntimeError(
                "Direct trace is not initialized; call set_env() after reset."
            )
        positions = self._global_positions()
        current_step = len(self._sampled_actions_by_step)
        if (
            self._targeted_trace_snapshot is not None
            and current_step == self._snapshot_step + 1
        ):
            agent = self._snapshot_agent
            if 0 <= agent < len(positions):
                previous = np.asarray(
                    self._targeted_trace_snapshot["position_xy"],
                    dtype=np.int64,
                )
                current = np.asarray(positions[agent], dtype=np.int64)
                action_index = int(
                    self._targeted_trace_snapshot["sampled_action_index"]
                )
                offset = np.asarray(self.ACTION_OFFSETS[action_index])
                attempted = previous + offset
                self._targeted_trace_snapshot.update(
                    {
                        "next_position_xy": [
                            int(current[0]),
                            int(current[1]),
                        ],
                        "attempted_position_xy": [
                            int(attempted[0]),
                            int(attempted[1]),
                        ],
                        "submitted_move_succeeded": bool(
                            np.array_equal(current, attempted)
                        ),
                    }
                )
        self.aco.observe_for_inference(
            observations,
            positions=positions,
            radius=self.tau_radius,
        )

        tau = torch.from_numpy(
            np.stack([observation.pop("tau") for observation in observations])
        ).to(self.device).float()
        centered_pressures = self.candidate_pressures(tau)
        pressures = self.transform_pressures(centered_pressures)
        obs_torch = TensorDict(
            {
                key: torch.from_numpy(
                    np.stack([observation[key] for observation in observations])
                ).to(self.device).float()
                for key in observations[0]
            }
        )
        model_input = prepare_and_normalize_obs(self.ppo, obs_torch)

        with torch.no_grad():
            head = self.ppo.forward_head(model_input)
            core, new_rnn_states = self.ppo.forward_core(head, self.rnn_states)
            outputs = self.ppo.forward_tail(
                core,
                values_only=False,
                sample_actions=False,
            )
            base_logits = outputs["action_logits"]
            adjusted_logits = self.adjust_logits(base_logits, pressures)
            base_log_probs = torch.log_softmax(base_logits, dim=-1)
            adjusted_log_probs = torch.log_softmax(adjusted_logits, dim=-1)
            base_probs = base_log_probs.exp()
            policy_kl = torch.sum(
                base_probs * (base_log_probs - adjusted_log_probs),
                dim=-1,
            )
            distribution = get_action_distribution(
                self.ppo.action_space,
                adjusted_logits,
            )
            actions, _ = sample_actions_log_probs(distribution)

        self.rnn_states = new_rnn_states
        corrections = adjusted_logits - base_logits
        self._correction_samples.append(corrections.cpu().numpy().reshape(-1))
        self._pressure_samples.append(pressures.cpu().numpy().reshape(-1))
        self._base_logit_range_samples.append(
            (base_logits.max(dim=-1).values - base_logits.min(dim=-1).values)
            .cpu()
            .numpy()
            .reshape(-1)
        )
        self._correction_range_samples.append(
            (corrections.max(dim=-1).values - corrections.min(dim=-1).values)
            .cpu()
            .numpy()
            .reshape(-1)
        )
        self._argmax_change_samples.append(
            (base_logits.argmax(dim=-1) != adjusted_logits.argmax(dim=-1))
            .float()
            .cpu()
            .numpy()
            .reshape(-1)
        )
        self._policy_kl_samples.append(policy_kl.cpu().numpy().reshape(-1))
        self._base_logits_by_action.append(base_logits.cpu().numpy())
        self._adjusted_logits_by_action.append(adjusted_logits.cpu().numpy())
        self._corrections_by_action.append(corrections.cpu().numpy())
        self._pressures_by_action.append(pressures.cpu().numpy())
        self._base_probabilities_by_action.append(base_probs.cpu().numpy())
        self._adjusted_probabilities_by_action.append(
            adjusted_log_probs.exp().cpu().numpy()
        )
        sampled_actions = actions.squeeze(dim=1).cpu().numpy()
        if (
            current_step == self._snapshot_step
            and 0 <= self._snapshot_agent < num_agents
        ):
            agent = self._snapshot_agent
            x, y = (int(value) for value in positions[agent])
            raw_local, free_mask = self.aco._local_tau_and_free_mask(
                x,
                y,
                self.tau_radius,
            )
            centered_local = self.aco._relative_pressure(
                raw_local.copy(),
                free_mask,
            )
            targets = getattr(
                getattr(self.env, "grid", None),
                "finishes_xy",
                None,
            )
            targets = None if targets is None else np.asarray(targets)
            local_agents = []
            for neighbor, (neighbor_x, neighbor_y) in enumerate(positions):
                dx = int(neighbor_x) - x
                dy = int(neighbor_y) - y
                if abs(dx) > self.tau_radius or abs(dy) > self.tau_radius:
                    continue
                item = {
                    "agent": int(neighbor),
                    "position_xy": [int(neighbor_x), int(neighbor_y)],
                    "relative_xy": [dx, dy],
                    "sampled_action": self.ACTION_NAMES[
                        int(sampled_actions[neighbor])
                    ],
                    "base_argmax": self.ACTION_NAMES[
                        int(base_logits[neighbor].argmax().item())
                    ],
                    "adjusted_argmax": self.ACTION_NAMES[
                        int(adjusted_logits[neighbor].argmax().item())
                    ],
                }
                if targets is not None and neighbor < len(targets):
                    item["target_xy"] = [
                        int(targets[neighbor][0]),
                        int(targets[neighbor][1]),
                    ]
                local_agents.append(item)

            center = self.tau_radius
            candidate_raw = [
                float(raw_local[center + dx, center + dy])
                for dx, dy in self.ACTION_OFFSETS
            ]
            candidate_free = [
                bool(free_mask[center + dx, center + dy])
                for dx, dy in self.ACTION_OFFSETS
            ]
            self._targeted_trace_snapshot = {
                "step": int(current_step),
                "agent": int(agent),
                "position_xy": [x, y],
                "target_xy": (
                    [int(targets[agent][0]), int(targets[agent][1])]
                    if targets is not None and agent < len(targets)
                    else None
                ),
                "radius": int(self.tau_radius),
                "action_order": list(self.ACTION_NAMES),
                "base_logits": self._rounded_vector(
                    base_logits[agent].cpu().numpy()
                ),
                "candidate_raw_trace": candidate_raw,
                "candidate_effective_trace": [
                    (
                        float(value)
                        if is_free
                        else self.obstacle_trace_value
                    )
                    for value, is_free in zip(candidate_raw, candidate_free)
                ],
                "candidate_raw_trace_mean": float(np.mean(candidate_raw)),
                "local_free_trace_mean": float(
                    raw_local[free_mask].mean()
                ),
                "candidate_centered_pressure": self._rounded_vector(
                    pressures[agent].cpu().numpy()
                ),
                "candidate_is_free": candidate_free,
                "corrections": self._rounded_vector(
                    corrections[agent].cpu().numpy()
                ),
                "adjusted_logits": self._rounded_vector(
                    adjusted_logits[agent].cpu().numpy()
                ),
                "base_probabilities": self._rounded_vector(
                    base_probs[agent].cpu().numpy()
                ),
                "adjusted_probabilities": self._rounded_vector(
                    adjusted_log_probs.exp()[agent].cpu().numpy()
                ),
                "sampled_action_index": int(sampled_actions[agent]),
                "sampled_action": self.ACTION_NAMES[int(sampled_actions[agent])],
                "raw_trace_11x11": raw_local.astype(float).tolist(),
                "centered_pressure_11x11": centered_local.astype(float).tolist(),
                "free_mask_11x11": free_mask.astype(int).tolist(),
                "local_agents": local_agents,
                "recent_agent_positions": [
                    [int(item[agent][0]), int(item[agent][1])]
                    for item in self._positions_by_step[
                        max(0, current_step - 20) : current_step
                    ]
                ],
            }
        self._sampled_actions_by_step.append(sampled_actions.copy())
        self._positions_by_step.append(positions.copy())
        return sampled_actions

    @staticmethod
    def _rounded_vector(values):
        return [float(value) for value in np.asarray(values).reshape(-1)]

    def _decision_example(self, selection, step, agent):
        base_logits = self._base_logits_by_action[step][agent]
        adjusted_logits = self._adjusted_logits_by_action[step][agent]
        corrections = self._corrections_by_action[step][agent]
        pressures = self._pressures_by_action[step][agent]
        base_probabilities = self._base_probabilities_by_action[step][agent]
        adjusted_probabilities = self._adjusted_probabilities_by_action[step][agent]
        base_log_probabilities = np.log(np.maximum(base_probabilities, 1e-30))
        adjusted_log_probabilities = np.log(
            np.maximum(adjusted_probabilities, 1e-30)
        )
        kl = np.sum(
            base_probabilities
            * (base_log_probabilities - adjusted_log_probabilities)
        )
        sampled_action = int(self._sampled_actions_by_step[step][agent])
        base_argmax = int(np.argmax(base_logits))
        adjusted_argmax = int(np.argmax(adjusted_logits))
        position = self._positions_by_step[step][agent]
        return {
            "selection": selection,
            "step": int(step),
            "agent": int(agent),
            "position_xy": [int(position[0]), int(position[1])],
            "action_order": list(self.ACTION_NAMES),
            "base_logits": self._rounded_vector(base_logits),
            "candidate_pressures": self._rounded_vector(pressures),
            "corrections": self._rounded_vector(corrections),
            "adjusted_logits": self._rounded_vector(adjusted_logits),
            "base_probabilities": self._rounded_vector(base_probabilities),
            "adjusted_probabilities": self._rounded_vector(
                adjusted_probabilities
            ),
            "probability_delta": self._rounded_vector(
                adjusted_probabilities - base_probabilities
            ),
            "base_argmax": self.ACTION_NAMES[base_argmax],
            "adjusted_argmax": self.ACTION_NAMES[adjusted_argmax],
            "sampled_action": self.ACTION_NAMES[sampled_action],
            "argmax_changed": bool(base_argmax != adjusted_argmax),
            "base_to_adjusted_kl": float(kl),
        }

    def _actual_decision_examples(self):
        if not self._sampled_actions_by_step:
            return []

        num_steps = len(self._sampled_actions_by_step)
        num_agents = len(self._sampled_actions_by_step[0])
        selected = []
        selected_keys = set()

        def add(selection, step, agent):
            key = (int(step), int(agent))
            if key in selected_keys:
                return
            selected_keys.add(key)
            selected.append(self._decision_example(selection, *key))

        fixed_steps = (0, 1, 2, 5, 10, 20, 50, 100, 200, 511)
        for step in fixed_steps:
            if step < num_steps:
                add("fixed_agent_0", step, 0)

        argmax_changes = np.stack(self._argmax_change_samples, axis=0)
        changed_indices = np.argwhere(argmax_changes > 0.5)
        for step, agent in changed_indices[:5]:
            add("first_argmax_change", step, agent)

        policy_kl = np.stack(self._policy_kl_samples, axis=0)
        flat_kl = policy_kl.reshape(-1)
        top_count = min(5, flat_kl.size)
        if top_count:
            top_indices = np.argpartition(flat_kl, -top_count)[-top_count:]
            top_indices = top_indices[np.argsort(flat_kl[top_indices])[::-1]]
            for flat_index in top_indices:
                step, agent = divmod(int(flat_index), num_agents)
                add("largest_probability_change", step, agent)

        return selected

    def get_action_correction_stats(self):
        if not self._correction_samples:
            return {}
        corrections = np.concatenate(self._correction_samples)
        pressures = np.concatenate(self._pressure_samples)
        stats = {
            "pressure_scale": self.pressure_scale,
            "pressure_cap": self.pressure_cap,
            "action_correction_mean": float(corrections.mean()),
            "action_correction_median": float(np.median(corrections)),
            "action_correction_p05": float(np.quantile(corrections, 0.05)),
            "action_correction_p95": float(np.quantile(corrections, 0.95)),
            "candidate_pressure_mean": float(pressures.mean()),
            "candidate_pressure_median": float(np.median(pressures)),
            "candidate_pressure_p05": float(np.quantile(pressures, 0.05)),
            "candidate_pressure_p95": float(np.quantile(pressures, 0.95)),
        }
        if self._base_logit_range_samples:
            base_ranges = np.concatenate(self._base_logit_range_samples)
            correction_ranges = np.concatenate(self._correction_range_samples)
            argmax_changes = np.concatenate(self._argmax_change_samples)
            policy_kl = np.concatenate(self._policy_kl_samples)
            stats.update(
                {
                    "base_logit_range_mean": float(base_ranges.mean()),
                    "correction_range_mean": float(correction_ranges.mean()),
                    "correction_to_logit_range_ratio": float(
                        correction_ranges.mean() / max(base_ranges.mean(), 1e-12)
                    ),
                    "argmax_change_rate": float(argmax_changes.mean()),
                    "base_to_adjusted_kl_mean": float(policy_kl.mean()),
                }
            )
        if self._base_logits_by_action:
            action_arrays = {
                "base_logit": np.concatenate(self._base_logits_by_action, axis=0),
                "adjusted_logit": np.concatenate(
                    self._adjusted_logits_by_action,
                    axis=0,
                ),
                "action_correction": np.concatenate(
                    self._corrections_by_action,
                    axis=0,
                ),
                "candidate_pressure": np.concatenate(
                    self._pressures_by_action,
                    axis=0,
                ),
                "base_probability": np.concatenate(
                    self._base_probabilities_by_action,
                    axis=0,
                ),
                "adjusted_probability": np.concatenate(
                    self._adjusted_probabilities_by_action,
                    axis=0,
                ),
            }
            action_arrays["probability_delta"] = (
                action_arrays["adjusted_probability"]
                - action_arrays["base_probability"]
            )
            for prefix, values in action_arrays.items():
                for action_index, action_name in enumerate(self.ACTION_NAMES):
                    action_values = values[:, action_index]
                    stats.update(
                        {
                            f"{prefix}_{action_name}_mean": float(
                                action_values.mean()
                            ),
                            f"{prefix}_{action_name}_p05": float(
                                np.quantile(action_values, 0.05)
                            ),
                            f"{prefix}_{action_name}_median": float(
                                np.median(action_values)
                            ),
                            f"{prefix}_{action_name}_p95": float(
                                np.quantile(action_values, 0.95)
                            ),
                        }
                    )
            stats["actual_decision_examples"] = self._actual_decision_examples()
        if self._targeted_trace_snapshot is not None:
            stats["targeted_trace_snapshot"] = self._targeted_trace_snapshot
        return stats

    def after_step(self, dones):
        super().after_step(dones)
        if all(dones):
            self.aco.clear()
