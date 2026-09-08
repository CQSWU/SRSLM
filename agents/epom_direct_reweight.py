"""Frozen EPOM-L with a parameter-free trace reweighting rule.

The correction adds no trainable parameters to the trained EPOM-L backbone.
Plain Direct applies signed pressure to every agent. The optional entropy gate
limits this correction to agents whose base action distribution is uncertain:

    H_i   = -sum_u p_i(u) log(p_i(u) + eps)          PRIMAL3 eq. 27
    gate  = H_i > eta,  eta = 0.46371241             PRIMAL3 eq. 28-29
    z'_i  = z_i - w * gate_i * f(tau~_i)

``eta`` is the entropy of the reference distribution [0.9, .025, .025, .025,
.025] under natural logarithms, exactly as in the PRIMAL3 paper and its released
``expert_guidance.find_definitive_pc``.  The entropy is unnormalised and taken
over the raw five-action distribution, matching that implementation.

By default, ``tau~`` uses the mean over every free in-map cell in the agent's
11x11 trace crop, then reads the five candidate cells from that centred crop.
The explicitly selected historical ``candidate`` scope instead centres on the
statically legal candidates' own mean. Obstacles and map padding are excluded
from either mean and have zero correction. Their logits stay unchanged, but
their probabilities can still change when softmax renormalises all five logits.

The default ``signed`` transform uses ``f(x)=x``.  The independent
``clipped_relu`` experiment uses ``f(x)=min(max(x, 0), 2)``: it can penalise a
legal candidate whose pressure is above the selected reference mean, but it never
boosts a below-mean candidate.  Both transforms use the same entropy gate,
frozen EPOM-L backbone, trace, and static-legality definition.

``w = 0`` preserves the base action distribution. Direct uses its own NumPy
sampler, so this does not promise the same sampled trajectory as the base
adapter's Torch sampler.
"""

from copy import deepcopy
from typing import Literal

import numpy as np
import torch
from pydantic import Extra
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.model_utils import get_rnn_size

from agents.epom import EPOM, EPOMConfig
from pomapf_env.stigmergic import AcoState
from pomapf_env.wrappers import MatrixObservationWrapper

#: PRIMAL3 eq. 29 / expert_guidance.find_definitive_pc
PRIMAL3_ENTROPY_THRESHOLD = 0.46371241
#: PRIMAL3 eq. 27 / expert_guidance.compute_entropy
PRIMAL3_ENTROPY_EPS = 1e-10
#: POGEMA action order: wait, up, down, left, right
MOVES = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


class EPOMDirectReweightConfig(EPOMConfig, extra=Extra.forbid):
    name: Literal["EPOM-DirectReweight"] = "EPOM-DirectReweight"
    #: strength of the correction; 0.0 preserves the EPOM-L action distribution
    reweight_scale: float = 1.0
    #: evaporation rate; retention is 1 - tau_rho, saturation is 1 / tau_rho
    tau_rho: float = 0.1
    entropy_threshold: float = PRIMAL3_ENTROPY_THRESHOLD
    #: apply the correction to every agent, to separate gating from reweighting
    gate: Literal["primal3", "always", "never"] = "always"
    #: signed is the current rule; clipped_relu is the conservative ablation
    pressure_transform: Literal["signed", "clipped_relu"] = "signed"
    pressure_cap: float = 2.0
    #: candidate preserves the historical five-cell mean; crop uses all free
    #: cells in the radius-5 (11x11) local trace crop.
    centering_scope: Literal["candidate", "crop"] = "crop"


class EPOMDirectReweight(EPOM):
    """Frozen EPOM-L plus signed trace correction, with optional entropy gating."""

    def __init__(self, cfg: EPOMDirectReweightConfig):
        super().__init__(cfg)
        self.reweight_scale = float(cfg.reweight_scale)
        self.entropy_threshold = float(cfg.entropy_threshold)
        self.gate_mode = cfg.gate
        self.pressure_transform = cfg.pressure_transform
        self.pressure_cap = float(cfg.pressure_cap)
        self.centering_scope = cfg.centering_scope
        if self.pressure_cap <= 0.0:
            raise ValueError("pressure_cap must be positive.")
        self.aco = AcoState(rho=float(cfg.tau_rho))
        self.env = None
        self._stats = []

    # -------------------------------------------------------------- plumbing

    def set_grid_config(self, grid_config):
        self.aco.configure_from_grid_config(grid_config, clear=True)

    def set_env(self, env):
        self.env = env
        obstacles = getattr(getattr(env, "grid", None), "obstacles", None)
        if obstacles is not None:
            self.aco.configure_from_obstacle_mask(
                np.asarray(obstacles, dtype=bool), clear=True
            )

    def after_reset(self):
        super().after_reset()
        self.aco.clear()
        self._stats = []
        self._numpy_rng = np.random.default_rng(self.algo_cfg.seed)

    def after_step(self, dones):
        super().after_step(dones)
        if all(dones):
            self.aco.clear()

    def _global_positions(self):
        grid = getattr(self.env, "grid", None) if self.env is not None else None
        positions = getattr(grid, "positions_xy", None) if grid is not None else None
        if positions is None:
            raise RuntimeError(
                "Direct reweighting needs global positions for the shared trace. "
                "Call set_env(env) after env.reset()."
            )
        return np.asarray(positions, dtype=np.int64)

    # ------------------------------------------------------------ correction

    def _candidate_trace(self, positions):
        """Return pressures for wait/up/down/left/right.

        ``candidate`` is the historical behaviour: centre the five statically
        legal candidate values on their own mean.  ``crop`` asks ``AcoState``
        for the free-cell-mean-centred 11x11 crop and samples its five candidate
        cells without centring those five values a second time.
        """
        if self.centering_scope == "crop":
            radius = 5
            center = radius
            indices = (
                (center, center),
                (center - 1, center),
                (center + 1, center),
                (center, center - 1),
                (center, center + 1),
            )
            result = np.empty((len(positions), len(indices)), dtype=np.float32)
            for agent_index, (x, y) in enumerate(positions):
                local = self.aco.extract_local_tau(int(x), int(y), radius)
                result[agent_index] = [local[row, col] for row, col in indices]
            return result

        tau = self.aco.tau
        height, width = tau.shape
        moves = np.asarray(MOVES, dtype=np.int64)              # [5, 2]
        cells = positions[:, None, :] + moves[None, :, :]      # [N, 5, 2]

        rows, cols = cells[..., 0], cells[..., 1]
        inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        clipped_rows = np.clip(rows, 0, height - 1)
        clipped_cols = np.clip(cols, 0, width - 1)

        legal = inside.copy()
        obstacles = self.aco._obstacle_mask
        if obstacles is not None:
            legal &= ~obstacles[clipped_rows, clipped_cols]

        values = np.where(legal, tau[clipped_rows, clipped_cols], 0.0)
        count = legal.sum(axis=1, keepdims=True)
        mean = np.divide(
            values.sum(axis=1, keepdims=True), count,
            out=np.zeros_like(values[:, :1]), where=count > 0)
        return np.where(legal, values - mean, 0.0).astype(np.float32)

    @staticmethod
    def transform_pressure(centered, transform="signed", cap=2.0):
        """Transform centred pressure; zero obstacle corrections stay zero."""
        centered = np.asarray(centered, dtype=np.float32)
        if transform == "signed":
            return centered.copy()
        if transform == "clipped_relu":
            return np.minimum(np.maximum(centered, 0.0), float(cap)).astype(
                np.float32, copy=False
            )
        raise ValueError(f"Unknown pressure transform: {transform}")

    # ----------------------------------------------------------------- act

    def act(self, observations, rewards=None, dones=None, infos=None):
        observations = deepcopy(observations)
        num_agents = len(observations)
        if self.rnn_states is None or len(self.rnn_states) != num_agents:
            self.rnn_states = torch.zeros(
                (num_agents, get_rnn_size(self.cfg)),
                dtype=torch.float32,
                device=self.device,
            )

        positions = self._global_positions()
        if self.aco.tau is None or self.aco.prev_positions is None:
            self.aco.reset_episode(observations, positions=positions)
        else:
            self.aco.observe_for_inference(observations, positions=positions)

        self.grid_memory.update(observations)
        self.grid_memory.modify_observation(observations, self.grid_memory_radius)
        matrix = MatrixObservationWrapper.to_matrix(observations)
        obs_torch = TensorDict({
            key: torch.from_numpy(
                np.stack([obs[key] for obs in matrix])
            ).to(self.device).float()
            for key in matrix[0]
        })
        model_input = (
            prepare_and_normalize_obs(self.ppo, obs_torch)
            if self._use_obs_normalization
            else obs_torch
        )
        with torch.no_grad():
            outputs = self.ppo(model_input, self.rnn_states)
        self.rnn_states = outputs["new_rnn_states"]

        logits = outputs["action_logits"].float().cpu().numpy()
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        entropy = -(probabilities
                    * np.log(probabilities + PRIMAL3_ENTROPY_EPS)).sum(axis=1)

        if self.gate_mode == "primal3":
            gate = (entropy > self.entropy_threshold).astype(np.float32)
        elif self.gate_mode == "always":
            gate = np.ones_like(entropy, dtype=np.float32)
        else:
            gate = np.zeros_like(entropy, dtype=np.float32)

        centred = self._candidate_trace(positions)
        pressure = self.transform_pressure(
            centred,
            transform=self.pressure_transform,
            cap=self.pressure_cap,
        )
        adjusted = logits - self.reweight_scale * gate[:, None] * pressure

        shifted = np.exp(adjusted - adjusted.max(axis=1, keepdims=True))
        shifted /= shifted.sum(axis=1, keepdims=True)
        actions = np.array([
            self._rng.choice(len(MOVES), p=row) for row in shifted
        ], dtype=np.int64)

        self._stats.append({
            "gated_fraction": float(gate.mean()),
            "entropy_mean": float(entropy.mean()),
            "logit_spread": float(
                (logits.max(axis=1) - logits.min(axis=1)).mean()),
            "trace_spread": float(
                (centred.max(axis=1) - centred.min(axis=1)).mean()),
            "applied_pressure_spread": float(
                (pressure.max(axis=1) - pressure.min(axis=1)).mean()),
            "argmax_flip_rate": float(
                (logits.argmax(1) != adjusted.argmax(1)).mean()),
            "wait_prob": float(shifted[:, 0].mean()),
        })
        return actions

    @property
    def _rng(self):
        if not hasattr(self, "_numpy_rng"):
            self._numpy_rng = np.random.default_rng(self.algo_cfg.seed)
        return self._numpy_rng

    def get_action_correction_stats(self):
        if not self._stats:
            return {}
        return {
            f"direct_{key}": float(np.mean([s[key] for s in self._stats]))
            for key in self._stats[0]
        }

    def get_model_provenance(self):
        """Extend the frozen EPOM provenance with the exact Direct rule."""

        provenance = dict(super().get_model_provenance())
        provenance["direct_reweight"] = {
            "reweight_scale": self.reweight_scale,
            "gate": self.gate_mode,
            "entropy_threshold": self.entropy_threshold,
            "tau_rho": float(self.algo_cfg.tau_rho),
            "trace_radius": 5,
            "trace_size": 11,
            "pressure_transform": self.pressure_transform,
            "pressure_cap": self.pressure_cap,
            "centering_scope": self.centering_scope,
            "centering_definition": (
                "free_cell_mean_over_radius5_crop_then_sample_five_candidates"
                if self.centering_scope == "crop"
                else "static_legal_five_candidate_mean"
            ),
            "legal_action_definition": "inside_map_and_not_static_obstacle",
        }
        return provenance

    def get_name(self):
        name = (
            "EPOM-DirectReweight("
            f"w={self.reweight_scale}, transform={self.pressure_transform}"
        )
        if self.centering_scope != "candidate":
            name += f", centering={self.centering_scope}"
        return name + ")"


__all__ = ["EPOMDirectReweight", "EPOMDirectReweightConfig",
           "PRIMAL3_ENTROPY_THRESHOLD"]
