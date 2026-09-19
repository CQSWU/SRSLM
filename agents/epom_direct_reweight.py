"""Frozen EPOM-L with the selected parameter-free Direct rule.

For an agent whose five-action entropy exceeds the fixed PRIMAL3 threshold,
Direct finds the two highest-logit statically legal movement directions.  It
adds one to the lower-pressure direction and leaves wait and every other logit
unchanged.  Pressure is read from the five candidate cells after centring the
entire free-cell region of the 11x11 shared-trace crop.  This module deliberately
contains only the final paper rule; earlier signed, clipped-ReLU and alternative
centring variants were exploratory experiments.
"""

from copy import deepcopy
from typing import Literal

import numpy as np
import torch
from pydantic import Extra
from sample_factory.model.model_utils import get_rnn_size

from agents.epom import EPOM, EPOMConfig
from pomapf_env.stigmergic import AcoState

#: PRIMAL3 eq. 29 / expert_guidance.find_definitive_pc
PRIMAL3_ENTROPY_THRESHOLD = 0.46371241
#: PRIMAL3 eq. 27 / expert_guidance.compute_entropy
PRIMAL3_ENTROPY_EPS = 1e-10
#: POGEMA action order: wait, up, down, left, right
MOVES = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


class EPOMDirectReweightConfig(EPOMConfig, extra=Extra.forbid):
    name: Literal["EPOM-DirectReweight"] = "EPOM-DirectReweight"
    #: selected logit bonus for the lower-pressure top-two direction
    reweight_bonus: float = 1.0
    #: evaporation rate; retention is 1 - tau_rho, saturation is 1 / tau_rho
    tau_rho: float = 0.1
    entropy_threshold: float = PRIMAL3_ENTROPY_THRESHOLD


class EPOMDirectReweight(EPOM):
    """Frozen EPOM-L plus the final entropy-gated top-two Direct rule."""

    def __init__(self, cfg: EPOMDirectReweightConfig):
        super().__init__(cfg)
        self.reweight_bonus = float(cfg.reweight_bonus)
        self.entropy_threshold = float(cfg.entropy_threshold)
        if self.reweight_bonus < 0.0:
            raise ValueError("reweight_bonus must be non-negative.")
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
        self._bonus_rng = np.random.default_rng(
            np.random.SeedSequence([int(self.algo_cfg.seed or 0), 130913, 1])
        )

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

        ``AcoState`` centres the complete free-cell region of the 11x11 crop.
        The five candidate cells are then sampled without a second centring.
        """
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

    def _legal_movements(self, positions):
        obstacles = self.aco._obstacle_mask
        if obstacles is None:
            raise RuntimeError("Direct reweighting needs a static obstacle mask.")
        cells = positions[:, None, :] + np.asarray(MOVES, dtype=np.int64)[None]
        rows, cols = cells[..., 0], cells[..., 1]
        inside = (
            (rows >= 0) & (rows < obstacles.shape[0])
            & (cols >= 0) & (cols < obstacles.shape[1])
        )
        legal = np.zeros_like(inside, dtype=bool)
        legal[inside] = ~obstacles[rows[inside], cols[inside]]
        legal[:, 0] = False
        return legal

    def _top2_bonus(self, pressure, legal, logits, gate):
        """Return a sparse bonus for one lower-pressure top-two direction."""
        eligible_logits = np.where(legal, logits, -np.inf)
        order = np.lexsort(
            (self._bonus_rng.random(logits.shape), eligible_logits), axis=1
        )
        top2 = np.zeros_like(legal)
        np.put_along_axis(top2, order[:, -2:], True, axis=1)
        top2 &= legal
        masked_pressure = np.where(top2, pressure, np.inf)
        tied = top2 & (
            masked_pressure == masked_pressure.min(axis=1, keepdims=True)
        )
        winners = np.where(
            tied, self._bonus_rng.random(pressure.shape), -1.0
        ).argmax(axis=1)
        active = np.flatnonzero(gate & (legal.sum(axis=1) >= 2))
        bonus = np.zeros_like(pressure, dtype=np.float32)
        bonus[active, winners[active]] = self.reweight_bonus
        return bonus

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

        outputs = self._forward_observations(observations)

        logits = outputs["action_logits"].float().cpu().numpy()
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        entropy = -(probabilities
                    * np.log(probabilities + PRIMAL3_ENTROPY_EPS)).sum(axis=1)

        gate = entropy > self.entropy_threshold
        centred = self._candidate_trace(positions)
        legal = self._legal_movements(positions)
        bonus = self._top2_bonus(centred, legal, logits, gate)
        adjusted = logits + bonus

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
            "corrected_agent_fraction": float(np.any(bonus > 0, axis=1).mean()),
            "selected_bonus_mean": float(bonus.max(axis=1).mean()),
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
            "rule": "entropy_gated_top2_lower_pressure_bonus",
            "reweight_bonus": self.reweight_bonus,
            "entropy_threshold": self.entropy_threshold,
            "tau_rho": float(self.algo_cfg.tau_rho),
            "trace_radius": 5,
            "trace_size": 11,
            "centering_definition": (
                "free_cell_mean_over_radius5_crop_then_sample_five_candidates"
            ),
            "eligible_actions": "two_highest_logit_static_legal_movements",
            "wait_logit_unchanged": True,
            "other_logits_unchanged": True,
        }
        return provenance

    def get_name(self):
        return f"EPOM-DirectReweight(top2_bonus={self.reweight_bonus})"


__all__ = ["EPOMDirectReweight", "EPOMDirectReweightConfig",
           "PRIMAL3_ENTROPY_THRESHOLD"]
