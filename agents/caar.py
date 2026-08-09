import hashlib
import io
import json
from copy import deepcopy
from os.path import join
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import Extra
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.envs.create_env import create_env
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.utils import log

from agents.utils_agents import AlgoBase
from learning.config import Environment
from pomapf_env.stigmergic import AcoState
from pomapf_env.wrappers import MatrixObservationWrapper
from train import register_custom_components, validate_config


class CAARConfig(AlgoBase, extra=Extra.forbid):
    name: Literal["CAAR"] = "CAAR"
    path_to_weights: str = "weights/CAAR/radius_ablation/R5"
    checkpoint_kind: Literal["auto", "latest", "best"] = "auto"


class NoTauConfig(AlgoBase, extra=Extra.forbid):
    name: Literal["NoTau"] = "NoTau"
    path_to_weights: str = "weights/NoTau/NoTau"
    checkpoint_kind: Literal["auto", "latest", "best"] = "auto"


class CAAR:
    """Recurrent policy with congestion-aware action reweighting."""

    USE_PHEROMONE = True

    def __init__(self, algo_cfg: CAARConfig):
        self.algo_cfg = algo_cfg
        path = algo_cfg.path_to_weights
        device = algo_cfg.device
        self.config_path = (Path(path) / "config.json").resolve()

        register_custom_components()

        config, self.config_sha256 = self._load_config_snapshot(
            self.config_path
        )
        _, flat_config = validate_config(config["full_config"])

        env = create_env(flat_config.env, cfg=flat_config, env_config={})
        self.model_uses_tau = "tau" in env.observation_space.spaces
        self.uses_tau = self.USE_PHEROMONE
        if self.model_uses_tau != self.uses_tau:
            expected = "with" if self.uses_tau else "without"
            raise RuntimeError(
                f"{type(self).__name__} requires a checkpoint trained {expected} "
                f"the separate tau observation. Checkpoint path: {path}"
            )
        actor_critic = create_actor_critic(flat_config, env.observation_space, env.action_space)
        env.close()

        if device == "cpu":
            device = torch.device("cpu")
        elif device.startswith("cuda") and torch.cuda.is_available():
            device = torch.device(device)
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        self.device = device

        if device.type == "mps":
            actor_critic.float()
        actor_critic.model_to_device(device)

        policy_id = flat_config.policy_index
        checkpoint_dir = join(path, f"checkpoint_p{policy_id}")
        checkpoint = self._load_checkpoint(checkpoint_dir, device, algo_cfg.checkpoint_kind)
        self._load_model_state(actor_critic, checkpoint["model"], path)

        self.ppo = actor_critic
        self.cfg = flat_config
        self.env_cfg = Environment(**self.cfg.full_config["environment"])
        self.tau_radius = self.env_cfg.tau_radius
        self.rnn_states = None
        self.aco = (
            AcoState(
                rho=self.env_cfg.tau_rho,
            )
            if self.uses_tau
            else None
        )
        self._last_augmented_observations = None
        self._action_correction_samples = []
        self._candidate_pressure_samples = []
        self._tau_residual_samples = []
        self._movement_adjustment_samples = []
        self._pressure_multiplier_samples = []
        self._last_switch_context = None
        self.env = None

    @staticmethod
    def _load_config_snapshot(config_path):
        config_path = Path(config_path).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Could not find {config_path}")
        with config_path.open("rb") as handle:
            payload = handle.read()
        return (
            json.loads(payload.decode("utf-8")),
            hashlib.sha256(payload).hexdigest(),
        )

    @staticmethod
    def _checkpoint_map_location(device):
        return "cpu" if device.type == "mps" else device

    @staticmethod
    def _latest_checkpoint_path(checkpoint_dir):
        checkpoints = Learner.get_checkpoints(checkpoint_dir)
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        return Path(checkpoints[-1]).resolve()

    @staticmethod
    def _best_checkpoint_path(checkpoint_dir):
        checkpoint_dir = Path(checkpoint_dir)
        best_files = sorted(checkpoint_dir.glob("best_*avg_throughput*.pth"))
        if not best_files:
            best_files = sorted(checkpoint_dir.glob("best_*.pth"))
        if not best_files:
            raise FileNotFoundError(f"No best checkpoints found in {checkpoint_dir}")
        return best_files[-1].resolve()

    @classmethod
    def _load_checkpoint_path(cls, checkpoint_path, device, label):
        checkpoint_path = Path(checkpoint_path).resolve()
        log.info("Loading %s checkpoint: %s", label, checkpoint_path)
        with checkpoint_path.open("rb") as handle:
            payload = handle.read()
        checkpoint = torch.load(
            io.BytesIO(payload),
            map_location=cls._checkpoint_map_location(device),
            weights_only=False,
        )
        return checkpoint, hashlib.sha256(payload).hexdigest()

    def _load_checkpoint(self, checkpoint_dir, device, checkpoint_kind):
        if checkpoint_kind == "latest":
            checkpoint_path = self._latest_checkpoint_path(checkpoint_dir)
            label = "latest"
        elif checkpoint_kind == "best":
            checkpoint_path = self._best_checkpoint_path(checkpoint_dir)
            label = "best"
        else:
            try:
                checkpoint_path = self._latest_checkpoint_path(checkpoint_dir)
                label = "latest"
            except FileNotFoundError as latest_error:
                log.warning(
                    "Failed to load latest checkpoint from %s, trying best "
                    "checkpoint: %s",
                    checkpoint_dir,
                    latest_error,
                )
                checkpoint_path = self._best_checkpoint_path(checkpoint_dir)
                label = "best"
        self.checkpoint_path = checkpoint_path
        checkpoint, self.checkpoint_sha256 = self._load_checkpoint_path(
            checkpoint_path,
            device,
            label,
        )
        return checkpoint

    @staticmethod
    def _load_model_state(actor_critic, checkpoint_state, path):
        try:
            actor_critic.load_state_dict(checkpoint_state)
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint architecture does not match this policy. CAAR and NoTau "
                "must use checkpoints with the original three-channel policy backbone. "
                f"Checkpoint path: {path}"
            ) from exc

    def set_grid_config(self, grid_config):
        if self.uses_tau:
            self.aco.configure_from_grid_config(grid_config, clear=True)

    def set_env(self, env):
        self.env = env
        if not self.uses_tau:
            return
        grid_obstacles = getattr(getattr(env, "grid", None), "obstacles", None)
        if grid_obstacles is not None:
            self.aco.configure_from_obstacle_mask(np.asarray(grid_obstacles, dtype=bool), clear=True)

    def after_reset(self):
        torch.manual_seed(self.algo_cfg.seed)
        self.rnn_states = None
        if self.uses_tau:
            self.aco.clear()
        self._last_augmented_observations = None
        self._action_correction_samples = []
        self._candidate_pressure_samples = []
        self._tau_residual_samples = []
        self._movement_adjustment_samples = []
        self._pressure_multiplier_samples = []
        self._last_switch_context = None

    def act(self, observations, rewards=None, dones=None, infos=None):
        raw_observations = deepcopy(observations)
        num_agents = len(raw_observations)

        if self.rnn_states is None or len(self.rnn_states) != num_agents:
            self.rnn_states = torch.zeros(
                [num_agents, get_rnn_size(self.cfg)],
                dtype=torch.float32,
                device=self.device,
            )
        observations = MatrixObservationWrapper.to_matrix(raw_observations)
        if self.uses_tau:
            if self.aco.tau is None:
                raise RuntimeError(
                    "CAAR trace state is not initialized. Call set_grid_config() "
                    "after env.reset() and before act()."
                )
            self.aco.observe_for_inference(
                observations,
                positions=self._global_positions(),
                radius=self.tau_radius,
            )

        self._last_augmented_observations = deepcopy(observations)

        with torch.no_grad():
            obs_torch = TensorDict({
                key: np.stack([obs[key] for obs in observations])
                for key in observations[0]
            })
            for key, value in obs_torch.items():
                obs_torch[key] = torch.from_numpy(value).to(self.device).float()
            obs_torch = prepare_and_normalize_obs(self.ppo, obs_torch)
            policy_outputs = self.ppo(obs_torch, self.rnn_states)
            self.rnn_states = policy_outputs["new_rnn_states"]
            actions = policy_outputs["actions"]
            if self.uses_tau:
                corrections = getattr(self.ppo, "last_action_correction", None)
                if corrections is None:
                    raise RuntimeError("CAAR model did not produce action corrections.")
                self._action_correction_samples.append(
                    corrections.float().cpu().numpy().reshape(-1)
                )
                pressures = getattr(self.ppo, "last_candidate_pressure", None)
                if pressures is None:
                    raise RuntimeError("CAAR model did not expose candidate pressures.")
                self._candidate_pressure_samples.append(
                    pressures.float().cpu().numpy().reshape(-1)
                )
                residuals = getattr(self.ppo, "last_tau_residual", None)
                if residuals is None:
                    raise RuntimeError("CAAR model did not expose tau residuals.")
                self._tau_residual_samples.append(
                    residuals.float().cpu().numpy().reshape(-1)
                )
                adjustments = getattr(self.ppo, "last_movement_adjustment", None)
                if adjustments is None:
                    raise RuntimeError("CAAR model did not expose movement adjustments.")
                self._movement_adjustment_samples.append(
                    adjustments.float().cpu().numpy().reshape(-1)
                )
                multipliers = getattr(self.ppo, "last_pressure_multiplier", None)
                if multipliers is None:
                    raise RuntimeError("CAAR model did not expose pressure multipliers.")
                self._pressure_multiplier_samples.append(
                    multipliers.float().cpu().numpy().reshape(-1)
                )
                switch_tensors = {
                    "decoder_output": getattr(
                        self.ppo, "last_decoder_output", None
                    ),
                    "base_logits": getattr(self.ppo, "last_base_logits", None),
                    "adjusted_logits": getattr(
                        self.ppo, "last_adjusted_logits", None
                    ),
                    "values": getattr(self.ppo, "last_values", None),
                    "candidate_pressure": pressures,
                    "tau_residual": residuals,
                }
                missing = [
                    name
                    for name, value in switch_tensors.items()
                    if value is None
                ]
                if missing:
                    raise RuntimeError(
                        "CAAR model did not expose switch context: "
                        + ", ".join(missing)
                    )
                self._last_switch_context = {
                    name: value.float().cpu().numpy().copy()
                    for name, value in switch_tensors.items()
                }
                self._last_switch_context["actions"] = (
                    actions.detach().cpu().numpy().copy()
                )

        action_array = actions.detach().cpu().numpy()

        return action_array

    def last_augmented_observations(self):
        return self._last_augmented_observations

    def last_switch_context(self):
        """Return frozen CAAR features from the most recent policy decision."""
        return self._last_switch_context

    def get_action_correction_stats(self):
        if not self._action_correction_samples:
            return {}
        values = np.concatenate(self._action_correction_samples)
        stats = {
            "action_correction_mean": float(values.mean()),
            "action_correction_median": float(np.median(values)),
            "action_correction_p05": float(np.quantile(values, 0.05)),
            "action_correction_p95": float(np.quantile(values, 0.95)),
        }
        if self._candidate_pressure_samples:
            pressures = np.concatenate(self._candidate_pressure_samples)
            stats.update(
                {
                    "candidate_pressure_mean": float(pressures.mean()),
                    "candidate_pressure_median": float(np.median(pressures)),
                    "candidate_pressure_p05": float(np.quantile(pressures, 0.05)),
                    "candidate_pressure_p95": float(np.quantile(pressures, 0.95)),
                }
            )
        if self._tau_residual_samples:
            residuals = np.concatenate(self._tau_residual_samples)
            stats.update(
                {
                    "tau_residual_mean": float(residuals.mean()),
                    "tau_residual_abs_mean": float(np.abs(residuals).mean()),
                    "tau_residual_median": float(np.median(residuals)),
                    "tau_residual_p05": float(np.quantile(residuals, 0.05)),
                    "tau_residual_p95": float(np.quantile(residuals, 0.95)),
                }
            )
        if self._movement_adjustment_samples:
            adjustments = np.concatenate(self._movement_adjustment_samples)
            stats.update(
                {
                    "movement_adjustment_mean": float(adjustments.mean()),
                    "movement_adjustment_median": float(np.median(adjustments)),
                    "movement_adjustment_p05": float(np.quantile(adjustments, 0.05)),
                    "movement_adjustment_p95": float(np.quantile(adjustments, 0.95)),
                }
            )
        if self._pressure_multiplier_samples:
            multipliers = np.concatenate(self._pressure_multiplier_samples)
            stats.update(
                {
                    "pressure_multiplier_mean": float(multipliers.mean()),
                    "pressure_multiplier_median": float(np.median(multipliers)),
                    "pressure_multiplier_p05": float(np.quantile(multipliers, 0.05)),
                    "pressure_multiplier_p95": float(np.quantile(multipliers, 0.95)),
                }
            )
        return stats

    def _global_positions(self):
        grid = getattr(self.env, "grid", None) if self.env is not None else None
        positions = getattr(grid, "positions_xy", None) if grid is not None else None
        if positions is None and grid is not None and hasattr(grid, "get_agents_xy"):
            positions = grid.get_agents_xy()
        if positions is None:
            raise RuntimeError(
                "CAAR tau inference requires global grid positions. Call set_env(env) "
                "after env.reset(); raw observation xy is egocentric and cannot be "
                "used for the global tau map."
            )
        return np.asarray(positions, dtype=np.int64)

    def after_step(self, dones):
        if all(dones):
            self.rnn_states = None
            if self.uses_tau:
                self.aco.clear()
            self._last_augmented_observations = None
            self._last_switch_context = None


class NoTau(CAAR):
    """The same recurrent policy without traffic memory or action reweighting."""

    USE_PHEROMONE = False
