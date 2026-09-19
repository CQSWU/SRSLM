import hashlib
import io
import json
from os.path import join
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import Extra
from sample_factory.algo.learning.learner import Learner
from sample_factory.envs.create_env import create_env
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.utils.utils import log

from agents.utils_agents import AlgoBase
from learning.config import Environment, checkpoint_experiment_config
from pomapf_env.stigmergic import AcoState
from train import register_custom_components, validate_config


class PolicyBackboneConfig(AlgoBase, extra=Extra.forbid):
    name: Literal["PolicyBackbone"] = "PolicyBackbone"
    path_to_weights: str
    checkpoint_kind: Literal["auto", "latest", "best"] = "auto"


class PolicyBackbone:
    """Shared checkpoint-loading backbone for trace-aware EPOM policies."""

    def __init__(self, algo_cfg: PolicyBackboneConfig):
        self.algo_cfg = algo_cfg
        path = algo_cfg.path_to_weights
        device = algo_cfg.device
        self.config_path = (Path(path) / "config.json").resolve()

        register_custom_components()

        config, self.config_sha256 = self._load_config_snapshot(
            self.config_path
        )
        _, flat_config = validate_config(checkpoint_experiment_config(config["full_config"]))

        env = create_env(flat_config.env, cfg=flat_config, env_config={})
        if "tau" not in env.observation_space.spaces:
            raise RuntimeError(
                f"{type(self).__name__} requires a checkpoint trained with "
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
        self.aco = AcoState(rho=self.env_cfg.tau_rho)
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
        """Reject architecture or forward-rule mismatches before loading tensors."""

        current = actor_critic.state_dict()
        missing = sorted(current.keys() - checkpoint_state.keys())
        unexpected = sorted(checkpoint_state.keys() - current.keys())
        shape_mismatches = [
            f"{key}: checkpoint={tuple(checkpoint_state[key].shape)}, "
            f"model={tuple(current[key].shape)}"
            for key in sorted(current.keys() & checkpoint_state.keys())
            if checkpoint_state[key].shape != current[key].shape
        ]
        # load_state_dict(strict=True) checks names and shapes, but would happily
        # overwrite a version marker with one for a different forward equation.
        semantic_buffers = {
            "fixed_entropy_threshold",
            "paper_entropy_gate_version",
            "independent_critic_version",
            "allaction_residual_version",
        }
        semantic_mismatches = [
            key
            for key in sorted(semantic_buffers & current.keys() & checkpoint_state.keys())
            if checkpoint_state[key].dtype != current[key].dtype
            or not torch.equal(
                checkpoint_state[key].detach().cpu(), current[key].detach().cpu()
            )
        ]
        if missing or unexpected or shape_mismatches or semantic_mismatches:
            raise RuntimeError(
                "Checkpoint architecture or forward-rule contract does not match "
                "this policy; no tensors were loaded. "
                f"Checkpoint path: {path}; missing={missing}, unexpected={unexpected}, "
                f"shape_mismatches={shape_mismatches}, "
                f"semantic_mismatches={semantic_mismatches}"
            )
        actor_critic.load_state_dict(checkpoint_state, strict=True)

    def set_grid_config(self, grid_config):
        self.aco.configure_from_grid_config(grid_config, clear=True)

    def set_env(self, env):
        self.env = env
        grid_obstacles = getattr(getattr(env, "grid", None), "obstacles", None)
        if grid_obstacles is not None:
            self.aco.configure_from_obstacle_mask(
                np.asarray(grid_obstacles, dtype=bool), clear=True
            )

    def after_reset(self):
        torch.manual_seed(self.algo_cfg.seed)
        self.rnn_states = None
        self.aco.clear()

    def _global_positions(self):
        grid = getattr(self.env, "grid", None) if self.env is not None else None
        positions = getattr(grid, "positions_xy", None) if grid is not None else None
        if positions is None and grid is not None and hasattr(grid, "get_agents_xy"):
            positions = grid.get_agents_xy()
        if positions is None:
            raise RuntimeError(
                "ARPE tau inference requires global grid positions. Call set_env(env) "
                "after env.reset(); raw observation xy is egocentric and cannot be "
                "used for the global tau map."
            )
        return np.asarray(positions, dtype=np.int64)

    def after_step(self, dones):
        if all(dones):
            self.rnn_states = None
            self.aco.clear()
