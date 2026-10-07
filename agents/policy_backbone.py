import json
from pathlib import Path
from typing import Literal
from copy import deepcopy

import numpy as np
import torch
from pydantic import Extra
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
    milestone_checkpoint: str
    base_weights_path: str | None = None
    base_checkpoint_path: str | None = None


class PolicyBackbone:
    def __init__(self, algo_cfg: PolicyBackboneConfig):
        self.algo_cfg = algo_cfg
        path = algo_cfg.path_to_weights
        device = algo_cfg.device
        self.config_path = (Path(path) / "config.json").resolve()

        register_custom_components()

        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        runtime_config = deepcopy(config["full_config"])
        if algo_cfg.base_weights_path is not None:
            runtime_config["experiment_settings"]["epom_base_weights_path"] = (
                algo_cfg.base_weights_path
            )
        _, flat_config = validate_config(checkpoint_experiment_config(runtime_config))

        flat_config.base_checkpoint_path = algo_cfg.base_checkpoint_path

        env = create_env(flat_config.env, cfg=flat_config, env_config={})
        try:
            if "tau" not in env.observation_space.spaces:
                raise RuntimeError(
                    f"{type(self).__name__} requires a checkpoint trained with "
                    f"the separate tau observation. Checkpoint path: {path}"
                )
            actor_critic = create_actor_critic(
                flat_config, env.observation_space, env.action_space
            )
        finally:
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

        checkpoint = self._load_checkpoint(device)
        actor_critic.load_state_dict(checkpoint["model"], strict=True)

        self.ppo = actor_critic
        self.cfg = flat_config
        self.env_cfg = Environment(**self.cfg.full_config["environment"])
        self.tau_radius = self.env_cfg.tau_radius
        self.rnn_states = None
        self.aco = AcoState(rho=self.env_cfg.tau_rho)
        self.env = None

    def _load_checkpoint(self, device):
        candidate = Path(self.algo_cfg.milestone_checkpoint).expanduser()
        if not candidate.is_absolute():
            candidate = Path(__file__).resolve().parents[1] / candidate
        self.checkpoint_path = candidate.resolve()
        log.info("Loading checkpoint: %s", self.checkpoint_path)
        return torch.load(
            self.checkpoint_path,
            map_location="cpu" if device.type == "mps" else device,
            weights_only=False,
        )

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
