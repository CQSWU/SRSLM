"""Checkpoint loader for a Switcher trained with the selected ARPE branch."""

from __future__ import annotations

import hashlib
import io
import json
from copy import deepcopy
from os.path import join
from pathlib import Path
from typing import Literal

import gymnasium as gym
import torch
from pydantic import Extra
from sample_factory.model.actor_critic import create_actor_critic

from agents.switcher import Switcher, SwitcherConfig
from agents.arpe import ArpeCandidateArtifact
from agents.switcher_core import NUM_BRANCHES
from pomapf_env.switcher_arpe_env import switcher_observation_space
from train import register_custom_components, validate_config


ARPE_SWITCHER_LOADER_SCHEMA = "switcher_caar_checkpoint_loader_v1"


class AllStateArpeSwitcherConfig(SwitcherConfig, extra=Extra.forbid):
    name: Literal["AllStateArpeSwitcher"] = "AllStateArpeSwitcher"
    path_to_weights: str
    checkpoint_kind: Literal["latest"] = "latest"


class AllStateArpeSwitcher(Switcher):
    """Load a feed-forward all-state Switcher and reproduce its ARPE pin."""

    expected_encoder_custom = "switcher_all_state"
    allow_aoreplan_wait = True
    policy_label = "all-state ARPE Switcher"

    def __init__(self, cfg: AllStateArpeSwitcherConfig):
        self.cfg = cfg
        path = Path(cfg.path_to_weights)
        self.config_path = (path / "config.json").resolve()
        register_custom_components()
        payload = self.config_path.read_bytes()
        self.config_sha256 = hashlib.sha256(payload).hexdigest()
        config = json.loads(payload.decode("utf-8"))
        full_config = deepcopy(config["full_config"])
        declaration = full_config.pop("candidate_policy", None)
        if not isinstance(declaration, dict):
            raise RuntimeError("Switcher checkpoint has no frozen candidate_policy pin.")
        project_root = Path(__file__).resolve().parents[1]
        artifact = ArpeCandidateArtifact.from_mapping(declaration, project_root)
        artifact.verify_files()
        from learning.config import checkpoint_experiment_config
        _, flat_config = validate_config(checkpoint_experiment_config(full_config))
        if flat_config.encoder_custom != self.expected_encoder_custom:
            raise RuntimeError(
                "Checkpoint is not the all-state Switcher network: expected "
                f"encoder_custom={self.expected_encoder_custom!r}."
            )
        if bool(flat_config.use_rnn):
            raise RuntimeError("All-state Switcher checkpoint must be feed-forward.")

        observation_space = switcher_observation_space()
        action_space = gym.spaces.Discrete(NUM_BRANCHES)
        actor = create_actor_critic(flat_config, observation_space, action_space)
        self.device = self._resolve_device(cfg.device)
        actor.model_to_device(self.device)
        checkpoint_dir = join(str(path), f"checkpoint_p{flat_config.policy_index}")
        checkpoint_path = self._resolve_checkpoint(checkpoint_dir, cfg.checkpoint_kind)
        checkpoint_payload = checkpoint_path.read_bytes()
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = hashlib.sha256(checkpoint_payload).hexdigest()
        checkpoint = torch.load(
            io.BytesIO(checkpoint_payload),
            map_location=self.device,
            weights_only=False,
        )
        actor.load_state_dict(checkpoint["model"])
        actor.eval()
        for parameter in actor.parameters():
            parameter.requires_grad_(False)

        self.ppo = actor
        self.flat_config = flat_config
        self.candidate_artifact = artifact
        self.candidate_policy = deepcopy(declaration)
        self.loader_schema = ARPE_SWITCHER_LOADER_SCHEMA
        self.after_reset()

    def get_stats(self) -> dict:
        result = super().get_stats()
        result.update(
            {
                "switcher_loader_schema": self.loader_schema,
                "switcher_candidate_policy": deepcopy(self.candidate_policy),
                "switcher_candidate_artifact": deepcopy(
                    self.candidate_artifact.as_dict()
                ),
            }
        )
        return result


__all__ = [
    "AllStateArpeSwitcher",
    "AllStateArpeSwitcherConfig",
    "ARPE_SWITCHER_LOADER_SCHEMA",
]
