import json
import hashlib
import re
from copy import deepcopy
from pathlib import Path
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
from pydantic import Extra
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.cfg.arguments import default_cfg
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size

from agents.utils_agents import AlgoBase
from learning.grid_memory import MultipleGridMemory
from pomapf_env.wrappers import MatrixObservationWrapper
from train import register_custom_components


OFFICIAL_EPOM_RELEASE = "v0"
OFFICIAL_EPOM_CONFIG_SHA256 = (
    "ea9c470bad09e78c8b66dc579296a4da745e117affb84c7ce732d1dbd74e0c40"
)
OFFICIAL_EPOM_CHECKPOINT = "checkpoint_000311682_1000002674.pth"
OFFICIAL_EPOM_CHECKPOINT_SIZE = 116_465_689
OFFICIAL_EPOM_CHECKPOINT_SHA256 = (
    "549feac19e21593af072677305945d7c22bd7f66cb07927a92eb59f2f8a3cce9"
)
EPOM_ARTIFACT_PROFILES = ("official_v0", "lifelong_finetuned")
_CHECKPOINT_PATTERN = re.compile(r"^checkpoint_(\d+)_(\d+)\.pth$")


class EPOMConfig(AlgoBase, extra=Extra.forbid):
    name: Literal["EPOM"] = "EPOM"
    path_to_weights: str = "weights/EPOM/EPOM"
    artifact_profile: Literal["official_v0", "lifelong_finetuned"] = (
        "official_v0"
    )


def _flat_config(full_config):
    global_settings = full_config["global_settings"]
    cfg = default_cfg(
        algo=global_settings["algo"],
        env=global_settings["env"],
        experiment=global_settings.get("experiment", ""),
    )
    for section_name in (
        "async_ppo",
        "experiment_settings",
        "global_settings",
        "evaluation",
    ):
        for key, value in full_config[section_name].items():
            setattr(cfg, key, value)
    settings = full_config["experiment_settings"]
    cfg.num_batches_per_epoch = full_config["async_ppo"][
        "num_batches_per_iteration"
    ]
    cfg.rnn_size = settings["hidden_size"]
    cfg.encoder_conv_architecture = settings["encoder_subtype"]
    cfg.encoder_conv_mlp_layers = [settings["hidden_size"]]
    cfg.full_config = full_config
    return cfg


def _architecture_mismatches(
    full_config,
    *,
    expected_encoder_custom="pogema_residual",
):
    settings = full_config["experiment_settings"]
    async_ppo = full_config["async_ppo"]
    environment = full_config["environment"]
    grid_config = environment["grid_config"]
    expected = {
        "custom encoder": (
            settings.get("encoder_custom"),
            expected_encoder_custom,
        ),
        "encoder filters": (settings.get("pogema_encoder_num_filters"), 64),
        "encoder residual blocks": (
            settings.get("pogema_encoder_num_res_blocks"),
            3,
        ),
        "encoder FC layers": (settings.get("encoder_extra_fc_layers"), 1),
        "hidden size": (settings.get("hidden_size"), 512),
        "recurrent policy": (async_ppo.get("use_rnn"), True),
        "RNN type": (async_ppo.get("rnn_type"), "gru"),
        "RNN layers": (async_ppo.get("rnn_num_layers"), 1),
        "observation radius": (grid_config.get("obs_radius"), 5),
        "grid-memory radius": (environment.get("grid_memory_obs_radius"), 7),
    }
    return {
        name: {"actual": actual, "expected": wanted}
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    }


def _validate_official_config(full_config):
    """Reject a similarly named checkpoint with a different EPOM network."""
    mismatches = _architecture_mismatches(full_config)
    if full_config.get("name") != "pomapf-grid-memory-multiagent-full-v2":
        mismatches["experiment name"] = {
            "actual": full_config.get("name"),
            "expected": "pomapf-grid-memory-multiagent-full-v2",
        }
    if mismatches:
        raise RuntimeError(
            "EPOM config does not describe the official v0 network: "
            f"{mismatches}"
        )


def _validate_lifelong_finetuned_config(full_config):
    """Require the official EPOM structure and the paper's lifelong protocol."""
    mismatches = _architecture_mismatches(
        full_config,
        expected_encoder_custom="epom_finetune",
    )
    environment = full_config["environment"]
    grid_config = environment["grid_config"]
    expected_protocol = {
        "on-target rule": (grid_config.get("on_target"), "restart"),
        "collision system": (
            grid_config.get("collision_system"),
            "block_both",
        ),
        "episode horizon": (grid_config.get("max_episode_steps"), 512),
        "action order": (
            grid_config.get("MOVES"),
            [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]],
        ),
    }
    mismatches.update(
        {
            name: {"actual": actual, "expected": wanted}
            for name, (actual, wanted) in expected_protocol.items()
            if actual != wanted
        }
    )
    if mismatches:
        raise RuntimeError(
            "EPOM lifelong_finetuned config is incompatible with the audited "
            f"network/protocol: {mismatches}"
        )


def _inference_full_config(full_config, artifact_profile):
    """Remove training-only factories without mutating the hashed source config."""
    inference_config = deepcopy(full_config)
    if artifact_profile == "lifelong_finetuned":
        inference_config["experiment_settings"][
            "encoder_custom"
        ] = "pogema_residual"
    return inference_config


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _select_lifelong_checkpoint(weights_dir):
    checkpoint_dir = Path(weights_dir) / "checkpoint_p0"
    candidates = []
    if checkpoint_dir.is_dir():
        for path in checkpoint_dir.iterdir():
            match = _CHECKPOINT_PATTERN.fullmatch(path.name)
            if path.is_file() and match:
                updates, frames = (int(value) for value in match.groups())
                candidates.append((frames, updates, path.name, path))
    if not candidates:
        raise FileNotFoundError(
            "Missing EPOM lifelong_finetuned checkpoint matching "
            f"checkpoint_<updates>_<frames>.pth in {checkpoint_dir}"
        )
    return max(candidates)[-1], len(candidates)


def _select_config_path(weights_dir, artifact_profile):
    """Select the immutable config artifact written by each training stack."""
    weights_dir = Path(weights_dir)
    if artifact_profile == "official_v0":
        return weights_dir / "cfg.json"

    # Sample Factory 2 writes config.json for new experiments.  Keep cfg.json
    # as a compatibility fallback for explicitly exported fine-tuned bundles,
    # but prefer the native training artifact when both names are present.
    for filename in ("config.json", "cfg.json"):
        path = weights_dir / filename
        if path.is_file():
            return path
    return weights_dir / "config.json"


def _training_protocol(full_config):
    environment = full_config["environment"]
    grid_config = environment["grid_config"]
    return {
        "on_target": grid_config.get("on_target"),
        "collision_system": grid_config.get("collision_system"),
        "obs_radius": grid_config.get("obs_radius"),
        "max_episode_steps": grid_config.get("max_episode_steps"),
        "num_agents": grid_config.get("num_agents"),
        "grid_memory_obs_radius": environment.get("grid_memory_obs_radius"),
    }


class EPOM:
    """EPOM inference adapter with explicit, hash-audited artifact profiles."""

    def __init__(self, cfg: EPOMConfig):
        self.algo_cfg = cfg
        weights_dir = Path(cfg.path_to_weights)
        config_path = _select_config_path(weights_dir, cfg.artifact_profile)
        if not config_path.exists():
            raise FileNotFoundError(f"Missing official EPOM config: {config_path}")
        config_bytes = config_path.read_bytes()
        config_sha256 = hashlib.sha256(config_bytes).hexdigest()
        try:
            full_config = json.loads(config_bytes)["full_config"]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Invalid EPOM cfg.json: {config_path}") from exc
        if cfg.artifact_profile == "official_v0":
            if config_sha256 != OFFICIAL_EPOM_CONFIG_SHA256:
                raise RuntimeError(
                    "EPOM cfg.json is not the official v0 artifact: "
                    f"sha256={config_sha256}"
                )
            _validate_official_config(full_config)
        elif cfg.artifact_profile == "lifelong_finetuned":
            _validate_lifelong_finetuned_config(full_config)
        else:  # Pydantic normally prevents this, but keep the loader defensive.
            raise RuntimeError(
                f"Unsupported EPOM artifact profile: {cfg.artifact_profile!r}"
            )
        inference_full_config = _inference_full_config(
            full_config,
            cfg.artifact_profile,
        )
        self.cfg = _flat_config(inference_full_config)
        environment = full_config["environment"]
        self.grid_memory_radius = (
            environment.get("grid_memory_obs_radius")
            or environment["grid_config"]["obs_radius"]
        )

        register_custom_components()
        size = self.grid_memory_radius * 2 + 1
        obs_space = gym.spaces.Dict(
            {
                "obs": gym.spaces.Box(
                    0.0,
                    1.0,
                    shape=(3, size, size),
                    dtype=np.float32,
                ),
                "xy": gym.spaces.Box(
                    -1024,
                    1024,
                    shape=(2,),
                    dtype=np.float32,
                ),
                "target_xy": gym.spaces.Box(
                    -1024,
                    1024,
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )
        actor_critic = create_actor_critic(
            self.cfg,
            obs_space,
            gym.spaces.Discrete(5),
        )

        self.device = self._resolve_device(cfg.device)
        if self.device.type == "mps":
            actor_critic.float()
        actor_critic.model_to_device(self.device)
        if self.cfg.policy_index != 0:
            raise RuntimeError(
                f"EPOM requires policy_index=0, got {self.cfg.policy_index}"
            )
        if cfg.artifact_profile == "official_v0":
            checkpoint_path = (
                weights_dir / "checkpoint_p0" / OFFICIAL_EPOM_CHECKPOINT
            )
            checkpoint_candidates = 1
        else:
            checkpoint_path, checkpoint_candidates = _select_lifelong_checkpoint(
                weights_dir
            )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing EPOM checkpoint: {checkpoint_path}"
            )
        checkpoint_size = checkpoint_path.stat().st_size
        checkpoint_sha256 = _sha256(checkpoint_path)
        if (
            cfg.artifact_profile == "official_v0"
            and checkpoint_size != OFFICIAL_EPOM_CHECKPOINT_SIZE
        ):
            raise RuntimeError(
                "EPOM checkpoint size does not match the official v0 artifact: "
                f"{checkpoint_size} != {OFFICIAL_EPOM_CHECKPOINT_SIZE}"
            )
        if (
            cfg.artifact_profile == "official_v0"
            and checkpoint_sha256 != OFFICIAL_EPOM_CHECKPOINT_SHA256
        ):
            raise RuntimeError(
                "EPOM checkpoint is not the official v0 artifact: "
                f"sha256={checkpoint_sha256}"
            )
        checkpoint = torch.load(
            str(checkpoint_path),
            map_location=self.device,
            weights_only=False,
        )
        self._use_obs_normalization = any(
            key.startswith("obs_normalizer.")
            for key in checkpoint["model"]
        )
        self._load_model(actor_critic, checkpoint["model"])

        self.ppo = actor_critic
        self.rnn_states = None
        self.grid_memory = MultipleGridMemory()
        self.path = weights_dir
        self._artifact_provenance = {
            "method": "EPOM",
            "artifact_profile": cfg.artifact_profile,
            "release": (
                OFFICIAL_EPOM_RELEASE
                if cfg.artifact_profile == "official_v0"
                else None
            ),
            "weights_path": str(weights_dir.resolve()),
            "config_path": str(config_path.resolve()),
            "config_sha256": config_sha256,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_name": checkpoint_path.name,
            "checkpoint_size": checkpoint_size,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_candidates": checkpoint_candidates,
            "selection_rule": (
                "fixed official EPOM v0 filename"
                if cfg.artifact_profile == "official_v0"
                else "largest (environment frames, learner updates) checkpoint filename"
            ),
            "training_protocol": _training_protocol(full_config),
            "source_encoder_custom": full_config["experiment_settings"].get(
                "encoder_custom"
            ),
            "inference_encoder_custom": inference_full_config[
                "experiment_settings"
            ].get("encoder_custom"),
        }

    @staticmethod
    def _resolve_device(requested):
        if requested == "cpu":
            return torch.device("cpu")
        if requested.startswith("cuda") and torch.cuda.is_available():
            return torch.device(requested)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _load_model(actor_critic, checkpoint_state):
        current_state = actor_critic.state_dict()
        migrated = {
            key.replace("encoder.fc_after_enc.", "encoder.fc_blocks.", 1): value
            for key, value in checkpoint_state.items()
        }
        current_keys = set(current_state)
        migrated_keys = set(migrated)
        unexpected = migrated_keys - current_keys
        missing = current_keys - migrated_keys
        optional_missing = set()
        for prefix in ("obs_normalizer.", "returns_normalizer."):
            expected_group = {
                key for key in current_keys if key.startswith(prefix)
            }
            provided_group = expected_group & migrated_keys
            if expected_group and not provided_group:
                optional_missing.update(expected_group)
        required_missing = missing - optional_missing
        shape_mismatches = {
            key: (tuple(current_state[key].shape), tuple(value.shape))
            for key, value in migrated.items()
            if key in current_state and current_state[key].shape != value.shape
        }
        if required_missing or unexpected or shape_mismatches:
            raise RuntimeError(
                "EPOM checkpoint does not match the original network. "
                f"missing={sorted(required_missing)}, "
                f"unexpected={sorted(unexpected)}, "
                f"shape mismatches={shape_mismatches}"
            )
        current_state.update(migrated)
        actor_critic.load_state_dict(current_state, strict=True)

    def after_reset(self):
        torch.manual_seed(self.algo_cfg.seed)
        self.rnn_states = None
        self.grid_memory.clear()

    def act(self, observations, rewards=None, dones=None, infos=None):
        observations = deepcopy(observations)
        num_agents = len(observations)
        if self.rnn_states is None or len(self.rnn_states) != num_agents:
            self.rnn_states = torch.zeros(
                (num_agents, get_rnn_size(self.cfg)),
                dtype=torch.float32,
                device=self.device,
            )

        self.grid_memory.update(observations)
        self.grid_memory.modify_observation(
            observations,
            self.grid_memory_radius,
        )
        observations = MatrixObservationWrapper.to_matrix(observations)
        obs_torch = TensorDict(
            {
                key: torch.from_numpy(
                    np.stack([obs[key] for obs in observations])
                ).to(self.device).float()
                for key in observations[0]
            }
        )
        model_input = (
            prepare_and_normalize_obs(self.ppo, obs_torch)
            if self._use_obs_normalization
            else obs_torch
        )
        with torch.no_grad():
            outputs = self.ppo(model_input, self.rnn_states)
        self.rnn_states = outputs["new_rnn_states"]
        return outputs["actions"].cpu().numpy()

    def get_additional_info(self):
        return {"rl_used": 1.0}

    def get_model_provenance(self):
        return deepcopy(self._artifact_provenance)

    def get_name(self):
        return self.path.name

    def clear_hidden(self, agent_idx):
        if self.rnn_states is not None:
            self.rnn_states[agent_idx].zero_()

    def after_step(self, dones):
        for agent_idx, done in enumerate(dones):
            if done:
                self.clear_hidden(agent_idx)
        if all(dones):
            self.rnn_states = None
            self.grid_memory.clear()
