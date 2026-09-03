"""Official EPOM v0 warm-start for target-domain PPO fine-tuning.

This intentionally restores only the published actor-critic tensors.  The
Sample Factory optimizer and progress counters are created by the new run, so
the legacy SF1 optimizer state and its one-billion-step counter never leak into
lifelong fine-tuning.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from sample_factory.model.actor_critic import ActorCriticSharedWeights

from agents.epom import (
    EPOM,
    OFFICIAL_EPOM_CHECKPOINT,
    OFFICIAL_EPOM_CHECKPOINT_SHA256,
    OFFICIAL_EPOM_CHECKPOINT_SIZE,
    OFFICIAL_EPOM_CONFIG_SHA256,
)


class EPOMFineTuneActorCritic(ActorCriticSharedWeights):
    """Train every official EPOM parameter from the released v0 weights."""

    NUM_ACTIONS = 5

    def __init__(self, model_factory, obs_space, action_space, cfg):
        if not cfg.actor_critic_share_weights:
            raise ValueError("EPOM fine-tuning requires shared actor-critic weights.")
        if "tau" in obs_space.spaces:
            raise ValueError("EPOM fine-tuning must not receive the CAAR tau input.")
        if getattr(action_space, "n", None) != self.NUM_ACTIONS:
            raise ValueError(
                f"EPOM fine-tuning expects five discrete actions, got {action_space}."
            )

        super().__init__(model_factory, obs_space, action_space, cfg)
        self._warm_start_from_official_epom(cfg)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _resolve_weights_dir(configured_path: str) -> Path:
        path = Path(configured_path).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        return path.resolve()

    def _warm_start_from_official_epom(self, cfg) -> None:
        settings = cfg.full_config["experiment_settings"]
        weights_dir = self._resolve_weights_dir(settings["epom_base_weights_path"])
        config_path = weights_dir / "cfg.json"
        checkpoint_path = (
            weights_dir / "checkpoint_p0" / OFFICIAL_EPOM_CHECKPOINT
        )
        if not config_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Official EPOM artifacts are incomplete under {weights_dir}."
            )
        if self._sha256(config_path) != OFFICIAL_EPOM_CONFIG_SHA256:
            raise RuntimeError("EPOM fine-tune base cfg.json is not official EPOM v0.")
        if checkpoint_path.stat().st_size != OFFICIAL_EPOM_CHECKPOINT_SIZE:
            raise RuntimeError(
                "EPOM fine-tune base checkpoint has an unexpected size."
            )
        if self._sha256(checkpoint_path) != OFFICIAL_EPOM_CHECKPOINT_SHA256:
            raise RuntimeError(
                "EPOM fine-tune base checkpoint is not official EPOM v0."
            )

        official_config = json.loads(config_path.read_text(encoding="utf-8"))[
            "full_config"
        ]
        official_settings = official_config["experiment_settings"]
        expected = {
            "filters": (
                settings["pogema_encoder_num_filters"],
                official_settings["pogema_encoder_num_filters"],
            ),
            "residual blocks": (
                settings["pogema_encoder_num_res_blocks"],
                official_settings["pogema_encoder_num_res_blocks"],
            ),
            "hidden size": (
                settings["hidden_size"],
                official_settings["hidden_size"],
            ),
            "encoder FC layers": (
                settings["encoder_extra_fc_layers"],
                official_settings["encoder_extra_fc_layers"],
            ),
        }
        mismatches = {
            name: {"fine_tune": actual, "official": wanted}
            for name, (actual, wanted) in expected.items()
            if actual != wanted
        }
        if mismatches:
            raise RuntimeError(
                f"EPOM fine-tune network differs from EPOM v0: {mismatches}"
            )

        checkpoint = torch.load(
            str(checkpoint_path),
            map_location="cpu",
            weights_only=False,
        )
        if len(checkpoint["model"]) != 28:
            raise RuntimeError(
                "Official EPOM v0 is expected to contain exactly 28 model tensors."
            )
        EPOM._load_model(self, checkpoint["model"])

        # Provenance only.  These values are deliberately not assigned to the
        # Sample Factory learner, which starts with a fresh optimizer and zero
        # target-domain progress.
        self.official_epom_weights_dir = weights_dir
        self.official_epom_checkpoint_sha256 = OFFICIAL_EPOM_CHECKPOINT_SHA256
        self.official_epom_source_train_step = int(checkpoint["train_step"])
        self.official_epom_source_env_steps = int(checkpoint["env_steps"])
        self.official_epom_model_tensor_count = len(checkpoint["model"])

        frozen = [
            name
            for name, parameter in self.named_parameters()
            if not parameter.requires_grad
        ]
        if frozen:
            raise RuntimeError(
                "All EPOM actor-critic parameters must remain trainable, got frozen "
                f"parameters: {frozen}"
            )
