#!/usr/bin/env python3
"""Fail-closed preflight for EPOM v0 lifelong warm-start training."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_wrapper(env, wrapper_type):
    current = env
    while current is not None:
        if isinstance(current, wrapper_type):
            return current
        child = getattr(current, "env", None)
        if child is current:
            break
        current = child
    return None


def migrated_official_state(checkpoint_state):
    return {
        key.replace("encoder.fc_after_enc.", "encoder.fc_blocks.", 1): value
        for key, value in checkpoint_state.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.project_root.expanduser().resolve()
    config_path = args.config.expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()
    sys.path.insert(0, str(root))

    from agents.epom import (
        EPOM,
        OFFICIAL_EPOM_CHECKPOINT,
        OFFICIAL_EPOM_CHECKPOINT_SHA256,
        OFFICIAL_EPOM_CHECKPOINT_SIZE,
        OFFICIAL_EPOM_CONFIG_SHA256,
    )
    from learning.epom_finetune_actor_critic import EPOMFineTuneActorCritic
    from pomapf_env.wrappers import GridMemoryObservationWrapper
    from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
    from sample_factory.algo.utils.tensor_dict import TensorDict
    from sample_factory.envs.create_env import create_env
    from sample_factory.model.actor_critic import (
        create_actor_critic,
        default_make_actor_critic_func,
    )
    from sample_factory.model.model_utils import get_rnn_size
    from train import register_custom_components, validate_config

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment, cfg = validate_config(config)
    if experiment.experiment_settings.encoder_custom != "epom_finetune":
        raise RuntimeError("Config is not an EPOM fine-tuning run.")
    if cfg.normalize_input:
        raise RuntimeError("Official EPOM warm-start requires normalize_input=false.")
    if cfg.train_for_env_steps not in (1_048_576, 100_000_000, 250_000_000):
        raise RuntimeError(
            "Expected the smoke, 100M formal, or approved 250M extension target."
        )
    expected_workers = 2 if cfg.train_for_env_steps == 1_048_576 else 12
    if cfg.num_workers != expected_workers:
        raise RuntimeError(
            f"Expected {expected_workers} rollout workers, got {cfg.num_workers}."
        )

    weights_dir = root / experiment.experiment_settings.epom_base_weights_path
    official_cfg_path = weights_dir / "cfg.json"
    checkpoint_path = (
        weights_dir / "checkpoint_p0" / OFFICIAL_EPOM_CHECKPOINT
    )
    if sha256(official_cfg_path) != OFFICIAL_EPOM_CONFIG_SHA256:
        raise RuntimeError("Official EPOM cfg.json hash mismatch.")
    if checkpoint_path.stat().st_size != OFFICIAL_EPOM_CHECKPOINT_SIZE:
        raise RuntimeError("Official EPOM checkpoint size mismatch.")
    if sha256(checkpoint_path) != OFFICIAL_EPOM_CHECKPOINT_SHA256:
        raise RuntimeError("Official EPOM checkpoint hash mismatch.")
    train_map_path = root / experiment.environment.grid_config.map_name
    train_map_hash = sha256(train_map_path)
    if train_map_hash != (
        "3436e1efb55f2a5fe0e2428f7dde2e585cdf8502b1382a8708d9f0259cd90ad0"
    ):
        raise RuntimeError("maps/train.yaml differs from the audited training split.")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if set(checkpoint) != {"train_step", "env_steps", "model", "optimizer"}:
        raise RuntimeError(f"Unexpected official checkpoint keys: {sorted(checkpoint)}")
    if len(checkpoint["model"]) != 28:
        raise RuntimeError("Official EPOM checkpoint must contain 28 model tensors.")

    register_custom_components()
    env = create_env(cfg.env, cfg=cfg, env_config={})
    try:
        memory_wrapper = find_wrapper(env, GridMemoryObservationWrapper)
        if memory_wrapper is None or memory_wrapper.memory_radius != 7:
            raise RuntimeError("EPOM grid-memory wrapper with radius 7 is missing.")
        if "tau" in env.observation_space.spaces:
            raise RuntimeError("EPOM fine-tune environment unexpectedly exposes tau.")
        if tuple(env.observation_space["obs"].shape) != (3, 15, 15):
            raise RuntimeError(
                f"Unexpected EPOM spatial input {env.observation_space['obs'].shape}."
            )

        model = create_actor_critic(cfg, env.observation_space, env.action_space)
        if not isinstance(model, EPOMFineTuneActorCritic):
            raise RuntimeError(f"Unexpected actor-critic type: {type(model).__name__}")
        if hasattr(model, "env_steps") or hasattr(model, "train_step"):
            raise RuntimeError("Learner progress leaked into the warm-start model.")

        model_state = model.state_dict()
        migrated = migrated_official_state(checkpoint["model"])
        for key, expected in migrated.items():
            if key not in model_state or not torch.equal(model_state[key], expected):
                raise RuntimeError(f"Official tensor was not restored exactly: {key}")
        frozen = [
            name for name, parameter in model.named_parameters()
            if not parameter.requires_grad
        ]
        if frozen:
            raise RuntimeError(f"Fine-tune parameters are frozen: {frozen}")

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.learning_rate,
            eps=cfg.adam_eps,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
        )
        if optimizer.state:
            raise RuntimeError("Fresh optimizer unexpectedly contains legacy state.")

        reference_cfg = copy.deepcopy(cfg)
        reference_cfg.encoder_custom = "pogema_residual"
        reference_cfg.full_config["experiment_settings"][
            "encoder_custom"
        ] = "pogema_residual"
        reference = default_make_actor_critic_func(
            reference_cfg,
            env.observation_space,
            env.action_space,
        )
        EPOM._load_model(reference, checkpoint["model"])

        observations, _ = env.reset()
        batch = TensorDict(
            {
                key: torch.from_numpy(
                    np.stack([observation[key] for observation in observations])
                ).float()
                for key in observations[0]
            }
        )
        model_batch = prepare_and_normalize_obs(
            model,
            TensorDict({key: value.clone() for key, value in batch.items()}),
        )
        reference_batch = prepare_and_normalize_obs(
            reference,
            TensorDict({key: value.clone() for key, value in batch.items()}),
        )
        rnn_states = torch.zeros(
            (len(observations), get_rnn_size(cfg)),
            dtype=torch.float32,
        )

        model.eval()
        reference.eval()
        torch.manual_seed(1729)
        model_outputs = model(model_batch, rnn_states.clone())
        torch.manual_seed(1729)
        reference_outputs = reference(reference_batch, rnn_states.clone())
        for key in ("action_logits", "values", "actions"):
            torch.testing.assert_close(
                model_outputs[key],
                reference_outputs[key],
                rtol=0,
                atol=0,
            )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_outputs = model(model_batch, rnn_states.clone())
        loss = (
            train_outputs["action_logits"].square().mean()
            + train_outputs["values"].square().mean()
        )
        loss.backward()
        gradient_groups = {
            "encoder": "encoder.",
            "gru": "core.",
            "actor": "action_parameterization.",
            "critic": "critic_linear.",
        }
        gradient_norms = {}
        for group, prefix in gradient_groups.items():
            squared = [
                parameter.grad.detach().float().square().sum()
                for name, parameter in model.named_parameters()
                if name.startswith(prefix) and parameter.grad is not None
            ]
            norm = float(torch.sqrt(torch.stack(squared).sum()).item()) if squared else 0.0
            if not np.isfinite(norm) or norm <= 0:
                raise RuntimeError(f"No finite nonzero {group} gradient.")
            gradient_norms[group] = norm

        probe_name = "action_parameterization.distribution_linear.weight"
        probe_parameter = dict(model.named_parameters())[probe_name]
        before_step = probe_parameter.detach().clone()
        optimizer.step()
        if torch.equal(before_step, probe_parameter.detach()):
            raise RuntimeError("Fresh optimizer step did not update the EPOM actor.")

        result = {
            "validated": True,
            "config": str(config_path),
            "experiment": cfg.experiment,
            "target_domain_env_steps": int(cfg.train_for_env_steps),
            "source_checkpoint_train_step": int(checkpoint["train_step"]),
            "source_checkpoint_env_steps": int(checkpoint["env_steps"]),
            "learner_starts_from_env_steps": 0,
            "model_tensor_count": len(checkpoint["model"]),
            "all_model_parameters_trainable": True,
            "fresh_optimizer_state_entries": 0,
            "step0_exact_keys": ["action_logits", "values", "actions"],
            "gradient_norms": gradient_norms,
            "official_config_sha256": OFFICIAL_EPOM_CONFIG_SHA256,
            "official_checkpoint_sha256": OFFICIAL_EPOM_CHECKPOINT_SHA256,
            "train_map_sha256": train_map_hash,
            "environment": {
                "on_target": "restart",
                "collision_system": "block_both",
                "num_agents": 200,
                "max_episode_steps": 512,
                "live_observation_radius": 5,
                "grid_memory_radius": 7,
                "tau_present": False,
            },
        }
    finally:
        env.close()

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        output_path = args.output.expanduser()
        if not output_path.is_absolute():
            output_path = root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
