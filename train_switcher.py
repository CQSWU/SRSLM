"""Train the final or NoWait two-branch Switcher against pinned ARPE."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from functools import partial
from pathlib import Path

import yaml
from sample_factory.algo.utils.context import global_env_registry
from sample_factory.train import run_rl

import train as base_train
from agents.arpe import ArpeCandidateArtifact
from learning.config import Environment
from pomapf_env.switcher_arpe_env import (
    ARPE_NOWAIT_ENV_SCHEMA,
    ARPE_SWITCHER_ENV_SCHEMA,
    ArpeNoWaitSwitcherEnv,
    ArpeSwitcherEnv,
)


PROJECT_ROOT = Path(__file__).resolve().parent

MODES = {
    "final": {
        "environment": "POMAPF-SRSLM-v0",
        "encoder": "switcher",
        "schema": ARPE_SWITCHER_ENV_SCHEMA,
        "entrypoint_schema": "srslm_switcher_wait_caar_training_entrypoint_v1",
        "environment_class": ArpeSwitcherEnv,
        "decision_scope": "aoreplan_nonwait_only",
    },
    "nowait": {
        "environment": "POMAPF-SRSLM-NoWait-v0",
        "encoder": "switcher_all_state",
        "schema": ARPE_NOWAIT_ENV_SCHEMA,
        "entrypoint_schema": "srslm_switcher_nowait_training_entrypoint_v1",
        "environment_class": ArpeNoWaitSwitcherEnv,
        "decision_scope": "all_states",
    },
}


def _spec(mode: str) -> dict:
    try:
        return MODES[mode]
    except KeyError as exc:
        raise ValueError(f"Unknown Switcher training mode: {mode!r}") from exc


def _environment_for_worker(cfg, env_config) -> Environment:
    environment = Environment(**cfg.full_config["environment"])
    populations = environment.training_num_agents_by_worker
    if populations is None:
        return environment
    worker_index = (
        env_config.get("worker_index", 0)
        if isinstance(env_config, dict)
        else getattr(env_config, "worker_index", 0)
    )
    worker_index = int(worker_index or 0)
    if worker_index < 0:
        raise ValueError("Sample Factory worker index must be non-negative.")
    worker_grid = deepcopy(environment.grid_config)
    worker_grid.num_agents = populations[worker_index % len(populations)]
    return environment.copy(update={"grid_config": worker_grid})


def create_switcher_env(
    full_env_name,
    cfg=None,
    env_config=None,
    render_mode=None,
    *,
    mode="final",
):
    del render_mode
    spec = _spec(mode)
    if full_env_name != spec["environment"]:
        raise ValueError(
            f"{mode} Switcher entrypoint cannot construct {full_env_name!r}."
        )
    environment = _environment_for_worker(cfg, env_config)
    declaration = cfg.full_config.get("candidate_policy")
    if not isinstance(declaration, dict):
        raise RuntimeError("Saved Switcher config has no candidate_policy pin.")
    artifact = ArpeCandidateArtifact.from_mapping(declaration, PROJECT_ROOT)
    environment_class = spec["environment_class"]
    return environment_class(
        grid_config=environment.grid_config,
        candidate_artifact=artifact,
        candidate_device=environment.switcher_caar_device,
        max_planning_steps=environment.switcher_max_planning_steps,
        team_reward_coefficient=environment.switcher_team_reward_coefficient,
        feature_schema=environment.switcher_feature_schema,
    )


def register_switcher_components(mode="final") -> None:
    spec = _spec(mode)
    base_train.register_custom_components()
    global_env_registry()[spec["environment"]] = partial(
        create_switcher_env, mode=mode
    )


def prepare_switcher_config(config: dict, mode="final") -> tuple[object, object]:
    spec = _spec(mode)
    payload = deepcopy(config)
    declaration = payload.pop("candidate_policy", None)
    if not isinstance(declaration, dict):
        raise ValueError("Switcher config requires candidate_policy.")
    artifact = ArpeCandidateArtifact.from_mapping(declaration, PROJECT_ROOT)
    artifact.verify_files()

    experiment, flat_config = base_train.validate_config(payload)
    if flat_config.encoder_custom != spec["encoder"]:
        raise ValueError(
            f"{mode} Switcher requires encoder_custom={spec['encoder']!r}."
        )
    if flat_config.env != spec["environment"]:
        raise ValueError(
            f"{mode} Switcher requires environment {spec['environment']!r}."
        )
    if bool(flat_config.use_rnn):
        raise ValueError("Switcher must remain feed-forward.")
    configured = Path(experiment.environment.switcher_caar_weights_path).as_posix()
    if configured != artifact.weights_relative:
        raise ValueError("environment ARPE path differs from candidate_policy.")
    if experiment.environment.switcher_caar_device != "cuda":
        raise ValueError("Frozen ARPE candidate must use the PPU device.")

    flat_config.full_config = deepcopy(flat_config.full_config)
    flat_config.full_config["candidate_policy"] = deepcopy(declaration)
    flat_config.candidate_policy = deepcopy(declaration)
    flat_config.switcher_integration_schema = spec["schema"]
    flat_config.switcher_training_entrypoint_schema = spec["entrypoint_schema"]
    return experiment, flat_config


def _apply_overrides(config: dict, args) -> set[str]:
    explicit: set[str] = set()
    if args.run_name is not None:
        config["name"] = args.run_name
        config.setdefault("global_settings", {})["experiment"] = args.run_name
        explicit.add("experiment")
    if args.train_dir is not None:
        config.setdefault("global_settings", {})["train_dir"] = args.train_dir
        explicit.add("train_dir")
    if args.train_for_env_steps is not None:
        config.setdefault("experiment_settings", {})["train_for_env_steps"] = int(
            args.train_for_env_steps
        )
        explicit.add("train_for_env_steps")
    return explicit


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--mode", choices=tuple(MODES), default="final")
    parser.add_argument("--run_name")
    parser.add_argument("--train_dir")
    parser.add_argument("--train_for_env_steps", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    register_switcher_components(args.mode)
    config_path = Path(args.config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    explicit = _apply_overrides(config, args)
    _, flat_config = prepare_switcher_config(config, args.mode)
    base_train._sync_resume_cli_overrides(flat_config, explicit)
    if args.validate_only:
        spec = _spec(args.mode)
        print(
            json.dumps(
                {
                    "validated": True,
                    "mode": args.mode,
                    "schema": spec["entrypoint_schema"],
                    "integration_schema": spec["schema"],
                    "experiment": flat_config.experiment,
                    "target_frames": int(flat_config.train_for_env_steps),
                    "workers": int(flat_config.num_workers),
                    "candidate_policy": flat_config.candidate_policy,
                    "decision_scope": spec["decision_scope"],
                },
                sort_keys=True,
            )
        )
        return 0
    return int(run_rl(flat_config))


if __name__ == "__main__":
    raise SystemExit(main())
