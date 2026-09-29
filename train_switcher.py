"""Train the current wait-rule Switcher with a frozen ARPE branch."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import yaml
from sample_factory.algo.utils.context import global_env_registry
from sample_factory.train import run_rl

import train as base_train
from agents.arpe import ArpeCandidateArtifact
from learning.config import Environment
from pomapf_env.switcher_arpe_env import (
    ARPE_SWITCHER_ENV_SCHEMA,
    ArpeSwitcherEnv,
)


PROJECT_ROOT = Path(__file__).resolve().parent

ENV_NAME = "POMAPF-SRSLM-v0"
ENTRYPOINT_SCHEMA = "srslm_switcher_wait_caar_training_entrypoint_v1"


def create_switcher_env(
    full_env_name,
    cfg=None,
    env_config=None,
    render_mode=None,
):
    del render_mode
    if full_env_name != ENV_NAME:
        raise ValueError(
            f"Switcher entrypoint cannot construct {full_env_name!r}."
        )
    environment = Environment(**cfg.full_config["environment"]).for_worker(env_config)
    declaration = cfg.full_config.get("candidate_policy")
    if not isinstance(declaration, dict):
        raise RuntimeError("Saved Switcher config has no candidate_policy paths.")
    artifact = ArpeCandidateArtifact.from_mapping(declaration, PROJECT_ROOT)
    return ArpeSwitcherEnv(
        grid_config=environment.grid_config,
        candidate_artifact=artifact,
        candidate_device=environment.switcher_caar_device,
        max_planning_steps=environment.switcher_max_planning_steps,
        team_reward_coefficient=environment.switcher_team_reward_coefficient,
        feature_schema=environment.switcher_feature_schema,
    )


def register_switcher_components() -> None:
    base_train.register_custom_components()
    global_env_registry()[ENV_NAME] = create_switcher_env


def prepare_switcher_config(config: dict) -> tuple[object, object]:
    payload = deepcopy(config)
    declaration = payload.pop("candidate_policy", None)
    if not isinstance(declaration, dict):
        raise ValueError("Switcher config requires candidate_policy.")
    artifact = ArpeCandidateArtifact.from_mapping(declaration, PROJECT_ROOT)

    # Keep the duplicated environment field in sync automatically.  Users only
    # need to edit candidate_policy when selecting another ARPE checkpoint.
    payload.setdefault("environment", {})[
        "switcher_caar_weights_path"
    ] = artifact.weights_relative

    experiment, flat_config = base_train.validate_config(payload)
    if flat_config.encoder_custom != "switcher":
        raise ValueError(
            "Switcher requires encoder_custom='switcher'."
        )
    if flat_config.env != ENV_NAME:
        raise ValueError(
            f"Switcher requires environment {ENV_NAME!r}."
        )
    if bool(flat_config.use_rnn):
        raise ValueError("Switcher must remain feed-forward.")
    flat_config.full_config = deepcopy(flat_config.full_config)
    flat_config.full_config["candidate_policy"] = deepcopy(declaration)
    flat_config.candidate_policy = deepcopy(declaration)
    flat_config.switcher_integration_schema = ARPE_SWITCHER_ENV_SCHEMA
    flat_config.switcher_training_entrypoint_schema = ENTRYPOINT_SCHEMA
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
    parser.add_argument("--run_name")
    parser.add_argument("--train_dir")
    parser.add_argument("--train_for_env_steps", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    register_switcher_components()
    config_path = Path(args.config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    explicit = _apply_overrides(config, args)
    _, flat_config = prepare_switcher_config(config)
    base_train._sync_resume_cli_overrides(flat_config, explicit)
    if args.validate_only:
        print(
            json.dumps(
                {
                    "validated": True,
                    "schema": ENTRYPOINT_SCHEMA,
                    "integration_schema": ARPE_SWITCHER_ENV_SCHEMA,
                    "experiment": flat_config.experiment,
                    "target_frames": int(flat_config.train_for_env_steps),
                    "workers": int(flat_config.num_workers),
                    "candidate_policy": flat_config.candidate_policy,
                    "decision_scope": "aoreplan_nonwait_only",
                },
                sort_keys=True,
            )
        )
        return 0
    return int(run_rl(flat_config))


if __name__ == "__main__":
    raise SystemExit(main())
