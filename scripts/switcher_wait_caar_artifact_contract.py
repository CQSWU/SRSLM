#!/usr/bin/env python3
"""Create and reproduce the 100M wait-aware CAAR Switcher certificate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_IMPORT_ROOT))

from agents.switcher_caar_candidate import CaarCandidateArtifact
from scripts.switcher_artifact_contract import (
    atomic_json,
    checkpoint_identity,
    latest_regular_checkpoint,
    sha256_file,
)


SCHEMA = "switcher_wait_caar_training_artifact_v1"
EXPECTED_ENCODER = "switcher"
EXPECTED_ENVIRONMENT = "POMAPF-SRSLM-v0"
EXPECTED_FEATURE_SCHEMA = "srslm_switcher_state_v3"
EXPECTED_SCOPE = "aoreplan_nonwait_only"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"JSON artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON artifact is not a mapping: {path}")
    return payload


def _source_manifest(log_dir: Path, artifact: CaarCandidateArtifact) -> dict[str, Any]:
    before = log_dir / "source_before.sha256"
    after = log_dir / "source_after.sha256"
    require(before.is_file() and after.is_file(), "Source manifests are incomplete.")
    require(before.read_bytes() == after.read_bytes(), "Training inputs changed.")
    diff = log_dir / "source_hash_diff.txt"
    require(not diff.exists() or diff.stat().st_size == 0, "Source diff is non-empty.")
    text = before.read_text(encoding="utf-8")
    for digest in (
        artifact.checkpoint_sha256,
        artifact.config_sha256,
        artifact.base_checkpoint_sha256,
        artifact.base_config_sha256,
    ):
        require(digest in text, f"Source manifest does not bind {digest}.")
    entries = [line for line in text.splitlines() if line.strip()]
    require(entries, "Source manifest is empty.")
    return {
        "before_path": str(before.resolve()),
        "after_path": str(after.resolve()),
        "sha256": sha256_file(before),
        "entry_count": len(entries),
        "candidate_checkpoint_sha256": artifact.checkpoint_sha256,
        "candidate_config_sha256": artifact.config_sha256,
        "base_checkpoint_sha256": artifact.base_checkpoint_sha256,
        "base_config_sha256": artifact.base_config_sha256,
    }


def _validate_saved_config(
    config_path: Path,
    *,
    project_root: Path,
    expected_experiment: str,
    expected_target_frames: int,
) -> tuple[dict[str, Any], CaarCandidateArtifact]:
    config = _read_json(config_path)
    full = config.get("full_config")
    require(isinstance(full, dict), "Saved config has no full_config.")
    environment = full.get("environment")
    settings = full.get("experiment_settings")
    async_ppo = full.get("async_ppo")
    global_settings = full.get("global_settings")
    candidate = full.get("candidate_policy")
    require(
        all(isinstance(v, dict) for v in (environment, settings, async_ppo, global_settings, candidate)),
        "Saved wait-aware Switcher config is incomplete.",
    )
    artifact = CaarCandidateArtifact.from_mapping(candidate, project_root)
    artifact.verify_files()
    grid = environment.get("grid_config", {})
    expected = {
        "experiment": (config.get("experiment"), expected_experiment),
        "encoder": (config.get("encoder_custom"), EXPECTED_ENCODER),
        "environment": (environment.get("name"), EXPECTED_ENVIRONMENT),
        "feature_schema": (environment.get("switcher_feature_schema"), EXPECTED_FEATURE_SCHEMA),
        "candidate_path": (Path(environment.get("switcher_caar_weights_path", "")).as_posix(), artifact.weights_relative),
        "candidate_device": (environment.get("switcher_caar_device"), "cuda"),
        "seed": (global_settings.get("seed"), 0),
        "agents": (grid.get("num_agents"), 200),
        "map": (grid.get("map_name"), "maps/train.yaml"),
        "collision": (grid.get("collision_system"), "block_both"),
        "on_target": (grid.get("on_target"), "restart"),
        "horizon": (grid.get("max_episode_steps"), 512),
        "radius": (grid.get("obs_radius"), 5),
        "hidden": (settings.get("hidden_size"), 128),
        "ao_init": (settings.get("switcher_initial_ao_probability"), 0.1),
        "learning_rate": (settings.get("learning_rate"), 0.0001),
        "gamma": (settings.get("gamma"), 0.99),
        "target_frames": (settings.get("train_for_env_steps"), expected_target_frames),
        "workers": (async_ppo.get("num_workers"), 12),
        "batch": (async_ppo.get("batch_size"), 8192),
        "rollout": (async_ppo.get("rollout"), 32),
        "epochs": (async_ppo.get("num_epochs"), 2),
        "clip": (async_ppo.get("ppo_clip_ratio"), 0.2),
        "entropy": (async_ppo.get("exploration_loss_coeff"), 0.005),
        "value": (async_ppo.get("value_loss_coeff"), 0.5),
        "max_grad": (async_ppo.get("max_grad_norm"), 2.0),
        "use_rnn": (async_ppo.get("use_rnn"), False),
        "with_vtrace": (async_ppo.get("with_vtrace"), False),
        "team_reward": (environment.get("switcher_team_reward_coefficient"), 1.0),
        "max_planning_steps": (environment.get("switcher_max_planning_steps"), 10_000),
        "milestone_saving": (settings.get("save_milestones_sec"), -1),
    }
    mismatched = {
        key: {"actual": actual, "expected": wanted}
        for key, (actual, wanted) in expected.items()
        if actual != wanted
    }
    require(not mismatched, f"Saved wait-aware config differs: {mismatched}")
    require(environment.get("training_num_agents_by_worker") == [200] * 12, "Worker populations differ.")
    require(int(settings.get("save_best_after", -1)) > expected_target_frames, "Best checkpoint could be eligible.")
    return (
        {
            "path": str(config_path.resolve()),
            "sha256": sha256_file(config_path),
            "candidate_policy": candidate,
            "validated_values": {key: actual for key, (actual, _) in expected.items()},
        },
        artifact,
    )


def build_validation(
    *,
    weights_dir: Path,
    log_dir: Path,
    project_root: Path,
    checkpoint_path: Path,
    expected_experiment: str,
    expected_target_frames: int,
) -> dict[str, Any]:
    weights_dir = weights_dir.resolve()
    log_dir = log_dir.resolve()
    project_root = project_root.resolve()
    terminal = checkpoint_identity(checkpoint_path)
    latest = latest_regular_checkpoint(weights_dir)
    require(terminal == latest, "Selected checkpoint is not the latest regular checkpoint.")
    require(terminal["env_steps"] >= expected_target_frames, "Terminal checkpoint is below target.")
    require("best" not in terminal["filename"].lower(), "Best checkpoint selected.")
    require("milestone" not in terminal["filename"].lower(), "Milestone checkpoint selected.")
    config, artifact = _validate_saved_config(
        weights_dir / "config.json",
        project_root=project_root,
        expected_experiment=expected_experiment,
        expected_target_frames=expected_target_frames,
    )
    return {
        "schema": SCHEMA,
        "validated": True,
        "experiment": expected_experiment,
        "target_frames": expected_target_frames,
        "actual_checkpoint_frames": terminal["env_steps"],
        "initialization": {"seed": 0, "switcher_source": "from_scratch", "warm_start_checkpoint": None},
        "network_contract": {
            "architecture": "switcher_v3_hidden128_feed_forward",
            "feature_schema": EXPECTED_FEATURE_SCHEMA,
            "decision_scope": EXPECTED_SCOPE,
            "wait_routing": "aoreplan_wait_to_frozen_caar",
            "wait_detection_enabled": True,
            "actor_training_scope": EXPECTED_SCOPE,
            "critic_training_scope": "all_valid_states",
            "branch_0": "CAAR",
            "branch_1": "AORePlan",
        },
        "selection": {
            "kind": "terminal_latest_regular_checkpoint",
            "best_selected": False,
            "milestone_selected": False,
            "model_or_threshold_selection": False,
        },
        "terminal_checkpoint": terminal,
        "saved_config": config,
        "candidate_files": artifact.verify_files(),
        "source_manifest": _source_manifest(log_dir, artifact),
    }


def verify_validation(
    validation_path: Path,
    *,
    weights_dir: Path,
    log_dir: Path,
    project_root: Path,
    expected_experiment: str,
    expected_target_frames: int,
) -> dict[str, Any]:
    log_dir = log_dir.resolve()
    require((log_dir / "STATUS").read_text(encoding="utf-8").strip() == "COMPLETE", "STATUS is not COMPLETE.")
    require((log_dir / "COMPLETE").is_file(), "COMPLETE marker is missing.")
    saved = _read_json(validation_path)
    require(saved.get("schema") == SCHEMA and saved.get("validated") is True, "Wrong validation schema/status.")
    rebuilt = build_validation(
        weights_dir=weights_dir,
        log_dir=log_dir,
        project_root=project_root,
        checkpoint_path=Path(saved["terminal_checkpoint"]["path"]),
        expected_experiment=expected_experiment,
        expected_target_frames=expected_target_frames,
    )
    require(saved == rebuilt, "Validation no longer reproduces.")
    return rebuilt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    checkpoint = sub.add_parser("checkpoint")
    checkpoint.add_argument("--weights-dir", required=True, type=Path)
    for command in ("postflight", "verify"):
        item = sub.add_parser(command)
        item.add_argument("--weights-dir", required=True, type=Path)
        item.add_argument("--log-dir", required=True, type=Path)
        item.add_argument("--project-root", required=True, type=Path)
        item.add_argument("--expected-experiment", required=True)
        item.add_argument("--expected-target-frames", required=True, type=int)
        if command == "postflight":
            item.add_argument("--checkpoint", required=True, type=Path)
            item.add_argument("--output", required=True, type=Path)
        else:
            item.add_argument("--validation", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "checkpoint":
        print(json.dumps(latest_regular_checkpoint(args.weights_dir), sort_keys=True))
        return 0
    common = {
        "weights_dir": args.weights_dir,
        "log_dir": args.log_dir,
        "project_root": args.project_root,
        "expected_experiment": args.expected_experiment,
        "expected_target_frames": args.expected_target_frames,
    }
    if args.command == "postflight":
        payload = build_validation(checkpoint_path=args.checkpoint, **common)
        atomic_json(args.output, payload)
    else:
        payload = verify_validation(args.validation, **common)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
