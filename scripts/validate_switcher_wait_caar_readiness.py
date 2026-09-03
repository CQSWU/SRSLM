#!/usr/bin/env python3
"""Validate the terminal wait-aware Switcher with one real SRSLM episode.

This certificate binds the terminal Switcher checkpoint, its frozen CAAR
candidate, the completed training certificate, and a real block_both rollout.
It is intentionally a readiness gate, not a substitute for the later exact960
paper evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


PROJECT_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_IMPORT_ROOT))

from agents.switcher_caar_candidate import CaarCandidateArtifact
from run_experiments import validate_srslm_stats
from scripts.switcher_artifact_contract import (
    atomic_json,
    latest_regular_checkpoint,
    sha256_file,
)


SCHEMA = "switcher_wait_caar_exact960_readiness_v1"
TRAINING_SCHEMA = "switcher_wait_caar_training_artifact_v1"
EXPECTED_PROTOCOL = {
    "algorithm": "SRSLM",
    "map_name": "mazes-s40_wc4_od30",
    "num_agents": 200,
    "seed": 0,
    "collision_system": "block_both",
    "on_target": "restart",
    "max_steps": 512,
    "obs_radius": 5,
    "workers": 1,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"JSON artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON artifact is not a mapping: {path}")
    return payload


def validate_source_freeze(result_dir: Path) -> dict[str, Any]:
    before = result_dir / "source_before.sha256"
    after = result_dir / "source_after.sha256"
    require(before.is_file() and after.is_file(), "Smoke source manifests are incomplete.")
    require(before.read_bytes() == after.read_bytes(), "Smoke sources changed during inference.")
    entries = [line for line in before.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(entries, "Smoke source manifest is empty.")
    return {
        "before_path": str(before.resolve()),
        "after_path": str(after.resolve()),
        "sha256": sha256_file(before),
        "entry_count": len(entries),
    }


def validate_protocol(metadata: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    actual = {
        "algorithm": row.get("algorithm"),
        "map_name": row.get("map_name"),
        "num_agents": row.get("num_agents"),
        "seed": row.get("seed"),
        "collision_system": metadata.get("collision_system"),
        "on_target": metadata.get("on_target"),
        "max_steps": metadata.get("max_steps"),
        "obs_radius": metadata.get("obs_radius"),
        "workers": metadata.get("workers"),
    }
    require(actual == EXPECTED_PROTOCOL, f"Smoke protocol differs: {actual}")
    require(metadata.get("algorithms") == ["SRSLM"], "Smoke algorithm grid differs.")
    require(metadata.get("agent_counts") == [200], "Smoke population grid differs.")
    require(metadata.get("seeds") == [0], "Smoke seed grid differs.")
    require(metadata.get("maps") == {"mazes-s40_wc4_od30": "mazes-s40_wc4_od30"}, "Smoke map grid differs.")
    require(metadata.get("cache_algorithms_requested") is False, "Smoke used model caching.")
    require(
        metadata.get("cache_algorithms_effective_by_algorithm") == {"SRSLM": False},
        "Smoke unexpectedly reused an SRSLM object.",
    )
    return actual


def validate_candidate(
    *,
    declaration: dict[str, Any],
    row: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    artifact = CaarCandidateArtifact.from_mapping(declaration, project_root)
    files = artifact.verify_files()
    require(row.get("switcher_candidate_policy") == declaration, "Runtime CAAR declaration differs from the saved Switcher config.")
    runtime = row.get("switcher_candidate_artifact")
    require(isinstance(runtime, dict), "Runtime CAAR artifact diagnostics are missing.")
    expected = {
        "checkpoint_sha256": artifact.checkpoint_sha256,
        "config_sha256": artifact.config_sha256,
        "base_checkpoint_sha256": artifact.base_checkpoint_sha256,
        "base_config_sha256": artifact.base_config_sha256,
        "weights_path": artifact.weights_relative,
        "checkpoint_path": artifact.checkpoint_relative,
        "base_weights_path": artifact.base_weights_relative,
        "base_checkpoint_path": artifact.base_checkpoint_relative,
        "frozen": True,
    }
    mismatched = {
        key: {"expected": value, "actual": runtime.get(key)}
        for key, value in expected.items()
        if runtime.get(key) != value
    }
    require(not mismatched, f"Runtime CAAR artifact differs: {mismatched}")
    return {"declaration": declaration, "verified_files": files}


def build_validation(
    *,
    project_root: Path,
    result_dir: Path,
    weights_dir: Path,
    training_validation_path: Path,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    result_dir = result_dir.resolve()
    weights_dir = weights_dir.resolve()
    require((result_dir / "STATUS").read_text(encoding="utf-8").strip() == "COMPLETE", "Smoke STATUS is not COMPLETE.")
    require((result_dir / "COMPLETE").is_file(), "Smoke COMPLETE marker is missing.")

    training = read_json(training_validation_path.resolve())
    require(training.get("schema") == TRAINING_SCHEMA and training.get("validated") is True, "Training certificate is not valid.")
    terminal = latest_regular_checkpoint(weights_dir)
    training_terminal = training.get("terminal_checkpoint")
    require(isinstance(training_terminal, dict), "Training terminal checkpoint is missing.")
    identity_keys = (
        "filename",
        "env_steps",
        "train_step",
        "checkpoint_sha256",
        "policy_model_sha256",
    )
    require(
        all(terminal.get(key) == training_terminal.get(key) for key in identity_keys),
        "Terminal checkpoint no longer matches the training certificate.",
    )
    require(terminal["env_steps"] >= 100_000_000, "Terminal Switcher is below 100M environment steps.")

    config_path = weights_dir / "config.json"
    saved_config = read_json(config_path)
    declaration = (saved_config.get("full_config") or {}).get("candidate_policy")
    require(isinstance(declaration, dict), "Saved Switcher has no pinned CAAR declaration.")

    result_path = result_dir / "result.json"
    result = read_json(result_path)
    rows = result.get("results")
    metadata = result.get("metadata")
    require(isinstance(rows, list) and len(rows) == 1, "Smoke must contain exactly one result row.")
    require(isinstance(metadata, dict), "Smoke metadata is missing.")
    row = rows[0]
    require(isinstance(row, dict), "Smoke result row is malformed.")
    require(not row.get("error"), f"Smoke row contains an error: {row.get('error')}")
    for key in ("avg_throughput", "congestion_rate", "run_time_seconds"):
        value = row.get(key)
        require(isinstance(value, (int, float)) and math.isfinite(value), f"Smoke {key} is not finite.")

    validate_srslm_stats(row)
    protocol = validate_protocol(metadata, row)
    total = int(row["total_action_count"])
    choices = int(row["switcher_choice_count"])
    bypasses = int(row["aoreplan_wait_bypass_count"])
    executed_ao = int(row["executed_ao_count"])
    executed_caar = int(row["executed_caar_count"])
    require(total == 200 * 512, "Smoke action count does not equal population times horizon.")
    require(choices + bypasses == total, "Wait bypass and Switcher decisions do not cover every action.")
    require(executed_ao + executed_caar == total, "Executed branches do not cover every action.")
    require(row.get("wait_detection_enabled") is True, "Wait routing is disabled.")
    require(row.get("switcher_decision_scope") == "aoreplan_nonwait_only", "Switcher received wait states.")
    require(row.get("hybrid_mode") == "aoreplan_wait_bypass_switcher_v3", "Wrong runtime routing mode.")
    require(row.get("switcher_checkpoint_sha256") == terminal["checkpoint_sha256"], "Runtime Switcher checkpoint differs.")
    require(row.get("switcher_config_sha256") == sha256_file(config_path), "Runtime Switcher config differs.")
    candidate = validate_candidate(
        declaration=declaration,
        row=row,
        project_root=project_root,
    )

    return {
        "schema": SCHEMA,
        "validated": True,
        "scope": "real_episode_readiness_gate_not_exact960_result",
        "exact960_ready": True,
        "training_validation": {
            "path": str(training_validation_path.resolve()),
            "sha256": sha256_file(training_validation_path),
            "schema": training["schema"],
            "target_frames": training["target_frames"],
            "actual_checkpoint_frames": training["actual_checkpoint_frames"],
        },
        "switcher": {
            "weights_dir": str(weights_dir),
            "config_path": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            "terminal_checkpoint": terminal,
        },
        "candidate": candidate,
        "smoke": {
            "result_path": str(result_path.resolve()),
            "result_sha256": sha256_file(result_path),
            "protocol": protocol,
            "row_summary": {
                "avg_throughput": row["avg_throughput"],
                "congestion_rate": row["congestion_rate"],
                "total_action_count": total,
                "switcher_choice_count": choices,
                "aoreplan_wait_bypass_count": bypasses,
                "executed_ao_count": executed_ao,
                "executed_caar_count": executed_caar,
            },
            "source_freeze": validate_source_freeze(result_dir),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--weights-dir", required=True, type=Path)
    parser.add_argument("--training-validation", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    output = args.output or args.result_dir / "VALIDATION.json"
    report = build_validation(
        project_root=args.project_root,
        result_dir=args.result_dir,
        weights_dir=args.weights_dir,
        training_validation_path=args.training_validation,
    )
    atomic_json(output, report)
    print(json.dumps(report["smoke"]["row_summary"], sort_keys=True))
    print(f"VALIDATED {Path(output).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
