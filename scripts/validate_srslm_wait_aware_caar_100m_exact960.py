#!/usr/bin/env python3
"""Validate the final 100M wait-aware CAAR Switcher exact960 run."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path


PROJECT_IMPORT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_IMPORT_ROOT))

from agents.switcher_caar_candidate import CaarCandidateArtifact
from run_experiments import validate_srslm_stats
from scripts.switcher_artifact_contract import latest_regular_checkpoint, sha256_file
from scripts.validate_srslm_wait_ablation_exact960 import (
    EXPECTED_MAP_SHA256,
    EXPECTED_POPULATIONS,
    EXPECTED_ROWS,
    EXPECTED_SEEDS,
    atomic_json,
    finite,
    map_names,
    validate_move_metrics,
    validate_result_journal,
    validate_routing,
)


SCHEMA = "srslm_wait_aware_caar_100m_exact960_validation_v1"
TRAINING_SCHEMA = "switcher_wait_caar_training_artifact_v1"
EXPECTED_SWITCHER_FRAMES = 100_016_128
EXPECTED_SWITCHER_CHECKPOINT_SHA256 = (
    "4973fa420a093e043d2aafb2340863a2be3ad7dda3362ef278a98ef8c1a75185"
)
EXPECTED_SWITCHER_MODEL_SHA256 = (
    "c2bd85a0cbcffe49dec8a393e84f022efe9bc8ce916190b497d0571acbb75aa9"
)
EXPECTED_SWITCHER_CONFIG_SHA256 = (
    "de387d7b00f7cb0d56b11d78389d702d301a39fb33a7f3f666189c685e7c0bc6"
)
EXPECTED_TRAINING_VALIDATION_SHA256 = (
    "b326895585ba64ef7842003dde9cd2edf26824a4b578cdf451dd57e4a01f26f1"
)
EXPECTED_CAAR_CHECKPOINT_SHA256 = (
    "497118e3aa4fbaecde35e53f31fe3126e11c1a1e5b0b621b89ac0d340002d41b"
)
EXPECTED_CAAR_CONFIG_SHA256 = (
    "e76a2b238f196752ec358ce8946eb353caa3a4fe3e4df2a92cf812506d008747"
)
EXPECTED_BASE_CHECKPOINT_SHA256 = (
    "f70a305ee68546be95e0a93d7f61c9aec435a50da20624a3b382af2276ad79d2"
)
EXPECTED_BASE_CONFIG_SHA256 = (
    "74c5cc0f1c5fdc0043bfcaa2e48e3be9c46c2c652f489a2b83379788e5da69b9"
)
EXPECTED_CANDIDATE_MANIFEST_SHA256 = (
    "75df038934fd10a71ce5b7e97aca7456546a18940553aa49eb454c89510e654f"
)
EXPECTED_EPISODE_FRESH_REASON = "disabled_to_preserve_episode_fresh_policy_state"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict:
    require(path.is_file(), f"Missing JSON artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON artifact is not a mapping: {path}")
    return payload


def compare_identity(actual: dict, expected: dict, label: str) -> None:
    for key in (
        "filename",
        "env_steps",
        "train_step",
        "checkpoint_sha256",
        "policy_model_sha256",
    ):
        require(actual.get(key) == expected.get(key), f"{label} differs at {key}")


def validate_training(
    *,
    project_root: Path,
    weights_dir: Path,
    training_validation_path: Path,
    candidate_manifest_path: Path,
) -> tuple[dict, dict, CaarCandidateArtifact]:
    require(
        sha256_file(training_validation_path) == EXPECTED_TRAINING_VALIDATION_SHA256,
        "Training certificate SHA256 differs.",
    )
    training = read_json(training_validation_path)
    require(training.get("schema") == TRAINING_SCHEMA, "Training schema differs.")
    require(training.get("validated") is True, "Training certificate is not validated.")
    require(int(training.get("target_frames", -1)) == 100_000_000, "Training target differs.")
    require(
        int(training.get("actual_checkpoint_frames", -1)) == EXPECTED_SWITCHER_FRAMES,
        "Training terminal frame count differs.",
    )
    contract = training.get("network_contract")
    require(isinstance(contract, dict), "Training network contract is missing.")
    expected_contract = {
        "architecture": "switcher_v3_hidden128_feed_forward",
        "feature_schema": "srslm_switcher_state_v3",
        "decision_scope": "aoreplan_nonwait_only",
        "wait_routing": "aoreplan_wait_to_frozen_caar",
        "wait_detection_enabled": True,
        "actor_training_scope": "aoreplan_nonwait_only",
        "critic_training_scope": "all_valid_states",
        "branch_0": "CAAR",
        "branch_1": "AORePlan",
    }
    require(contract == expected_contract, "Training routing contract differs.")
    selection = training.get("selection")
    require(
        isinstance(selection, dict)
        and selection.get("kind") == "terminal_latest_regular_checkpoint"
        and selection.get("best_selected") is False
        and selection.get("milestone_selected") is False
        and selection.get("model_or_threshold_selection") is False,
        "Training checkpoint selection differs.",
    )

    latest = latest_regular_checkpoint(weights_dir)
    terminal = training.get("terminal_checkpoint")
    require(isinstance(terminal, dict), "Training terminal identity is missing.")
    compare_identity(latest, terminal, "Terminal checkpoint")
    require(latest["env_steps"] == EXPECTED_SWITCHER_FRAMES, "Switcher frame count differs.")
    require(
        latest["checkpoint_sha256"] == EXPECTED_SWITCHER_CHECKPOINT_SHA256,
        "Switcher checkpoint SHA256 differs.",
    )
    require(
        latest["policy_model_sha256"] == EXPECTED_SWITCHER_MODEL_SHA256,
        "Switcher policy-model SHA256 differs.",
    )

    config_path = weights_dir / "config.json"
    require(
        sha256_file(config_path) == EXPECTED_SWITCHER_CONFIG_SHA256,
        "Switcher config SHA256 differs.",
    )
    config = read_json(config_path)
    declaration = (config.get("full_config") or {}).get("candidate_policy")
    require(isinstance(declaration, dict), "Switcher has no pinned CAAR declaration.")
    require(
        sha256_file(candidate_manifest_path) == EXPECTED_CANDIDATE_MANIFEST_SHA256,
        "CAAR candidate manifest SHA256 differs.",
    )
    manifest = read_json(candidate_manifest_path)
    require(manifest == declaration, "Candidate manifest differs from the saved Switcher.")
    artifact = CaarCandidateArtifact.from_mapping(declaration, project_root)
    verified = artifact.verify_files()
    require(artifact.checkpoint_sha256 == EXPECTED_CAAR_CHECKPOINT_SHA256, "CAAR checkpoint differs.")
    require(artifact.config_sha256 == EXPECTED_CAAR_CONFIG_SHA256, "CAAR config differs.")
    require(artifact.base_checkpoint_sha256 == EXPECTED_BASE_CHECKPOINT_SHA256, "EPOM-L checkpoint differs.")
    require(artifact.base_config_sha256 == EXPECTED_BASE_CONFIG_SHA256, "EPOM-L config differs.")

    log_dir = training_validation_path.parent
    require((log_dir / "COMPLETE").is_file(), "Training COMPLETE marker is missing.")
    require((log_dir / "STATUS").read_text(encoding="utf-8").strip() == "COMPLETE", "Training STATUS differs.")
    before = log_dir / "source_before.sha256"
    after = log_dir / "source_after.sha256"
    require(before.is_file() and after.is_file(), "Training source manifests are missing.")
    require(before.read_bytes() == after.read_bytes(), "Training sources changed.")
    recorded_source = training.get("source_manifest")
    require(isinstance(recorded_source, dict), "Training source identity is missing.")
    require(sha256_file(before) == recorded_source.get("sha256"), "Training source hash differs.")
    entries = [line for line in before.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(entries) == int(recorded_source.get("entry_count", -1)), "Training source entry count differs.")
    return (
        {
            "path": str(training_validation_path.resolve()),
            "sha256": EXPECTED_TRAINING_VALIDATION_SHA256,
            "target_frames": 100_000_000,
            "actual_checkpoint_frames": EXPECTED_SWITCHER_FRAMES,
            "source_manifest_sha256": sha256_file(before),
            "terminal_checkpoint": latest,
        },
        declaration,
        artifact,
    )


def validate_runtime_candidate(row: dict, declaration: dict, artifact: CaarCandidateArtifact, label: str) -> None:
    require(row.get("switcher_candidate_policy") == declaration, f"{label}: candidate declaration differs")
    runtime = row.get("switcher_candidate_artifact")
    require(isinstance(runtime, dict), f"{label}: candidate diagnostics are missing")
    expected = {
        "weights_path": artifact.weights_relative,
        "checkpoint_path": artifact.checkpoint_relative,
        "checkpoint_sha256": EXPECTED_CAAR_CHECKPOINT_SHA256,
        "config_sha256": EXPECTED_CAAR_CONFIG_SHA256,
        "base_weights_path": artifact.base_weights_relative,
        "base_checkpoint_path": artifact.base_checkpoint_relative,
        "base_checkpoint_sha256": EXPECTED_BASE_CHECKPOINT_SHA256,
        "base_config_sha256": EXPECTED_BASE_CONFIG_SHA256,
        "checkpoint_selection": "exact_milestone",
        "frozen": True,
    }
    for key, value in expected.items():
        require(runtime.get(key) == value, f"{label}: candidate {key} differs")


def ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--map-list", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--weights-dir", required=True, type=Path)
    parser.add_argument("--training-validation", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--result-journal", required=True, type=Path)
    parser.add_argument("--expected-journal-contract-sha256", required=True)
    parser.add_argument("--expected-code-snapshot-sha256", required=True)
    parser.add_argument("--expected-workers", required=True, type=int)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    result_path = args.input.resolve()
    require(result_path.parent == output_dir and result_path.is_file(), "Result path differs.")
    require(not (output_dir / "COMPLETE").exists(), "COMPLETE predates validation.")
    for stem in ("source", "code", "artifact"):
        before = output_dir / f"{stem}_before.sha256"
        after = output_dir / f"{stem}_after.sha256"
        require(before.is_file() and after.is_file(), f"{stem} manifests are missing.")
        require(before.read_bytes() == after.read_bytes(), f"{stem} snapshot changed.")
    require(
        sha256_file(output_dir / "code_before.sha256") == args.expected_code_snapshot_sha256,
        "Code snapshot SHA256 differs.",
    )
    run_contract_path = output_dir / "RUN_CONTRACT.json"
    require(run_contract_path.is_file(), "RUN_CONTRACT.json is missing.")
    require(
        sha256_file(run_contract_path) == args.expected_journal_contract_sha256,
        "Run/journal contract SHA256 differs.",
    )
    run_contract = read_json(run_contract_path)
    expected_run_contract = {
        "schema": "srslm_wait_aware_caar_100m_exact960_run_contract_v1",
        "algorithm": "SRSLM",
        "hybrid_mode": "aoreplan_wait_bypass_switcher_v3",
        "map_list_sha256": EXPECTED_MAP_SHA256,
        "populations": list(EXPECTED_POPULATIONS),
        "seeds": list(EXPECTED_SEEDS),
        "collision_system": "block_both",
        "on_target": "restart",
        "max_steps": 512,
        "obs_radius": 5,
        "expected_rows": EXPECTED_ROWS,
        "switcher_frames": EXPECTED_SWITCHER_FRAMES,
        "switcher_checkpoint_sha256": EXPECTED_SWITCHER_CHECKPOINT_SHA256,
        "switcher_policy_model_sha256": EXPECTED_SWITCHER_MODEL_SHA256,
        "caar_checkpoint_sha256": EXPECTED_CAAR_CHECKPOINT_SHA256,
        "training_validation_sha256": EXPECTED_TRAINING_VALIDATION_SHA256,
        "candidate_manifest_sha256": EXPECTED_CANDIDATE_MANIFEST_SHA256,
        "code_snapshot_sha256": args.expected_code_snapshot_sha256,
        "workers": args.expected_workers,
    }
    require(run_contract == expected_run_contract, "Frozen run contract differs.")

    training, declaration, artifact = validate_training(
        project_root=project_root,
        weights_dir=args.weights_dir.resolve(),
        training_validation_path=args.training_validation.resolve(),
        candidate_manifest_path=args.candidate_manifest.resolve(),
    )
    maps = map_names(args.map_list.resolve())
    payload = read_json(result_path)
    metadata = payload.get("metadata")
    rows = payload.get("results")
    require(isinstance(metadata, dict) and isinstance(rows, list), "Malformed result payload.")
    require(metadata.get("algorithms") == ["SRSLM"], "Metadata algorithm differs.")
    require(metadata.get("agent_counts") == list(EXPECTED_POPULATIONS), "Population grid differs.")
    require(metadata.get("seeds") == list(EXPECTED_SEEDS), "Seed grid differs.")
    require(tuple(metadata.get("maps", {}).keys()) == maps, "Map grid differs.")
    require(metadata.get("map_list_sha256") == EXPECTED_MAP_SHA256, "Map-list hash differs.")
    require(metadata.get("collision_system") == "block_both", "Collision mode differs.")
    require(metadata.get("on_target") == "restart", "Target mode differs.")
    require(int(metadata.get("max_steps", -1)) == 512, "Horizon differs.")
    require(int(metadata.get("obs_radius", -1)) == 5, "Observation radius differs.")
    require(int(metadata.get("workers", -1)) == args.expected_workers, "Worker count differs.")
    require(metadata.get("cache_algorithms_requested") is True, "Caching was not requested.")
    require(
        metadata.get("cache_algorithms_effective_by_algorithm", {}).get("SRSLM") is False,
        "SRSLM state was reused across episodes.",
    )
    require(
        metadata.get("cache_algorithms_exceptions", {}).get("SRSLM") == EXPECTED_EPISODE_FRESH_REASON,
        "Episode-fresh provenance differs.",
    )
    require(metadata.get("hybrid_mode") == "aoreplan_wait_bypass_switcher_v3", "Metadata routing mode differs.")
    require(metadata.get("result_journal_contract") == args.expected_journal_contract_sha256, "Metadata journal contract differs.")
    integrity = metadata.get("integrity")
    require(isinstance(integrity, dict), "Integrity metadata is missing.")
    require(integrity.get("hybrid_mode") == "aoreplan_wait_bypass_switcher_v3", "Integrity routing mode differs.")
    require(integrity.get("caar_checkpoint_sha256") == EXPECTED_CAAR_CHECKPOINT_SHA256, "Integrity CAAR differs.")
    require(integrity.get("switcher_checkpoint_sha256") == EXPECTED_SWITCHER_CHECKPOINT_SHA256, "Integrity Switcher differs.")

    require(len(rows) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows, found {len(rows)}.")
    actual = set()
    for index, row in enumerate(rows):
        label = f"row[{index}]"
        require(isinstance(row, dict), f"{label}: row is malformed")
        require(row.get("error") in (None, "", False), f"{label}: contains an error")
        finite(row, label)
        key = (row.get("map_name"), int(row.get("num_agents", -1)), int(row.get("seed", -1)))
        require(key[0] in maps and key[1] in EXPECTED_POPULATIONS and key[2] in EXPECTED_SEEDS, f"{label}: tuple differs")
        require(key not in actual, f"Duplicate tuple: {key}")
        actual.add(key)
        require(row.get("algorithm") == "SRSLM", f"{label}: algorithm differs")
        require(int(row.get("max_steps", -1)) == 512, f"{label}: horizon differs")
        require(row.get("on_target") == "restart", f"{label}: target mode differs")
        require(int(row.get("total_experiments", -1)) == EXPECTED_ROWS, f"{label}: total differs")
        require(isinstance(row.get("avg_throughput"), (int, float)) and math.isfinite(float(row["avg_throughput"])), f"{label}: throughput differs")
        validate_srslm_stats(row)
        validate_routing(row, "SRSLM", EXPECTED_SWITCHER_CHECKPOINT_SHA256, label)
        validate_move_metrics(row, label)
        require(row.get("switcher_config_sha256") == EXPECTED_SWITCHER_CONFIG_SHA256, f"{label}: Switcher config differs")
        validate_runtime_candidate(row, declaration, artifact, label)
    expected = {
        (map_name, population, seed)
        for map_name in maps
        for population in EXPECTED_POPULATIONS
        for seed in EXPECTED_SEEDS
    }
    require(actual == expected, "Result tuples do not match exact960.")
    journal = validate_result_journal(
        args.result_journal.resolve(),
        contract=args.expected_journal_contract_sha256,
        rows=rows,
    )

    def summary(population: int | None = None) -> dict:
        selected = [row for row in rows if population is None or row["num_agents"] == population]
        active = sum(int(row["active_agent_step_count"]) for row in selected)
        waits = sum(int(row["wait_action_count"]) for row in selected)
        moves = sum(int(row["move_attempt_count"]) for row in selected)
        blocked = sum(int(row["conflict_count"]) for row in selected)
        calls = sum(int(row["switcher_model_choice_count"]) for row in selected)
        bypasses = sum(int(row["aoreplan_wait_bypass_count"]) for row in selected)
        return {
            "episodes": len(selected),
            "mean_throughput": statistics.fmean(float(row["avg_throughput"]) for row in selected),
            "mean_congestion_rate": statistics.fmean(float(row["congestion_rate"]) for row in selected),
            "active_agent_steps": active,
            "wait_action_count": waits,
            "move_attempt_count": moves,
            "blocked_move_count": blocked,
            "switcher_model_call_count": calls,
            "aoreplan_wait_bypass_count": bypasses,
            "pooled_wait_action_rate": ratio(waits, active),
            "pooled_blocked_move_rate": ratio(blocked, moves),
            "pooled_switcher_model_call_rate": ratio(calls, active),
            "pooled_aoreplan_wait_bypass_rate": ratio(bypasses, active),
        }

    report = {
        "schema": SCHEMA,
        "validated": True,
        "algorithm": "SRSLM",
        "hybrid_mode": "aoreplan_wait_bypass_switcher_v3",
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "source_manifest_sha256": sha256_file(output_dir / "source_before.sha256"),
        "code_snapshot_sha256": sha256_file(output_dir / "code_before.sha256"),
        "artifact_snapshot_sha256": sha256_file(output_dir / "artifact_before.sha256"),
        "map_list_sha256": EXPECTED_MAP_SHA256,
        "rows": EXPECTED_ROWS,
        "populations": list(EXPECTED_POPULATIONS),
        "seeds": list(EXPECTED_SEEDS),
        "collision_system": "block_both",
        "on_target": "restart",
        "max_steps": 512,
        "obs_radius": 5,
        "workers": args.expected_workers,
        "switcher_checkpoint_sha256": EXPECTED_SWITCHER_CHECKPOINT_SHA256,
        "switcher_policy_model_sha256": EXPECTED_SWITCHER_MODEL_SHA256,
        "switcher_config_sha256": EXPECTED_SWITCHER_CONFIG_SHA256,
        "caar_checkpoint_sha256": EXPECTED_CAAR_CHECKPOINT_SHA256,
        "caar_config_sha256": EXPECTED_CAAR_CONFIG_SHA256,
        "training_provenance": training,
        "result_journal": journal,
        "overall": summary(),
        "by_population": {str(population): summary(population) for population in EXPECTED_POPULATIONS},
    }
    atomic_json(output_dir / "VALIDATION.json", report)
    (output_dir / "STATUS").write_text("COMPLETE\n", encoding="utf-8")
    (output_dir / "COMPLETE").touch(exist_ok=False)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
