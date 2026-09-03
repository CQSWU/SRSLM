#!/usr/bin/env python3
"""Independently validate one current-V3 SRSLM wait-ablation exact960."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path

try:
    from scripts.switcher_artifact_contract import (
        EXPECTED_CAAR_CHECKPOINT_SHA256,
        MIN_FORMAL_FRAMES,
        SCHEMA as TRAINING_SCHEMA,
        checkpoint_identity,
        source_manifest_identity,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...py`` execution.
    from switcher_artifact_contract import (
        EXPECTED_CAAR_CHECKPOINT_SHA256,
        MIN_FORMAL_FRAMES,
        SCHEMA as TRAINING_SCHEMA,
        checkpoint_identity,
        source_manifest_identity,
    )


ALGORITHMS = (
    "SRSLM-NoWaitDetect",
    "SRSLM-WaitDetectOnly",
    "SRSLM",
)
EXPECTED_MAP_SHA256 = (
    "da5c3d4cbd4cbdc8ce9f6b271ca258d4e7b69d6aa76524c6d17201718efb02f0"
)
EXPECTED_POPULATIONS = (100, 200, 300, 400, 500, 600)
EXPECTED_SEEDS = (0, 42, 123, 2024, 3407)
EXPECTED_ROWS = 32 * len(EXPECTED_POPULATIONS) * len(EXPECTED_SEEDS)
MAX_FORMAL_FRAMES = MIN_FORMAL_FRAMES + 1_000_000
EXPECTED_DEFAULT_SWITCHER_FRAMES = 500_015_104
EXPECTED_DEFAULT_SWITCHER_CHECKPOINT_SHA256 = (
    "2ddb0f4e639076a8d2300e4498b0725f1d7c7646ab5510024af8e4ac02c50e02"
)
EXPECTED_DEFAULT_SWITCHER_MODEL_SHA256 = (
    "e6f0bb8ae1869cc2017a67f16e801b32d56b10676ac62ad351e6e55e34e354eb"
)
EXPECTED_MODES = {
    "SRSLM-NoWaitDetect": "all_state_switcher_v3",
    "SRSLM-WaitDetectOnly": "aoreplan_wait_detect_only_v3",
    "SRSLM": "aoreplan_wait_bypass_switcher_v3",
}
EXPECTED_CONGESTION_METRIC_VERSION = "submitted_nonwait_no_position_change_v1"
EXPECTED_EPISODE_FRESH_REASON = "disabled_to_preserve_episode_fresh_policy_state"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def map_names(path):
    path = Path(path)
    require(path.is_file(), f"Map list is missing: {path}")
    require(sha256(path) == EXPECTED_MAP_SHA256, "Map-list SHA256 differs")
    names = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        require(":" in line, f"Malformed map-list line: {raw!r}")
        names.append(line.split(":", 1)[0].strip())
    require(len(names) == 32 and len(set(names)) == 32, "Map list is not 32 unique maps")
    return tuple(names)


def finite(value, label):
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        require(math.isfinite(float(value)), f"{label} is non-finite")
    elif isinstance(value, dict):
        for key, child in value.items():
            finite(child, f"{label}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            finite(child, f"{label}[{index}]")


def probability(row, key, label):
    value = row.get(key)
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{label}: missing {key}",
    )
    require(math.isfinite(float(value)) and 0.0 <= float(value) <= 1.0, f"{label}: invalid {key}")


def integer_counts(row, label):
    keys = (
        "total_action_count",
        "switcher_choice_count",
        "switcher_model_choice_count",
        "selected_ao_count",
        "switcher_model_selected_ao_count",
        "executed_ao_count",
        "executed_caar_count",
        "aoreplan_wait_bypass_count",
        "branch_action_agreement_count",
        "static_astar_query_count",
        "aoreplan_commit_count",
    )
    result = {}
    for key in keys:
        value = row.get(key)
        require(
            isinstance(value, int) and not isinstance(value, bool) and value >= 0,
            f"{label}: invalid {key}",
        )
        result[key] = value
    return result


def _nonnegative_integer(row, key, label):
    value = row.get(key)
    require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{label}: invalid {key}",
    )
    return value


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def validate_move_metrics(row, label):
    """Validate explicit waits and failed non-wait moves from raw counters."""
    require(
        row.get("congestion_metric_version") == EXPECTED_CONGESTION_METRIC_VERSION,
        f"{label}: congestion metric version differs",
    )
    keys = (
        "environment_step_count_observed",
        "active_agent_step_count",
        "wait_action_count",
        "move_attempt_count",
        "successful_move_count",
        "conflict_count",
        "move_failure_count",
        "agent_conflict_count",
        "other_or_unattributed_conflict_count",
        "conflict_step_count",
    )
    counts = {key: _nonnegative_integer(row, key, label) for key in keys}
    require(
        counts["environment_step_count_observed"] > 0,
        f"{label}: no environment steps were observed",
    )
    require(
        counts["conflict_step_count"] <= counts["environment_step_count_observed"],
        f"{label}: conflict-step count exceeds the horizon",
    )
    require(
        counts["active_agent_step_count"]
        == counts["wait_action_count"] + counts["move_attempt_count"],
        f"{label}: active steps do not split into waits and move attempts",
    )
    require(
        counts["move_attempt_count"]
        == counts["successful_move_count"] + counts["conflict_count"],
        f"{label}: move attempts do not split into successes and blocked moves",
    )
    require(
        counts["move_failure_count"] == counts["conflict_count"],
        f"{label}: blocked-move counters differ",
    )
    require(
        counts["conflict_count"]
        == counts["agent_conflict_count"]
        + counts["other_or_unattributed_conflict_count"],
        f"{label}: blocked-move attribution does not sum",
    )
    require(
        counts["active_agent_step_count"] == row.get("total_action_count"),
        f"{label}: movement and Switcher action denominators differ",
    )
    expected_rates = {
        "congestion_rate": _ratio(
            counts["conflict_count"], counts["move_attempt_count"]
        ),
        "agent_conflict_rate": _ratio(
            counts["agent_conflict_count"], counts["move_attempt_count"]
        ),
        "conflict_agent_step_rate": _ratio(
            counts["conflict_count"], counts["active_agent_step_count"]
        ),
        "conflict_step_rate": _ratio(
            counts["conflict_step_count"], counts["environment_step_count_observed"]
        ),
    }
    for key, expected in expected_rates.items():
        value = row.get(key)
        require(
            isinstance(value, (int, float)) and not isinstance(value, bool),
            f"{label}: missing {key}",
        )
        require(
            math.isfinite(float(value))
            and 0.0 <= float(value) <= 1.0
            and math.isclose(float(value), expected, rel_tol=1e-11, abs_tol=1e-12),
            f"{label}: {key} differs from raw counters",
        )
    return counts


def validate_routing(row, algorithm, expected_checkpoint, label):
    require(row.get("hybrid_mode") == EXPECTED_MODES[algorithm], f"{label}: wrong mode")
    require(row.get("switch_pair") == ["CAAR", "AORePlan"], f"{label}: wrong branches")
    require(row.get("value_predictor_loaded") is False, f"{label}: value predictor loaded")
    require(row.get("switcher_feature_schema") == "srslm_switcher_state_v3", f"{label}: wrong schema")
    require(row.get("joint_conflict_prediction_enabled") is False, f"{label}: joint predictor enabled")
    counts = integer_counts(row, label)
    total = counts["total_action_count"]
    choices = counts["switcher_choice_count"]
    selected = counts["selected_ao_count"]
    executed_ao = counts["executed_ao_count"]
    executed_caar = counts["executed_caar_count"]
    bypasses = counts["aoreplan_wait_bypass_count"]
    require(executed_ao + executed_caar == total, f"{label}: branch counts do not sum")
    require(counts["aoreplan_commit_count"] <= total, f"{label}: too many AO commits")
    require(counts["branch_action_agreement_count"] <= total, f"{label}: too many agreements")

    if algorithm == "SRSLM-NoWaitDetect":
        require(row.get("ablation_name") == algorithm, f"{label}: wrong ablation label")
        require(row.get("switcher_training") == "PPO", f"{label}: Switcher is not PPO")
        require(row.get("selector_kind") == "ppo_two_branch_categorical", f"{label}: wrong selector")
        require(row.get("switcher_decision_scope") == "all_states", f"{label}: actor is masked")
        require(row.get("wait_detection_enabled") is False, f"{label}: wait detector enabled")
        require(row.get("learned_switcher_called") is True, f"{label}: model was not called")
        require(choices == total and bypasses == 0, f"{label}: all-state accounting differs")
        require(selected == executed_ao, f"{label}: selected AO differs from execution")
        require(counts["switcher_model_choice_count"] == total, f"{label}: model missed states")
        require(counts["switcher_model_selected_ao_count"] == selected, f"{label}: model AO differs")
        require(row.get("switcher_stochastic") is True, f"{label}: model is deterministic")
    elif algorithm == "SRSLM-WaitDetectOnly":
        require(row.get("ablation_name") == algorithm, f"{label}: wrong ablation label")
        require(row.get("switcher_training") == "none", f"{label}: learned selector declared")
        require(row.get("selector_kind") == "deterministic_wait_detect_only", f"{label}: wrong selector")
        require(row.get("switcher_decision_scope") == "none", f"{label}: learned scope declared")
        require(row.get("wait_detection_enabled") is True, f"{label}: wait detector disabled")
        require(row.get("learned_switcher_called") is False, f"{label}: model was called")
        require(choices == selected == 0, f"{label}: Switcher counters are nonzero")
        require(counts["switcher_model_choice_count"] == 0, f"{label}: model choice count is nonzero")
        require(counts["switcher_model_selected_ao_count"] == 0, f"{label}: model AO count is nonzero")
        require(executed_caar == bypasses and executed_ao + bypasses == total, f"{label}: wait routing differs")
        require(row.get("switcher_stochastic") is False, f"{label}: deterministic rule marked stochastic")
    else:
        require(row.get("switcher_training") == "PPO", f"{label}: Switcher is not PPO")
        require(row.get("selector_kind") == "ppo_two_branch_categorical", f"{label}: wrong selector")
        require(row.get("switcher_decision_scope") == "aoreplan_nonwait_only", f"{label}: wrong scope")
        require(row.get("wait_detection_enabled") is True, f"{label}: wait detector disabled")
        require(row.get("learned_switcher_called") is True, f"{label}: model was not called")
        require(choices + bypasses == total, f"{label}: choice/bypass counts do not sum")
        require(selected == executed_ao, f"{label}: selected AO differs from execution")
        require(counts["switcher_model_choice_count"] == choices, f"{label}: model choices differ")
        require(counts["switcher_model_selected_ao_count"] == selected, f"{label}: model AO differs")
        require(row.get("switcher_stochastic") is True, f"{label}: model is deterministic")

    if algorithm != "SRSLM-WaitDetectOnly":
        require(row.get("switcher_checkpoint_sha256") == expected_checkpoint, f"{label}: wrong checkpoint")
        for key in (
            "switcher_sampled_ao_rate",
            "switcher_ao_probability_mean",
            "switcher_ao_probability_p05",
            "switcher_ao_probability_p95",
        ):
            probability(row, key, label)
    for key in (
        "switcher_choice_rate",
        "selected_ao_rate",
        "executed_ao_rate",
        "aoreplan_wait_bypass_rate",
        "branch_action_agreement_rate",
    ):
        probability(row, key, label)


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_training_provenance(
    path,
    *,
    algorithm,
    expected_checkpoint,
    expected_model,
    expected_validation_sha256,
    reference_model,
):
    path = Path(path).resolve()
    require(path.is_file(), "Switcher training VALIDATION.json is missing")
    require(sha256(path) == expected_validation_sha256, "Training validation SHA256 differs")
    log_dir = path.parent
    require((log_dir / "COMPLETE").is_file(), "Switcher training COMPLETE is missing")
    require(
        (log_dir / "STATUS").read_text(encoding="utf-8").strip() == "COMPLETE",
        "Switcher training STATUS is not COMPLETE",
    )
    source = source_manifest_identity(log_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("validated") is True, "Switcher training is not validated")
    require(payload.get("schema") == TRAINING_SCHEMA, "Switcher training schema differs")
    require(int(payload.get("target_frames", -1)) == MIN_FORMAL_FRAMES, "Training target differs")
    require(
        int(payload.get("actual_checkpoint_frames", -1)) >= MIN_FORMAL_FRAMES,
        "Switcher checkpoint has fewer than 500M embedded frames",
    )
    require(
        int(payload.get("actual_checkpoint_frames", -1)) <= MAX_FORMAL_FRAMES,
        "Switcher checkpoint exceeds the 500M run tolerance",
    )
    require(payload.get("checkpoint_sha256") == expected_checkpoint, "Training checkpoint differs")
    require(payload.get("policy_model_sha256") == expected_model, "Training policy model differs")
    require(
        payload.get("frozen_caar_checkpoint_sha256")
        == EXPECTED_CAAR_CHECKPOINT_SHA256,
        "Training used a different frozen CAAR checkpoint",
    )
    recorded_source = payload.get("source_manifest")
    require(isinstance(recorded_source, dict), "Training source provenance is missing")
    require(recorded_source == source, "Training source provenance no longer matches")
    checkpoint = checkpoint_identity(payload.get("checkpoint_path"))
    require(checkpoint["checkpoint_sha256"] == expected_checkpoint, "Checkpoint bytes changed")
    require(checkpoint["policy_model_sha256"] == expected_model, "Checkpoint model changed")
    require(
        checkpoint["env_steps"] == payload.get("actual_checkpoint_frames"),
        "Checkpoint frame evidence differs",
    )
    smoke = payload.get("runtime_smoke")
    if smoke is not None:
        require(isinstance(smoke, dict), "Runtime-smoke provenance is malformed")
        smoke_path = Path(smoke.get("path", ""))
        require(smoke_path.is_file(), "Runtime-smoke certificate is missing")
        require(sha256(smoke_path) == smoke.get("sha256"), "Runtime-smoke certificate changed")
        require(
            smoke.get("switcher_checkpoint_sha256") == expected_checkpoint,
            "Runtime smoke used a different Switcher checkpoint",
        )
        smoke_payload = json.loads(smoke_path.read_text(encoding="utf-8"))
        require(smoke_payload.get("validated") is True, "Runtime smoke is not validated")
        if algorithm == "SRSLM-NoWaitDetect":
            require(
                smoke_payload.get("switcher_choice_count")
                == smoke_payload.get("total_action_count"),
                "NoWait runtime smoke did not select on every state",
            )
            require(
                smoke_payload.get("aoreplan_wait_bypass_count") == 0,
                "NoWait runtime smoke bypassed waits",
            )

    distinct = payload.get("distinct_from_default")
    if algorithm == "SRSLM-NoWaitDetect":
        require(payload.get("training_contract") == "all_states", "NoWait training contract differs")
        require(isinstance(distinct, dict), "NoWait model-distinction proof is missing")
        require(distinct.get("model_sha256_differs") is True, "NoWait model is not distinct")
        require(
            distinct.get("policy_model_sha256") == reference_model,
            "Default reference model SHA256 differs",
        )
        require(expected_model != reference_model, "NoWait and default policy models are identical")
    else:
        require(
            payload.get("training_contract") == "aoreplan_nonwait_only",
            "Default training contract differs",
        )
        require(reference_model is None, "Default SRSLM must not receive a reference model")
        require(distinct is None, "Default SRSLM contains a foreign distinction proof")
    return {
        "path": str(path),
        "sha256": expected_validation_sha256,
        "checkpoint_sha256": expected_checkpoint,
        "policy_model_sha256": expected_model,
        "actual_checkpoint_frames": checkpoint["env_steps"],
        "source_manifest_sha256": source["sha256"],
        "reference_policy_model_sha256": reference_model,
    }


def validate_result_journal(path, *, contract, rows):
    path = Path(path).resolve()
    raw = path.read_bytes()
    require(raw.endswith(b"\n"), "Result journal has a truncated final record")
    lines = raw.splitlines()
    require(lines, "Result journal is empty")
    header = json.loads(lines[0])
    require(
        header
        == {
            "record_type": "header",
            "schema": "experiment_result_journal_v1",
            "contract_sha256": contract,
            "expected_tasks": EXPECTED_ROWS,
        },
        "Result journal header/contract differs",
    )
    successful = {}
    failures = 0
    for line_number, raw_line in enumerate(lines[1:], 2):
        record = json.loads(raw_line)
        require(record.get("record_type") == "result", f"Journal record {line_number} differs")
        row = record.get("result")
        require(isinstance(row, dict), f"Journal record {line_number} has no row")
        key = (
            row.get("algorithm"),
            row.get("map_name"),
            int(row.get("num_agents", -1)),
            int(row.get("seed", -1)),
            str(row.get("task_id") or ""),
        )
        require(record.get("task_key") == list(key), f"Journal record {line_number} key differs")
        if row.get("error"):
            failures += 1
            continue
        require(key not in successful, f"Journal repeats successful tuple {key}")
        successful[key] = row
    expected = {
        (
            row.get("algorithm"),
            row.get("map_name"),
            int(row.get("num_agents", -1)),
            int(row.get("seed", -1)),
            str(row.get("task_id") or ""),
        ): row
        for row in rows
    }
    require(len(expected) == EXPECTED_ROWS, "Result JSON has duplicate journal keys")
    require(successful == expected, "Result journal and final result JSON differ")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "contract_sha256": contract,
        "successful_rows": len(successful),
        "failed_attempt_records": failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--map-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-workers", type=int, required=True)
    parser.add_argument("--expected-switcher-checkpoint-sha256")
    parser.add_argument("--expected-switcher-model-sha256")
    parser.add_argument("--reference-switcher-model-sha256")
    parser.add_argument("--training-validation", type=Path)
    parser.add_argument("--expected-training-validation-sha256")
    parser.add_argument("--expected-caar-checkpoint-sha256", required=True)
    parser.add_argument("--expected-code-snapshot-sha256", required=True)
    parser.add_argument("--result-journal", type=Path, required=True)
    parser.add_argument("--expected-journal-contract-sha256", required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    result_path = args.input.resolve()
    require(output_dir.is_dir(), "Output directory is missing")
    require(result_path.parent == output_dir and result_path.is_file(), "Result path differs")
    require(not (output_dir / "COMPLETE").exists(), "COMPLETE predates validation")
    before = output_dir / "source_before.sha256"
    after = output_dir / "source_after.sha256"
    require(before.is_file() and after.is_file(), "Source manifests are missing")
    require(before.read_bytes() == after.read_bytes(), "Tracked inputs changed")
    code_before = output_dir / "code_before.sha256"
    code_after = output_dir / "code_after.sha256"
    artifact_before = output_dir / "artifact_before.sha256"
    artifact_after = output_dir / "artifact_after.sha256"
    require(code_before.is_file() and code_after.is_file(), "Code manifests are missing")
    require(artifact_before.is_file() and artifact_after.is_file(), "Artifact manifests are missing")
    require(code_before.read_bytes() == code_after.read_bytes(), "Code snapshot changed")
    require(artifact_before.read_bytes() == artifact_after.read_bytes(), "Artifact snapshot changed")
    require(
        sha256(code_before) == args.expected_code_snapshot_sha256,
        "Code snapshot SHA256 differs",
    )
    require(
        args.expected_caar_checkpoint_sha256 == EXPECTED_CAAR_CHECKPOINT_SHA256,
        "Expected CAAR SHA256 is not the frozen paper artifact",
    )
    run_contract = output_dir / "RUN_CONTRACT.json"
    require(run_contract.is_file(), "RUN_CONTRACT.json is missing")
    require(
        sha256(run_contract) == args.expected_journal_contract_sha256,
        "Run/journal contract SHA256 differs",
    )
    maps = map_names(args.map_list.resolve())

    expected_checkpoint = args.expected_switcher_checkpoint_sha256
    if args.algorithm == "SRSLM-WaitDetectOnly":
        require(expected_checkpoint is None, "WaitDetectOnly must not bind a Switcher checkpoint")
        require(args.expected_switcher_model_sha256 is None, "WaitDetectOnly bound a model")
        require(args.training_validation is None, "WaitDetectOnly bound training provenance")
        require(args.expected_training_validation_sha256 is None, "WaitDetectOnly bound validation")
        require(args.reference_switcher_model_sha256 is None, "WaitDetectOnly bound a reference")
        require(
            not (output_dir / "SWITCHER_TRAINING_PROOF.json").exists(),
            "WaitDetectOnly unexpectedly contains a Switcher training proof",
        )
        training_provenance = None
    else:
        require(
            isinstance(expected_checkpoint, str)
            and len(expected_checkpoint) == 64
            and all(char in "0123456789abcdef" for char in expected_checkpoint),
            "A valid expected Switcher checkpoint SHA256 is required",
        )
        require(
            isinstance(args.expected_switcher_model_sha256, str)
            and len(args.expected_switcher_model_sha256) == 64,
            "A Switcher policy-model SHA256 is required",
        )
        require(args.training_validation is not None, "Training validation is required")
        require(
            isinstance(args.expected_training_validation_sha256, str)
            and len(args.expected_training_validation_sha256) == 64,
            "Training validation SHA256 is required",
        )
        if args.algorithm == "SRSLM-NoWaitDetect":
            require(
                isinstance(args.reference_switcher_model_sha256, str)
                and len(args.reference_switcher_model_sha256) == 64,
                "NoWaitDetect requires the default model SHA256",
            )
        training_provenance = validate_training_provenance(
            args.training_validation,
            algorithm=args.algorithm,
            expected_checkpoint=expected_checkpoint,
            expected_model=args.expected_switcher_model_sha256,
            expected_validation_sha256=args.expected_training_validation_sha256,
            reference_model=args.reference_switcher_model_sha256,
        )
        if args.algorithm == "SRSLM":
            require(
                expected_checkpoint == EXPECTED_DEFAULT_SWITCHER_CHECKPOINT_SHA256,
                "Default Switcher checkpoint is not the frozen 500,015,104-frame artifact",
            )
            require(
                args.expected_switcher_model_sha256
                == EXPECTED_DEFAULT_SWITCHER_MODEL_SHA256,
                "Default Switcher policy-model SHA256 differs",
            )
            require(
                training_provenance["actual_checkpoint_frames"]
                == EXPECTED_DEFAULT_SWITCHER_FRAMES,
                "Default Switcher frame count differs",
            )
        proof_path = output_dir / "SWITCHER_TRAINING_PROOF.json"
        require(proof_path.is_file(), "Frozen Switcher training proof is missing")
        frozen_proof = json.loads(proof_path.read_text(encoding="utf-8"))
        require(
            frozen_proof.get("validation_sha256")
            == args.expected_training_validation_sha256,
            "Frozen training proof validation SHA256 differs",
        )
        require(
            frozen_proof.get("checkpoint_sha256") == expected_checkpoint,
            "Frozen training proof checkpoint differs",
        )
        require(
            frozen_proof.get("policy_model_sha256")
            == args.expected_switcher_model_sha256,
            "Frozen training proof model differs",
        )

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata")
    rows = payload.get("results")
    require(isinstance(metadata, dict) and isinstance(rows, list), "Malformed result payload")
    require(metadata.get("algorithms") == [args.algorithm], "Metadata algorithm differs")
    require(metadata.get("agent_counts") == list(EXPECTED_POPULATIONS), "Population grid differs")
    require(metadata.get("seeds") == list(EXPECTED_SEEDS), "Seed grid differs")
    require(tuple(metadata.get("maps", {}).keys()) == maps, "Map grid differs")
    require(metadata.get("map_list_sha256") == EXPECTED_MAP_SHA256, "Metadata map hash differs")
    require(metadata.get("collision_system") == "block_both", "Collision mode differs")
    require(metadata.get("on_target") == "restart", "on_target differs")
    require(int(metadata.get("max_steps", -1)) == 512, "Horizon differs")
    require(int(metadata.get("obs_radius", -1)) == 5, "Radius differs")
    require(int(metadata.get("workers", -1)) == args.expected_workers, "Worker count differs")
    require(metadata.get("cache_algorithms_requested") is True, "Caching was not requested")
    require(
        metadata.get("cache_algorithms_effective_by_algorithm", {}).get(args.algorithm) is False,
        "Policy state was reused across episodes",
    )
    require(
        metadata.get("cache_algorithms_exceptions", {}).get(args.algorithm)
        == EXPECTED_EPISODE_FRESH_REASON,
        "Episode-fresh cache exception provenance differs",
    )
    require(metadata.get("hybrid_mode") == EXPECTED_MODES[args.algorithm], "Metadata mode differs")
    require(
        metadata.get("result_journal_contract")
        == args.expected_journal_contract_sha256,
        "Metadata journal contract differs",
    )
    integrity = metadata.get("integrity")
    require(isinstance(integrity, dict), "Integrity metadata is missing")
    require(integrity.get("algorithm", args.algorithm) == args.algorithm, "Integrity algorithm differs")
    require(integrity.get("hybrid_mode") == EXPECTED_MODES[args.algorithm], "Integrity mode differs")
    require(integrity.get("map_list_sha256") == EXPECTED_MAP_SHA256, "Integrity map differs")
    require(
        integrity.get("caar_checkpoint_sha256") == EXPECTED_CAAR_CHECKPOINT_SHA256,
        "Integrity CAAR checkpoint differs",
    )
    if args.algorithm == "SRSLM-WaitDetectOnly":
        require(integrity.get("switcher_checkpoint_sha256") is None, "WaitDetectOnly bound a Switcher")
    else:
        require(integrity.get("switcher_checkpoint_sha256") == expected_checkpoint, "Integrity checkpoint differs")

    require(len(rows) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows, found {len(rows)}")
    actual = set()
    for index, row in enumerate(rows):
        label = f"row[{index}]"
        require(isinstance(row, dict), f"{label}: not an object")
        require(row.get("error") in (None, "", False), f"{label}: contains an error")
        finite(row, label)
        require(row.get("algorithm") == args.algorithm, f"{label}: algorithm differs")
        key = (row.get("map_name"), int(row.get("num_agents", -1)), int(row.get("seed", -1)))
        require(key[0] in maps and key[1] in EXPECTED_POPULATIONS and key[2] in EXPECTED_SEEDS, f"{label}: tuple differs")
        require(key not in actual, f"Duplicate tuple: {key}")
        actual.add(key)
        require(int(row.get("max_steps", -1)) == 512, f"{label}: horizon differs")
        require(row.get("on_target") == "restart", f"{label}: on_target differs")
        require(int(row.get("total_experiments", -1)) == EXPECTED_ROWS, f"{label}: total differs")
        require(isinstance(row.get("avg_throughput"), (int, float)), f"{label}: throughput missing")
        require(isinstance(row.get("congestion_rate"), (int, float)), f"{label}: congestion missing")
        validate_routing(row, args.algorithm, expected_checkpoint, label)
        validate_move_metrics(row, label)
    expected = {
        (map_name, population, seed)
        for map_name in maps
        for population in EXPECTED_POPULATIONS
        for seed in EXPECTED_SEEDS
    }
    require(actual == expected, "Result tuples do not match exact960")
    journal = validate_result_journal(
        args.result_journal,
        contract=args.expected_journal_contract_sha256,
        rows=rows,
    )

    def summary(population=None):
        selected = [row for row in rows if population is None or row["num_agents"] == population]
        active_steps = sum(int(row["active_agent_step_count"]) for row in selected)
        wait_actions = sum(int(row["wait_action_count"]) for row in selected)
        move_attempts = sum(int(row["move_attempt_count"]) for row in selected)
        blocked_moves = sum(int(row["conflict_count"]) for row in selected)
        model_calls = sum(int(row["switcher_model_choice_count"]) for row in selected)
        switcher_choices = sum(int(row["switcher_choice_count"]) for row in selected)
        wait_bypasses = sum(int(row["aoreplan_wait_bypass_count"]) for row in selected)
        return {
            "episodes": len(selected),
            "mean_throughput": statistics.fmean(float(row["avg_throughput"]) for row in selected),
            "mean_congestion_rate": statistics.fmean(float(row["congestion_rate"]) for row in selected),
            "active_agent_steps": active_steps,
            "wait_action_count": wait_actions,
            "move_attempt_count": move_attempts,
            "blocked_move_count": blocked_moves,
            "switcher_model_call_count": model_calls,
            "switcher_choice_count": switcher_choices,
            "aoreplan_wait_bypass_count": wait_bypasses,
            "pooled_wait_action_rate": _ratio(wait_actions, active_steps),
            "pooled_blocked_move_rate": _ratio(blocked_moves, move_attempts),
            "pooled_stationary_agent_step_rate": _ratio(
                wait_actions + blocked_moves, active_steps
            ),
            "pooled_switcher_model_call_rate": _ratio(model_calls, active_steps),
            "pooled_switcher_choice_rate": _ratio(switcher_choices, active_steps),
            "pooled_aoreplan_wait_bypass_rate": _ratio(wait_bypasses, active_steps),
        }

    report = {
        "validated": True,
        "schema": "srslm_wait_ablation_exact960_v1",
        "algorithm": args.algorithm,
        "hybrid_mode": EXPECTED_MODES[args.algorithm],
        "result_path": str(result_path),
        "result_sha256": sha256(result_path),
        "source_manifest_sha256": sha256(before),
        "code_snapshot_sha256": sha256(code_before),
        "artifact_snapshot_sha256": sha256(artifact_before),
        "map_list_sha256": EXPECTED_MAP_SHA256,
        "rows": EXPECTED_ROWS,
        "populations": list(EXPECTED_POPULATIONS),
        "seeds": list(EXPECTED_SEEDS),
        "collision_system": "block_both",
        "on_target": "restart",
        "max_steps": 512,
        "obs_radius": 5,
        "workers": args.expected_workers,
        "switcher_checkpoint_sha256": expected_checkpoint,
        "switcher_policy_model_sha256": args.expected_switcher_model_sha256,
        "default_reference_policy_model_sha256": (
            args.reference_switcher_model_sha256
        ),
        "frozen_caar_checkpoint_sha256": EXPECTED_CAAR_CHECKPOINT_SHA256,
        "switcher_training_provenance": training_provenance,
        "result_journal": journal,
        "overall": summary(),
        "by_population": {
            str(population): summary(population)
            for population in EXPECTED_POPULATIONS
        },
    }
    atomic_json(output_dir / "VALIDATION.json", report)
    (output_dir / "STATUS").write_text("COMPLETE\n", encoding="utf-8")
    (output_dir / "COMPLETE").touch(exist_ok=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
