#!/usr/bin/env python3
"""Create and verify auditable Switcher training-artifact certificates.

The regular checkpoint filename is only a convenience.  Formal gates in this
module load the checkpoint and use its embedded ``env_steps`` value as the
authoritative frame count.  They also hash only the policy-model tensors so a
NoWaitDetect policy cannot be mistaken for the default policy merely because
the two models share the same architecture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping

import torch

try:
    from scripts.switcher_checkpoint_identity import policy_model_sha256
except ModuleNotFoundError:  # Direct ``python scripts/...py`` execution.
    from switcher_checkpoint_identity import policy_model_sha256


SCHEMA = "switcher_training_artifact_v2"
RUNTIME_SMOKE_SCHEMA = "switcher_runtime_smoke_v1"
EXPECTED_CAAR_CHECKPOINT_SHA256 = (
    "fb302e14543c6138ae2375f1fa6617198dc2d0e106a1d1c0a2d5b298238b4a3f"
)
MIN_FORMAL_FRAMES = 500_000_000
_REGULAR_CHECKPOINT = re.compile(
    r"^checkpoint_(?P<train_step>[0-9]+)_(?P<env_steps>[0-9]+)\.pth$"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(checkpoint, Mapping), f"Checkpoint is not a mapping: {path}")
    return checkpoint


def checkpoint_identity(path: Path | str) -> dict[str, Any]:
    path = Path(path).resolve()
    require(path.is_file() and path.stat().st_size > 0, f"Checkpoint is missing: {path}")
    match = _REGULAR_CHECKPOINT.fullmatch(path.name)
    require(match is not None, f"Not a regular checkpoint filename: {path.name}")
    checkpoint = _load_checkpoint(path)
    embedded_frames = checkpoint.get("env_steps")
    embedded_train_step = checkpoint.get("train_step")
    require(
        isinstance(embedded_frames, int) and not isinstance(embedded_frames, bool),
        f"Checkpoint has no integer env_steps: {path}",
    )
    require(
        isinstance(embedded_train_step, int)
        and not isinstance(embedded_train_step, bool),
        f"Checkpoint has no integer train_step: {path}",
    )
    filename_frames = int(match.group("env_steps"))
    filename_train_step = int(match.group("train_step"))
    require(
        int(embedded_frames) == filename_frames,
        f"Checkpoint filename/env_steps mismatch: {path}",
    )
    require(
        int(embedded_train_step) == filename_train_step,
        f"Checkpoint filename/train_step mismatch: {path}",
    )
    require(embedded_frames >= 0 and embedded_train_step >= 0, "Negative checkpoint counters")
    return {
        "path": str(path),
        "filename": path.name,
        "env_steps": int(embedded_frames),
        "train_step": int(embedded_train_step),
        "checkpoint_sha256": sha256_file(path),
        "policy_model_sha256": policy_model_sha256(path),
    }


def latest_regular_checkpoint(weights_dir: Path | str) -> dict[str, Any]:
    weights_dir = Path(weights_dir).resolve()
    checkpoint_dir = weights_dir / "checkpoint_p0"
    require(checkpoint_dir.is_dir(), f"Checkpoint directory is missing: {checkpoint_dir}")
    identities = []
    errors = []
    for path in sorted(checkpoint_dir.glob("checkpoint_*.pth")):
        try:
            identities.append(checkpoint_identity(path))
        except RuntimeError as error:
            errors.append(str(error))
    require(identities, f"No valid regular checkpoint in {checkpoint_dir}; errors={errors[:2]}")
    identities.sort(key=lambda item: (item["env_steps"], item["train_step"], item["path"]))
    return identities[-1]


def _parse_sha256_manifest(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Source manifest is missing: {path}")
    entries = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.rstrip("\r\n")
        require(line.strip(), f"Blank source-manifest line {line_number}: {path}")
        parts = line.split(None, 1)
        require(len(parts) == 2, f"Malformed source-manifest line {line_number}: {path}")
        digest, item_path = parts
        item_path = item_path.lstrip(" *")
        require(
            len(digest) == 64 and all(char in "0123456789abcdef" for char in digest),
            f"Invalid SHA256 on source-manifest line {line_number}: {path}",
        )
        require(item_path, f"Missing path on source-manifest line {line_number}: {path}")
        entries.append({"sha256": digest, "path": item_path})
    require(entries, f"Source manifest is empty: {path}")
    return entries


def source_manifest_identity(log_dir: Path | str) -> dict[str, Any]:
    log_dir = Path(log_dir).resolve()
    before = log_dir / "source_before.sha256"
    after = log_dir / "source_after.sha256"
    require(before.is_file() and after.is_file(), "Training source manifests are incomplete")
    require(before.read_bytes() == after.read_bytes(), "Training inputs changed during the run")
    diff = log_dir / "source_hash_diff.txt"
    require(not diff.exists() or diff.stat().st_size == 0, "Training source diff is non-empty")
    entries = _parse_sha256_manifest(before)
    require(
        any(
            item["sha256"] == EXPECTED_CAAR_CHECKPOINT_SHA256
            and "CAAR" in item["path"]
            and item["path"].endswith(".pth")
            for item in entries
        ),
        "Training source manifest does not bind the frozen CAAR checkpoint",
    )
    return {
        "before_path": str(before),
        "after_path": str(after),
        "sha256": sha256_file(before),
        "entries": len(entries),
        "frozen_caar_checkpoint_sha256": EXPECTED_CAAR_CHECKPOINT_SHA256,
    }


def _validate_config(
    config_path: Path,
    *,
    expected_experiment: str,
    expected_encoder: str,
    expected_environment: str,
    expected_target_frames: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    require(isinstance(config, dict), "Switcher config is not a JSON object")
    environment = config.get("full_config", {}).get("environment", {})
    grid = environment.get("grid_config", {})
    checks = {
        "experiment": config.get("experiment") == expected_experiment,
        "encoder": config.get("encoder_custom") == expected_encoder,
        "environment": environment.get("name") == expected_environment,
        "target": int(config.get("train_for_env_steps", -1)) == expected_target_frames,
        "feed_forward": config.get("use_rnn") is False,
        "workers": int(config.get("num_workers", -1)) == 12,
        "schema": environment.get("switcher_feature_schema") == "srslm_switcher_state_v3",
        "collision": grid.get("collision_system") == "block_both",
        "on_target": grid.get("on_target") == "restart",
        "horizon": int(grid.get("max_episode_steps", -1)) == 512,
        "radius": int(grid.get("obs_radius", -1)) == 5,
        "population": int(grid.get("num_agents", -1)) == 200,
    }
    failed = [name for name, passed in checks.items() if not passed]
    require(not failed, "Switcher config failed: " + ", ".join(failed))
    return config, environment


def _runtime_smoke_identity(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = Path(path).resolve()
    require(path.is_file(), f"Runtime-smoke certificate is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("validated") is True, "Runtime smoke is not validated")
    require(payload.get("schema") == RUNTIME_SMOKE_SCHEMA, "Runtime smoke schema differs")
    result_path = Path(str(payload.get("result_path", "")))
    require(result_path.is_file(), "Runtime-smoke result payload is missing")
    require(
        sha256_file(result_path) == payload.get("result_sha256"),
        "Runtime-smoke result payload changed",
    )
    total = payload.get("total_action_count")
    choices = payload.get("switcher_choice_count")
    bypasses = payload.get("aoreplan_wait_bypass_count")
    require(isinstance(total, int) and total > 0, "Runtime smoke has no action evidence")
    require(isinstance(choices, int), "Runtime smoke has no choice evidence")
    if payload.get("algorithm") == "SRSLM-NoWaitDetect":
        require(choices == total, "NoWaitDetect smoke did not select on every state")
        require(bypasses == 0, "NoWaitDetect smoke bypassed wait states")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "algorithm": payload.get("algorithm"),
        "switcher_checkpoint_sha256": payload.get("switcher_checkpoint_sha256"),
    }


def build_training_validation(
    *,
    weights_dir: Path | str,
    log_dir: Path | str,
    checkpoint_path: Path | str | None,
    expected_experiment: str,
    expected_encoder: str,
    expected_environment: str,
    expected_contract: str,
    expected_target_frames: int,
    runtime_smoke_validation: Path | str | None = None,
    reference_checkpoint: Path | str | None = None,
) -> dict[str, Any]:
    weights_dir = Path(weights_dir).resolve()
    log_dir = Path(log_dir).resolve()
    config_path = weights_dir / "config.json"
    require(config_path.is_file(), f"Switcher config is missing: {config_path}")
    _validate_config(
        config_path,
        expected_experiment=expected_experiment,
        expected_encoder=expected_encoder,
        expected_environment=expected_environment,
        expected_target_frames=expected_target_frames,
    )
    checkpoint = (
        checkpoint_identity(checkpoint_path)
        if checkpoint_path is not None
        else latest_regular_checkpoint(weights_dir)
    )
    require(
        Path(checkpoint["path"]).parent == weights_dir / "checkpoint_p0",
        "Checkpoint is outside the requested Switcher weights directory",
    )
    require(
        checkpoint["env_steps"] >= expected_target_frames,
        "Embedded checkpoint env_steps are below the training target",
    )
    if expected_target_frames >= MIN_FORMAL_FRAMES:
        require(
            checkpoint["env_steps"] >= MIN_FORMAL_FRAMES,
            "Formal Switcher checkpoint has fewer than 500M embedded frames",
        )
    source = source_manifest_identity(log_dir)
    smoke = _runtime_smoke_identity(runtime_smoke_validation)
    if smoke is not None:
        require(
            smoke["switcher_checkpoint_sha256"] == checkpoint["checkpoint_sha256"],
            "Runtime smoke used a different Switcher checkpoint",
        )

    reference = None
    if reference_checkpoint is not None:
        reference = checkpoint_identity(reference_checkpoint)
        require(
            reference["env_steps"] >= MIN_FORMAL_FRAMES,
            "Reference default Switcher has fewer than 500M embedded frames",
        )
        require(
            checkpoint["policy_model_sha256"] != reference["policy_model_sha256"],
            "NoWaitDetect and default Switcher have identical policy-model tensors",
        )

    return {
        "validated": True,
        "schema": SCHEMA,
        "training_contract": expected_contract,
        "target_frames": int(expected_target_frames),
        "actual_checkpoint_frames": checkpoint["env_steps"],
        "actual_checkpoint_train_step": checkpoint["train_step"],
        "checkpoint_path": checkpoint["path"],
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "policy_model_sha256": checkpoint["policy_model_sha256"],
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "experiment": expected_experiment,
        "encoder_custom": expected_encoder,
        "environment": expected_environment,
        "workers": 12,
        "collision_system": "block_both",
        "on_target": "restart",
        "max_episode_steps": 512,
        "obs_radius": 5,
        "num_agents": 200,
        "frozen_caar_checkpoint_sha256": EXPECTED_CAAR_CHECKPOINT_SHA256,
        "source_manifest": source,
        "runtime_smoke": smoke,
        "distinct_from_default": (
            None
            if reference is None
            else {
                "checkpoint_path": reference["path"],
                "checkpoint_sha256": reference["checkpoint_sha256"],
                "actual_checkpoint_frames": reference["env_steps"],
                "policy_model_sha256": reference["policy_model_sha256"],
                "model_sha256_differs": True,
            }
        ),
    }


def verify_training_validation(
    validation_path: Path | str,
    *,
    weights_dir: Path | str,
    log_dir: Path | str,
    expected_experiment: str,
    expected_encoder: str,
    expected_environment: str,
    expected_contract: str,
    expected_target_frames: int = MIN_FORMAL_FRAMES,
    reference_checkpoint: Path | str | None = None,
) -> dict[str, Any]:
    validation_path = Path(validation_path).resolve()
    log_dir = Path(log_dir).resolve()
    require((log_dir / "COMPLETE").is_file(), "Training COMPLETE marker is missing")
    require(
        (log_dir / "STATUS").read_text(encoding="utf-8").strip() == "COMPLETE",
        "Training STATUS is not COMPLETE",
    )
    require(validation_path == log_dir / "VALIDATION.json", "Training validation path differs")
    require(validation_path.is_file(), "Training VALIDATION.json is missing")
    recorded = json.loads(validation_path.read_text(encoding="utf-8"))
    require(recorded.get("validated") is True, "Training validation is not validated")
    require(recorded.get("schema") == SCHEMA, "Training validation schema differs")
    runtime_smoke = recorded.get("runtime_smoke")
    rebuilt = build_training_validation(
        weights_dir=weights_dir,
        log_dir=log_dir,
        checkpoint_path=recorded.get("checkpoint_path"),
        expected_experiment=expected_experiment,
        expected_encoder=expected_encoder,
        expected_environment=expected_environment,
        expected_contract=expected_contract,
        expected_target_frames=expected_target_frames,
        runtime_smoke_validation=(
            runtime_smoke.get("path") if isinstance(runtime_smoke, dict) else None
        ),
        reference_checkpoint=reference_checkpoint,
    )
    require(recorded == rebuilt, "Training VALIDATION.json no longer matches the artifact")
    proof = dict(rebuilt)
    proof["validation_path"] = str(validation_path)
    proof["validation_sha256"] = sha256_file(validation_path)
    return proof


def validate_runtime_smoke(
    *,
    result_path: Path | str,
    algorithm: str,
    expected_switcher_checkpoint_sha256: str,
) -> dict[str, Any]:
    result_path = Path(result_path).resolve()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata")
    rows = payload.get("results")
    require(isinstance(metadata, dict) and isinstance(rows, list), "Malformed smoke result")
    require(metadata.get("algorithms") == [algorithm], "Smoke algorithm differs")
    require(len(rows) == 1, "Runtime smoke must contain exactly one episode")
    row = rows[0]
    require(row.get("error") in (None, "", False), "Runtime smoke contains an error")
    total = row.get("total_action_count")
    choices = row.get("switcher_choice_count")
    require(isinstance(total, int) and total > 0, "Runtime smoke has no actions")
    require(isinstance(choices, int), "Runtime smoke has no Switcher choice count")
    if algorithm == "SRSLM-NoWaitDetect":
        require(choices == total, "NoWaitDetect runtime smoke did not route every state")
        require(row.get("switcher_model_choice_count") == total, "NoWaitDetect model missed states")
        require(row.get("aoreplan_wait_bypass_count") == 0, "NoWaitDetect bypassed waits")
    else:
        bypasses = row.get("aoreplan_wait_bypass_count")
        require(isinstance(bypasses, int), "Default smoke has no bypass count")
        require(choices + bypasses == total, "Default smoke routing counts do not sum")
    require(
        row.get("switcher_checkpoint_sha256") == expected_switcher_checkpoint_sha256,
        "Runtime smoke used a different Switcher checkpoint",
    )
    integrity = metadata.get("integrity")
    require(isinstance(integrity, dict), "Runtime smoke integrity metadata is missing")
    require(
        integrity.get("switcher_checkpoint_sha256")
        == expected_switcher_checkpoint_sha256,
        "Runtime smoke integrity checkpoint differs",
    )
    require(
        integrity.get("caar_checkpoint_sha256") == EXPECTED_CAAR_CHECKPOINT_SHA256,
        "Runtime smoke did not use the frozen CAAR checkpoint",
    )
    return {
        "validated": True,
        "schema": RUNTIME_SMOKE_SCHEMA,
        "algorithm": algorithm,
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "switcher_checkpoint_sha256": expected_switcher_checkpoint_sha256,
        "frozen_caar_checkpoint_sha256": EXPECTED_CAAR_CHECKPOINT_SHA256,
        "total_action_count": total,
        "switcher_choice_count": choices,
        "aoreplan_wait_bypass_count": row.get("aoreplan_wait_bypass_count"),
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    checkpoint_parser = subparsers.add_parser("checkpoint")
    checkpoint_parser.add_argument("--weights-dir", type=Path, required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--weights-dir", type=Path, required=True)
    common.add_argument("--log-dir", type=Path, required=True)
    common.add_argument("--expected-experiment", required=True)
    common.add_argument("--expected-encoder", required=True)
    common.add_argument("--expected-environment", required=True)
    common.add_argument("--expected-contract", required=True)
    common.add_argument("--expected-target-frames", type=int, required=True)
    common.add_argument("--reference-checkpoint", type=Path)

    postflight = subparsers.add_parser("postflight", parents=[common])
    postflight.add_argument("--checkpoint", type=Path)
    postflight.add_argument("--runtime-smoke-validation", type=Path)
    postflight.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify", parents=[common])
    verify.add_argument("--validation", type=Path, required=True)

    smoke = subparsers.add_parser("runtime-smoke")
    smoke.add_argument("--result", type=Path, required=True)
    smoke.add_argument(
        "--algorithm", choices=("SRSLM-NoWaitDetect", "SRSLM"), required=True
    )
    smoke.add_argument("--expected-switcher-checkpoint-sha256", required=True)
    smoke.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "checkpoint":
        result = latest_regular_checkpoint(args.weights_dir)
    elif args.command == "postflight":
        result = build_training_validation(
            weights_dir=args.weights_dir,
            log_dir=args.log_dir,
            checkpoint_path=args.checkpoint,
            expected_experiment=args.expected_experiment,
            expected_encoder=args.expected_encoder,
            expected_environment=args.expected_environment,
            expected_contract=args.expected_contract,
            expected_target_frames=args.expected_target_frames,
            runtime_smoke_validation=args.runtime_smoke_validation,
            reference_checkpoint=args.reference_checkpoint,
        )
        atomic_json(args.output, result)
    elif args.command == "verify":
        result = verify_training_validation(
            args.validation,
            weights_dir=args.weights_dir,
            log_dir=args.log_dir,
            expected_experiment=args.expected_experiment,
            expected_encoder=args.expected_encoder,
            expected_environment=args.expected_environment,
            expected_contract=args.expected_contract,
            expected_target_frames=args.expected_target_frames,
            reference_checkpoint=args.reference_checkpoint,
        )
    else:
        result = validate_runtime_smoke(
            result_path=args.result,
            algorithm=args.algorithm,
            expected_switcher_checkpoint_sha256=(
                args.expected_switcher_checkpoint_sha256
            ),
        )
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
