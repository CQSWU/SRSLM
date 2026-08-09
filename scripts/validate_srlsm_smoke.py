#!/usr/bin/env python3
"""Fail-closed validation for SRSLM smoke artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from policy_estimation.caar_ao_rollout import collection_implementation_identity


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def latest_checkpoint(weights: Path) -> Path:
    checkpoints = sorted((weights / "checkpoint_p0").glob("checkpoint_*.pth"))
    if not checkpoints:
        raise RuntimeError(f"No CAAR checkpoint under {weights}")
    return checkpoints[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--caar-weights", required=True, type=Path)
    parser.add_argument("--smoke", required=True, type=Path)
    args = parser.parse_args()

    root = args.project_root.resolve()
    weights = args.weights.resolve()
    caar_weights = args.caar_weights.resolve()
    manifest_path = weights / "manifest.json"
    caar_estimator = weights / "caar_estimator.pth"
    ao_estimator = weights / "ao_estimator.pth"
    caar_config = caar_weights / "config.json"
    caar_checkpoint = latest_checkpoint(caar_weights)
    required = (
        manifest_path,
        caar_estimator,
        ao_estimator,
        caar_config,
        caar_checkpoint,
        args.smoke,
    )
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty required artifact: {path}")

    expected = {
        "caar_config_sha256": sha256_file(caar_config),
        "caar_checkpoint_sha256": sha256_file(caar_checkpoint),
        "caar_estimator_checkpoint_sha256": sha256_file(caar_estimator),
        "ao_estimator_checkpoint_sha256": sha256_file(ao_estimator),
        "collection_implementation_sha256": collection_implementation_identity(
            root, required=True
        )["collection_implementation_sha256"],
    }

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = manifest.get("behavior_contract", {})
    source_identity = manifest.get("source_policy_identity", {})
    if manifest.get("deployable") is not True:
        raise RuntimeError("Estimator training manifest is not deployable.")
    if set(manifest.get("branches", [])) != {"caar", "ao_safe"}:
        raise RuntimeError("Estimator manifest does not contain both branches.")
    for field in ("caar_config_sha256", "caar_checkpoint_sha256"):
        if contract.get(field) != expected[field] or source_identity.get(field) != expected[field]:
            raise RuntimeError(f"Estimator manifest does not match deployed {field}.")
    smoke = json.loads(args.smoke.read_text(encoding="utf-8"))
    rows = smoke.get("results", [])
    if len(rows) != 1:
        raise RuntimeError(f"Smoke result count is {len(rows)}, expected 1.")
    row = rows[0]
    if row.get("error"):
        raise RuntimeError(f"Smoke row failed: {row['error']}")
    if row.get("algorithm") != "SRSLM":
        raise RuntimeError(f"Unexpected smoke algorithm: {row.get('algorithm')}")
    if row.get("reverse_caar_override_enabled") is not True:
        raise RuntimeError("Reverse-to-CAAR rule was not enabled.")
    if row.get("switch_constraint") != "reverse_to_caar_current_step":
        raise RuntimeError("The per-step reverse fallback rule is not active.")
    if int(row.get("reverse_ao_executed_count", -1)) != 0:
        raise RuntimeError("A reverse AO action escaped the hard rule.")
    if int(row.get("value_comparison_count", 0)) <= 0:
        raise RuntimeError("Value estimators were not exercised in smoke run.")

    integrity = smoke.get("metadata", {}).get("integrity", {})
    for field in (
        "caar_config_sha256",
        "caar_checkpoint_sha256",
        "caar_estimator_checkpoint_sha256",
        "ao_estimator_checkpoint_sha256",
    ):
        if integrity.get(field) != expected[field]:
            raise RuntimeError(f"Smoke integrity mismatch for {field}.")

    print(
        json.dumps(
            {
                "status": "passed",
                **expected,
                "weights_manifest": str(manifest_path),
                "runtime_script": str(root / "run_experiments.py"),
                "value_comparison_count": row["value_comparison_count"],
                "executed_ao_rate": row.get("executed_ao_rate"),
            },
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
