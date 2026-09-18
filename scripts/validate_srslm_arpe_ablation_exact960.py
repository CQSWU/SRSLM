#!/usr/bin/env python3
"""Strict validator for the three final CAAR/SRSLM exact960 deployments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

from agents.arpe import ArpeCandidateArtifact
from run_experiments import validate_final_srslm_ablation_stats
from scripts.switcher_nowait_artifact_contract import verify_validation


ALGORITHMS = (
    "SRSLM-NoWait",
    "SRSLM-OnlyWait",
)
LEARNED = frozenset(("SRSLM-NoWait",))
POPULATIONS = (100, 200, 300, 400, 500, 600)
SEEDS = (0, 42, 123, 2024, 3407)
ROWS = 32 * len(POPULATIONS) * len(SEEDS)
MAP_SHA256 = "da5c3d4cbd4cbdc8ce9f6b271ca258d4e7b69d6aa76524c6d17201718efb02f0"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"JSON artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def finite(value: Any, location: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        require(math.isfinite(float(value)), f"Non-finite value at {location}")
    elif isinstance(value, dict):
        for key, child in value.items():
            finite(child, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            finite(child, f"{location}[{index}]")


def map_names(path: Path) -> tuple[str, ...]:
    require(path.is_file(), f"Map list is missing: {path}")
    require(sha256(path) == MAP_SHA256, "Canonical map-list SHA256 differs")
    names = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        require(":" in line, f"Malformed map-list line: {raw!r}")
        names.append(line.split(":", 1)[0].strip())
    require(len(names) == len(set(names)) == 32, "Expected 32 unique maps")
    return tuple(names)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--map-list", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--expected-workers", type=int, required=True)
    parser.add_argument("--switcher-weights", type=Path)
    parser.add_argument("--switcher-log", type=Path)
    parser.add_argument("--switcher-validation", type=Path)
    args = parser.parse_args()

    project = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    result_path = args.input.resolve()
    require(output_dir.is_dir(), f"Output directory is missing: {output_dir}")
    require(result_path.parent == output_dir, "Result must be directly in output-dir")
    require(not (output_dir / "COMPLETE").exists(), "COMPLETE existed before validation")
    require(args.expected_workers > 0, "Expected worker count must be positive")

    before = output_dir / "source_before.sha256"
    after = output_dir / "source_after.sha256"
    require(before.is_file() and after.is_file(), "Source manifests are incomplete")
    require(before.read_bytes() == after.read_bytes(), "Evaluation inputs changed")

    declaration = load_json(args.candidate_manifest.resolve())
    candidate = ArpeCandidateArtifact.from_mapping(declaration, project)
    candidate.verify_files()
    candidate_hashes = {
        "checkpoint_sha256": candidate.checkpoint_sha256,
        "config_sha256": candidate.config_sha256,
        "base_checkpoint_sha256": candidate.base_checkpoint_sha256,
        "base_config_sha256": candidate.base_config_sha256,
    }

    switcher_checkpoint_sha256 = None
    switcher_config_sha256 = None
    switcher_validation_sha256 = None
    if args.algorithm in LEARNED:
        require(
            all(
                value is not None
                for value in (
                    args.switcher_weights,
                    args.switcher_log,
                    args.switcher_validation,
                )
            ),
            "Learned deployment requires the NoWait training artifacts",
        )
        certificate = verify_validation(
            args.switcher_validation.resolve(),
            weights_dir=args.switcher_weights.resolve(),
            log_dir=args.switcher_log.resolve(),
            project_root=project,
            expected_experiment="SRSLM-NoWait-CAAR-100M",
            expected_target_frames=100_000_000,
        )
        switcher_checkpoint_sha256 = certificate["terminal_checkpoint"][
            "checkpoint_sha256"
        ]
        switcher_config_sha256 = certificate["saved_config"]["sha256"]
        switcher_validation_sha256 = sha256(args.switcher_validation.resolve())
        saved_candidate = certificate["saved_config"]["candidate_policy"]
        for key, digest in candidate_hashes.items():
            require(saved_candidate.get(key) == digest, f"Training candidate {key} differs")
    else:
        require(
            args.switcher_weights is None
            and args.switcher_log is None
            and args.switcher_validation is None,
            "OnlyWait must not receive Switcher artifacts",
        )

    maps = map_names(args.map_list.resolve())
    expected = {
        (args.algorithm, map_name, population, seed)
        for map_name in maps
        for population in POPULATIONS
        for seed in SEEDS
    }
    payload = load_json(result_path)
    metadata = payload.get("metadata")
    rows = payload.get("results")
    require(isinstance(metadata, dict), "Result metadata is missing")
    require(isinstance(rows, list), "Result rows are missing")
    require(metadata.get("algorithms") == [args.algorithm], "Metadata algorithm differs")
    require(metadata.get("agent_counts") == list(POPULATIONS), "Population grid differs")
    require(metadata.get("seeds") == list(SEEDS), "Seed grid differs")
    require(tuple((metadata.get("maps") or {}).keys()) == maps, "Map grid differs")
    require(metadata.get("collision_system") == "block_both", "Collision mode differs")
    require(metadata.get("on_target") == "restart", "on_target differs")
    require(int(metadata.get("max_steps", -1)) == 512, "Horizon differs")
    require(int(metadata.get("obs_radius", -1)) == 5, "Observation radius differs")
    require(int(metadata.get("workers", -1)) == args.expected_workers, "Worker count differs")
    require(metadata.get("map_list_sha256") == MAP_SHA256, "Map hash differs")
    require(metadata.get("cache_algorithms_requested") is True, "Caching was not requested")
    require(
        (metadata.get("cache_algorithms_effective_by_algorithm") or {}).get(
            args.algorithm
        )
        is False,
        "Final SRSLM deployment was not episode-fresh",
    )

    require(len(rows) == ROWS, f"Expected {ROWS} rows, found {len(rows)}")
    actual = set()
    for index, row in enumerate(rows):
        label = f"row[{index}]"
        require(isinstance(row, dict), f"{label}: row is not an object")
        require(row.get("error") in (None, "", False), f"{label}: episode error")
        finite(row, label)
        key = (
            row.get("algorithm"),
            row.get("map_name"),
            int(row.get("num_agents", -1)),
            int(row.get("seed", -1)),
        )
        require(key in expected, f"{label}: tuple outside exact grid: {key}")
        require(key not in actual, f"Duplicate tuple: {key}")
        actual.add(key)
        require(int(row.get("max_steps", -1)) == 512, f"{label}: horizon differs")
        require(row.get("on_target") == "restart", f"{label}: on_target differs")
        require(int(row.get("total_experiments", -1)) == ROWS, f"{label}: total differs")
        for metric in ("avg_throughput", "congestion_rate"):
            value = row.get(metric)
            require(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value)),
                f"{label}: invalid {metric}",
            )
        validate_final_srslm_ablation_stats(args.algorithm, row)
        artifact = (row.get("candidate_provenance") or {}).get("candidate") or {}
        for hash_key, digest in candidate_hashes.items():
            require(artifact.get(hash_key) == digest, f"{label}: candidate {hash_key} differs")
        if args.algorithm in LEARNED:
            require(
                row.get("switcher_checkpoint_sha256") == switcher_checkpoint_sha256,
                f"{label}: Switcher checkpoint differs",
            )
            require(
                row.get("switcher_config_sha256") == switcher_config_sha256,
                f"{label}: Switcher config differs",
            )
    require(actual == expected, "Rows do not cover the exact960 grid")

    def summarize(population: int | None = None) -> dict[str, Any]:
        selected = [
            row for row in rows
            if population is None or int(row["num_agents"]) == population
        ]
        return {
            "episodes": len(selected),
            "mean_throughput": statistics.fmean(
                float(row["avg_throughput"]) for row in selected
            ),
            "mean_congestion_rate": statistics.fmean(
                float(row["congestion_rate"]) for row in selected
            ),
        }

    report = {
        "schema": "srslm_caar_ablation_exact960_validation_v1",
        "validated": True,
        "algorithm": args.algorithm,
        "result_path": str(result_path),
        "result_sha256": sha256(result_path),
        "source_manifest_sha256": sha256(before),
        "candidate_manifest_path": str(args.candidate_manifest.resolve()),
        "candidate_manifest_sha256": sha256(args.candidate_manifest.resolve()),
        "candidate_hashes": candidate_hashes,
        "switcher_checkpoint_sha256": switcher_checkpoint_sha256,
        "switcher_config_sha256": switcher_config_sha256,
        "switcher_validation_sha256": switcher_validation_sha256,
        "rows": ROWS,
        "maps": 32,
        "populations": list(POPULATIONS),
        "seeds": list(SEEDS),
        "collision_system": "block_both",
        "on_target": "restart",
        "max_steps": 512,
        "obs_radius": 5,
        "workers": args.expected_workers,
        "overall": summarize(),
        "by_population": {
            str(population): summarize(population) for population in POPULATIONS
        },
    }
    atomic_json(output_dir / "VALIDATION.json", report)
    (output_dir / "STATUS").write_text("COMPLETE\n", encoding="utf-8")
    (output_dir / "COMPLETE").touch(exist_ok=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

