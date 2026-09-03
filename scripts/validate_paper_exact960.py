#!/usr/bin/env python3
"""Strict terminal validator for the paper's lifelong exact-960 runs.

The validator is deliberately independent of the evaluation launcher.  It
accepts only the fixed 32-map, six-population, five-seed protocol and writes
the terminal markers only after every row and both source manifests pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any


EXPECTED_MAP_SHA256 = (
    "da5c3d4cbd4cbdc8ce9f6b271ca258d4e7b69d6aa76524c6d17201718efb02f0"
)
EXPECTED_POPULATIONS = (100, 200, 300, 400, 500, 600)
EXPECTED_SEEDS = (0, 42, 123, 2024, 3407)
EXPECTED_ROWS = 32 * len(EXPECTED_POPULATIONS) * len(EXPECTED_SEEDS)
EXPECTED_REVERSE_METRIC = "previous_timestep_position_target_segment_v3"
EXPECTED_SRSLM_MODE = "aoreplan_wait_bypass_switcher_v3"
EXPECTED_SWITCHER_SCHEMA = "srslm_switcher_state_v3"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def map_names(path: Path) -> tuple[str, ...]:
    require(path.is_file(), f"Map list is missing: {path}")
    require(sha256(path) == EXPECTED_MAP_SHA256, "Fixed map-list SHA256 differs")
    names = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        require(":" in line, f"Malformed map-list line: {raw_line!r}")
        names.append(line.split(":", 1)[0].strip())
    require(len(names) == 32, f"Expected 32 maps, found {len(names)}")
    require(len(set(names)) == 32, "Map list contains duplicate names")
    return tuple(names)


def require_finite(value: Any, location: str) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        require(math.isfinite(float(value)), f"Non-finite number at {location}")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            require_finite(child, f"{location}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            require_finite(child, f"{location}[{index}]")


def require_probability(row: dict[str, Any], key: str, label: str) -> None:
    value = row.get(key)
    require(value is not None, f"{label}: missing {key}")
    require_finite(value, f"{label}.{key}")
    require(0.0 <= float(value) <= 1.0, f"{label}: {key} is not a probability")


def validate_aoreplan_row(row: dict[str, Any], label: str) -> None:
    require(
        row.get("reverse_metric_version") == EXPECTED_REVERSE_METRIC,
        f"{label}: reverse metric is not the previous-timestep definition",
    )
    for key in (
        "reverse_action_count",
        "reverse_action_denominator",
        "static_astar_query_count",
        "static_astar_query_denominator",
        "no_path_fallback_count",
    ):
        value = row.get(key)
        require(isinstance(value, int) and not isinstance(value, bool), f"{label}: invalid {key}")
        require(value >= 0, f"{label}: negative {key}")
    require_probability(row, "reverse_action_rate", label)
    require_probability(row, "static_astar_query_rate", label)
    require(
        int(row["reverse_action_count"]) <= int(row["reverse_action_denominator"]),
        f"{label}: reverse count exceeds its denominator",
    )
    require(
        int(row["static_astar_query_count"])
        <= int(row["static_astar_query_denominator"]),
        f"{label}: static-A* count exceeds its denominator",
    )


def validate_srslm_row(
    row: dict[str, Any],
    label: str,
    expected_switcher_sha256: str,
) -> None:
    require(row.get("hybrid_mode") == EXPECTED_SRSLM_MODE, f"{label}: wrong SRSLM mode")
    require(row.get("switch_pair") == ["CAAR", "AORePlan"], f"{label}: wrong branches")
    require(row.get("switcher_training") == "PPO", f"{label}: Switcher is not PPO-trained")
    require(row.get("value_predictor_loaded") is False, f"{label}: retired value head loaded")
    require(
        row.get("switcher_feature_schema") == EXPECTED_SWITCHER_SCHEMA,
        f"{label}: wrong Switcher feature schema",
    )
    require(
        row.get("selector_kind") == "ppo_two_branch_categorical",
        f"{label}: wrong selector kind",
    )
    require(
        row.get("switcher_decision_scope") == "aoreplan_nonwait_only",
        f"{label}: waits reached the Switcher",
    )
    require(
        row.get("joint_conflict_prediction_enabled") is False,
        f"{label}: retired joint-conflict prediction is enabled",
    )
    require(row.get("switcher_stochastic") is True, f"{label}: Switcher is not stochastic")
    require(
        row.get("switcher_checkpoint_sha256") == expected_switcher_sha256,
        f"{label}: unexpected Switcher checkpoint",
    )

    integer_keys = (
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
    values: dict[str, int] = {}
    for key in integer_keys:
        value = row.get(key)
        require(isinstance(value, int) and not isinstance(value, bool), f"{label}: invalid {key}")
        require(value >= 0, f"{label}: negative {key}")
        values[key] = int(value)

    total = values["total_action_count"]
    choices = values["switcher_choice_count"]
    bypasses = values["aoreplan_wait_bypass_count"]
    selected_ao = values["selected_ao_count"]
    require(choices + bypasses == total, f"{label}: Switcher/bypass accounting differs")
    require(
        values["executed_ao_count"] + values["executed_caar_count"] == total,
        f"{label}: executed branch accounting differs",
    )
    require(selected_ao == values["executed_ao_count"], f"{label}: selected AO was not executed")
    require(selected_ao <= choices, f"{label}: AO selections exceed Switcher choices")
    require(values["switcher_model_choice_count"] == choices, f"{label}: model/router counts differ")
    require(
        values["switcher_model_selected_ao_count"] == selected_ao,
        f"{label}: model/router AO counts differ",
    )
    require(values["aoreplan_commit_count"] <= total, f"{label}: too many AORePlan commits")
    require(
        values["branch_action_agreement_count"] <= total,
        f"{label}: too many branch agreements",
    )
    for key in (
        "switcher_choice_rate",
        "selected_ao_rate",
        "executed_ao_rate",
        "aoreplan_wait_bypass_rate",
        "branch_action_agreement_rate",
        "switcher_sampled_ao_rate",
        "switcher_ao_probability_mean",
        "switcher_ao_probability_p05",
        "switcher_ao_probability_p95",
    ):
        require_probability(row, key, label)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("AORePlan", "SRSLM"), required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--map-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-workers", type=int, required=True)
    parser.add_argument("--expected-switcher-checkpoint-sha256")
    args = parser.parse_args()

    result_path = args.input.resolve()
    output_dir = args.output_dir.resolve()
    map_list = args.map_list.resolve()
    require(args.expected_workers > 0, "Expected workers must be positive")
    require(output_dir.is_dir(), f"Output directory is missing: {output_dir}")
    require(result_path.is_file(), f"Result JSON is missing: {result_path}")
    require(result_path.parent == output_dir, "Result JSON must be directly inside output-dir")
    require(not (output_dir / "COMPLETE").exists(), "COMPLETE already exists before validation")

    before = output_dir / "source_before.sha256"
    after = output_dir / "source_after.sha256"
    require(before.is_file() and after.is_file(), "Source manifests are incomplete")
    require(before.read_bytes() == after.read_bytes(), "Tracked inputs changed during evaluation")

    maps = map_names(map_list)
    expected = {
        (args.algorithm, map_name, population, seed)
        for map_name in maps
        for population in EXPECTED_POPULATIONS
        for seed in EXPECTED_SEEDS
    }
    if args.algorithm == "SRSLM":
        digest = args.expected_switcher_checkpoint_sha256 or ""
        require(
            len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest),
            "SRSLM validation requires an expected Switcher checkpoint SHA256",
        )
    else:
        require(
            args.expected_switcher_checkpoint_sha256 is None,
            "AORePlan validation must not receive a Switcher checkpoint",
        )
        digest = ""

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), "Result payload is not an object")
    metadata = payload.get("metadata")
    rows = payload.get("results")
    require(isinstance(metadata, dict), "Result metadata is missing")
    require(isinstance(rows, list), "Result rows are missing")

    require(metadata.get("algorithms") == [args.algorithm], "Metadata algorithm differs")
    require(metadata.get("agent_counts") == list(EXPECTED_POPULATIONS), "Population grid differs")
    require(metadata.get("seeds") == list(EXPECTED_SEEDS), "Seed grid differs")
    metadata_maps = metadata.get("maps")
    require(isinstance(metadata_maps, dict), "Metadata maps are missing")
    require(tuple(metadata_maps.keys()) == maps, "Metadata map order or names differ")
    require(metadata.get("collision_system") == "block_both", "Collision mode differs")
    require(metadata.get("on_target") == "restart", "on_target differs")
    require(int(metadata.get("max_steps", -1)) == 512, "Horizon differs")
    require(int(metadata.get("obs_radius", -1)) == 5, "Observation radius differs")
    require(int(metadata.get("workers", -1)) == args.expected_workers, "Worker count differs")
    require(metadata.get("map_list_sha256") == EXPECTED_MAP_SHA256, "Metadata map hash differs")
    require(metadata.get("cache_algorithms_requested") is True, "Algorithm caching was not requested")
    effective = metadata.get("cache_algorithms_effective_by_algorithm", {})
    exceptions = metadata.get("cache_algorithms_exceptions", {})
    if args.algorithm == "AORePlan":
        require(effective.get(args.algorithm) is True, "AORePlan cache contract differs")
        require(args.algorithm not in exceptions, "AORePlan has an unexpected cache exception")
        reverse = metadata.get("reverse_metric", {})
        require(reverse.get("version") == EXPECTED_REVERSE_METRIC, "Metadata reverse definition differs")
    else:
        require(effective.get(args.algorithm) is False, "SRSLM reused policy state across episodes")
        require(args.algorithm in exceptions, "SRSLM episode-fresh cache exception is missing")
        require(metadata.get("hybrid_mode") == EXPECTED_SRSLM_MODE, "Metadata SRSLM mode differs")
        integrity = metadata.get("integrity")
        require(isinstance(integrity, dict), "SRSLM integrity metadata is missing")
        require(integrity.get("hybrid_mode") == EXPECTED_SRSLM_MODE, "Integrity mode differs")
        require(
            integrity.get("switcher_checkpoint_sha256") == digest,
            "Integrity metadata binds another Switcher checkpoint",
        )
        require(integrity.get("map_list_sha256") == EXPECTED_MAP_SHA256, "Integrity map hash differs")

    require(len(rows) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows, found {len(rows)}")
    actual = set()
    for index, row in enumerate(rows):
        label = f"row[{index}]"
        require(isinstance(row, dict), f"{label}: row is not an object")
        error = row.get("error")
        require(error in (None, "", False), f"{label}: episode contains an error")
        require_finite(row, label)
        require(row.get("algorithm") == args.algorithm, f"{label}: algorithm differs")
        require(row.get("map_name") in maps, f"{label}: map differs")
        require(int(row.get("num_agents", -1)) in EXPECTED_POPULATIONS, f"{label}: population differs")
        require(int(row.get("seed", -1)) in EXPECTED_SEEDS, f"{label}: seed differs")
        require(int(row.get("max_steps", -1)) == 512, f"{label}: horizon differs")
        require(row.get("on_target") == "restart", f"{label}: on_target differs")
        require(int(row.get("total_experiments", -1)) == EXPECTED_ROWS, f"{label}: total_experiments differs")
        for metric in ("avg_throughput", "congestion_rate"):
            value = row.get(metric)
            require(
                isinstance(value, (int, float)) and not isinstance(value, bool),
                f"{label}: missing or non-numeric {metric}",
            )
            require_finite(value, f"{label}.{metric}")
        key = (
            str(row["algorithm"]),
            str(row["map_name"]),
            int(row["num_agents"]),
            int(row["seed"]),
        )
        require(key not in actual, f"Duplicate tuple: {key}")
        actual.add(key)
        if args.algorithm == "AORePlan":
            validate_aoreplan_row(row, label)
        else:
            validate_srslm_row(row, label, digest)

    require(actual == expected, "Result tuples do not match the exact-960 grid")

    def summary(population: int | None = None) -> dict[str, Any]:
        selected = [
            row
            for row in rows
            if population is None or int(row["num_agents"]) == population
        ]
        return {
            "episodes": len(selected),
            "mean_throughput": statistics.fmean(float(row["avg_throughput"]) for row in selected),
            "mean_congestion_rate": statistics.fmean(float(row["congestion_rate"]) for row in selected),
        }

    report = {
        "validated": True,
        "algorithm": args.algorithm,
        "result_path": str(result_path),
        "result_sha256": sha256(result_path),
        "source_manifest_sha256": sha256(before),
        "map_list_path": str(map_list),
        "map_list_sha256": EXPECTED_MAP_SHA256,
        "rows": EXPECTED_ROWS,
        "maps": 32,
        "populations": list(EXPECTED_POPULATIONS),
        "seeds": list(EXPECTED_SEEDS),
        "collision_system": "block_both",
        "on_target": "restart",
        "max_steps": 512,
        "obs_radius": 5,
        "workers": args.expected_workers,
        "switcher_checkpoint_sha256": digest or None,
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
