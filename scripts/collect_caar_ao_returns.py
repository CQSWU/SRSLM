#!/usr/bin/env python3
"""Collect paired absolute-return datasets for CAAR and AO-safe.

The command writes one independent dataset below ``OUTPUT/caar`` and/or
``OUTPUT/ao_safe``. Both manifests embed the same canonical scenario-manifest
hash, making the shared initial map/agent/seed instances auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from policy_estimation.caar_ao_rollout import (
    AO_SAFE_LANE,
    CAAR_LANE,
    EpisodeSpec,
    RolloutJob,
    derive_episode_sample_seed,
    iter_collected_jobs,
    validate_paired_episode_samples,
)
from policy_estimation.dataset import NpzShardWriter, sha256_file


def _canonical_scenarios(scenarios: Sequence[EpisodeSpec]) -> str:
    payload = [
        {
            "scenario_id": scenario.scenario_id,
            "grid_config": dict(scenario.grid_config),
        }
        for scenario in scenarios
    ]
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def load_scenarios(path: str | Path) -> tuple[list[EpisodeSpec], str]:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(raw, Mapping):
        raw = raw.get("scenarios")
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            "Scenario manifest must be a non-empty list or contain "
            "a non-empty 'scenarios' list."
        )
    scenarios = [EpisodeSpec.from_mapping(value) for value in raw]
    ids = [scenario.scenario_id for scenario in scenarios]
    if len(ids) != len(set(ids)):
        raise ValueError("scenario_id values must be unique.")
    canonical = _canonical_scenarios(scenarios)
    return scenarios, hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_jobs(
    scenarios: Sequence[EpisodeSpec],
    branches: Sequence[str],
    args: argparse.Namespace,
) -> list[RolloutJob]:
    jobs = []
    # Scenario-major order keeps paired jobs adjacent while executor.map
    # preserves deterministic output ordering.
    for scenario in scenarios:
        for branch in branches:
            jobs.append(
                RolloutJob(
                    episode=scenario,
                    lane=branch,
                    gamma=args.gamma,
                    sample_fraction=args.sample_fraction,
                    sample_seed=derive_episode_sample_seed(
                        args.sampling_seed,
                        scenario.scenario_id,
                        branch,
                    ),
                    caar_path_to_weights=args.caar_weights,
                    caar_checkpoint_kind=args.caar_checkpoint_kind,
                    caar_device=args.caar_device,
                    plan_use_best_move=args.plan_use_best_move,
                    plan_max_steps=args.plan_max_steps,
                    torch_num_threads=args.torch_num_threads,
                )
            )
    return jobs


def collect(
    args: argparse.Namespace,
    *,
    collected_jobs_iterator=iter_collected_jobs,
) -> dict[str, object]:
    scenarios, scenario_digest = load_scenarios(args.scenario_manifest)
    branches = tuple(dict.fromkeys(args.branches))
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_manifest = Path(args.scenario_manifest).resolve()
    common_metadata = {
        "collector": "absolute_policy_mc_returns",
        "paired_scenarios": len(branches) == 2,
        "scenario_count": len(scenarios),
        "scenario_manifest_path": str(source_manifest),
        "scenario_manifest_file_sha256": sha256_file(source_manifest),
        "canonical_scenario_manifest_sha256": scenario_digest,
        "scenario_ids": [scenario.scenario_id for scenario in scenarios],
        "gamma": float(args.gamma),
        "sample_fraction": float(args.sample_fraction),
        "sampling_seed": int(args.sampling_seed),
        "sampling": "fixed_size_without_replacement_per_episode",
        "caar_path_to_weights": str(args.caar_weights),
        "caar_checkpoint_kind": str(args.caar_checkpoint_kind),
        "plan_use_best_move": bool(args.plan_use_best_move),
        "plan_max_steps": int(args.plan_max_steps),
        "torch_num_threads": int(args.torch_num_threads),
    }
    writers = {
        branch: NpzShardWriter(
            output_root / branch,
            max_rows_per_shard=args.shard_rows,
            dataset_metadata={**common_metadata, "branch": branch},
        )
        for branch in branches
    }
    jobs = build_jobs(scenarios, branches, args)
    paired_collection = set(branches) == {CAAR_LANE, AO_SAFE_LANE}
    expected_scenarios = {scenario.scenario_id for scenario in scenarios}
    pending_pairs = {}
    completed = False
    manifests = {}
    try:
        for samples in collected_jobs_iterator(
            jobs,
            max_workers=args.workers,
        ):
            branch = str(samples.metadata.get("branch"))
            if branch not in writers:
                raise RuntimeError(f"Worker returned unexpected branch {branch!r}.")
            scenario_id = str(samples.metadata.get("scenario_id"))
            if scenario_id not in expected_scenarios:
                raise RuntimeError(
                    f"Worker returned unexpected scenario {scenario_id!r}."
                )
            if paired_collection:
                pair = pending_pairs.setdefault(scenario_id, {})
                if branch in pair:
                    raise RuntimeError(
                        f"Duplicate {branch!r} result for scenario {scenario_id!r}."
                    )
                pair[branch] = samples
                if set(pair) == {CAAR_LANE, AO_SAFE_LANE}:
                    caar_samples = pair[CAAR_LANE]
                    ao_safe_samples = pair[AO_SAFE_LANE]
                    # Nothing is written for this scenario until the complete
                    # static map, initial instance, actual map metadata, agent
                    # count, grid config, and CAAR artifacts all match.
                    validate_paired_episode_samples(
                        caar_samples,
                        ao_safe_samples,
                    )
                    writers[CAAR_LANE].append(caar_samples)
                    writers[AO_SAFE_LANE].append(ao_safe_samples)
                    del pending_pairs[scenario_id]
            else:
                writers[branch].append(samples)
        if pending_pairs:
            incomplete = {
                scenario_id: sorted(pair)
                for scenario_id, pair in pending_pairs.items()
            }
            raise RuntimeError(f"Incomplete paired branch results: {incomplete}")
        completed = True
    finally:
        if completed:
            manifests = {
                branch: writer.close() for branch, writer in writers.items()
            }
    return {
        "output": str(output_root),
        "canonical_scenario_manifest_sha256": scenario_digest,
        "branches": {
            branch: {
                "path": str(output_root / branch),
                "episodes": manifest["episode_count"],
                "rows": manifest["row_count"],
                "shards": manifest["shard_count"],
            }
            for branch, manifest in manifests.items()
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario-manifest",
        required=True,
        help=(
            "YAML/JSON list of {scenario_id, grid_config}; every grid_config "
            "must include fixed seed, num_agents, and max_episode_steps"
        ),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--branches",
        nargs="+",
        choices=(CAAR_LANE, AO_SAFE_LANE),
        default=[CAAR_LANE, AO_SAFE_LANE],
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--sample-fraction", type=float, default=0.2)
    parser.add_argument("--sampling-seed", type=int, default=20260731)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--shard-rows", type=int, default=100_000)
    parser.add_argument(
        "--caar-weights",
        default="weights/CAAR/radius_ablation/R5",
    )
    parser.add_argument(
        "--caar-checkpoint-kind",
        choices=("auto", "latest", "best"),
        default="auto",
    )
    parser.add_argument("--caar-device", default="cpu")
    parser.add_argument("--plan-max-steps", type=int, default=10_000)
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument(
        "--plan-use-best-move",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.shard_rows < 1:
        parser.error("--shard-rows must be positive")
    if not 0.0 <= args.gamma <= 1.0:
        parser.error("--gamma must be in [0, 1]")
    if not 0.0 < args.sample_fraction <= 1.0:
        parser.error("--sample-fraction must be in (0, 1]")
    if args.plan_max_steps < 1:
        parser.error("--plan-max-steps must be positive")
    if args.torch_num_threads < 1:
        parser.error("--torch-num-threads must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    result = collect(parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
