#!/usr/bin/env python3
"""Train two independent original-style absolute-return estimators.

Input is one or more ``caar_pe_mc_v1`` dataset directories/manifests or plain
NPZ shards containing ``obs``, ``xy``, ``target_xy``, and ``mc_return``.
Branch labels may live in sample arrays, dataset metadata, or episode metadata.
Validation is split by whole map (falling back to scenario), never by random
rows.  ``--overfit-small-data`` is the explicit exception for a tiny smoke set.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from policy_estimation.dataset import (  # noqa: E402
    DATASET_SCHEMA_VERSION,
    NpzShardDataset,
    load_npz_shard,
    sha256_file,
)
from policy_estimation.model import (  # noqa: E402
    CHECKPOINT_SCHEMA_VERSION,
    PolicyEstimationModel,
    PolicyEstimationModelConfig,
    resolve_device,
    save_policy_return_checkpoint,
)


TRAINING_SCHEMA_VERSION = "caar_ao_absolute_return_training_v1"
BRANCHES = ("caar", "ao_safe")
_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "test": "test",
}
_HEX_DIGITS = frozenset("0123456789abcdef")


def _normalize_branch(value: Any) -> str:
    text = str(value.decode() if isinstance(value, bytes) else value).strip().lower()
    aliases = {
        "caar": "caar",
        "ao": "ao_safe",
        "ao_safe": "ao_safe",
        "ao-safe": "ao_safe",
        "aoreplan": "ao_safe",
        "ao_replan": "ao_safe",
        "ao-replan": "ao_safe",
    }
    if text not in aliases:
        raise ValueError(f"Unknown estimator branch {value!r}.")
    return aliases[text]


def _normalize_split(value: Any | None) -> str | None:
    if value is None:
        return None
    text = str(value.decode() if isinstance(value, bytes) else value).strip().lower()
    if not text:
        return None
    if text not in _SPLIT_ALIASES:
        raise ValueError(f"Unknown dataset split {value!r}.")
    return _SPLIT_ALIASES[text]


def _string(value: Any | None, default: str = "unknown") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value).strip()
    return text or default


def _sha256(value: Any | None) -> str | None:
    """Return a normalized SHA-256 digest or ``None`` for invalid metadata."""

    if value is None:
        return None
    text = _string(value, "").lower()
    if len(text) != 64 or any(character not in _HEX_DIGITS for character in text):
        return None
    return text


def _canonical_digest(value: Mapping[str, Any] | None) -> str | None:
    if not isinstance(value, Mapping):
        return None
    payload = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _identity_hash(
    metadata: Mapping[str, Any],
    *keys: str,
) -> str | None:
    containers = [metadata]
    for nested_key in (
        "caar_identity",
        "source_policy_identity",
        "behavior_contract",
    ):
        nested = metadata.get(nested_key)
        if isinstance(nested, Mapping):
            containers.append(nested)
    for container in containers:
        for key in keys:
            if key in container:
                return _sha256(container[key])
    return None


def _coerce_contract_value(value: Any, conversion):
    if value is None:
        return None
    try:
        return conversion(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _strict_bool(value: Any) -> bool | None:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return None


def _collection_implementation_identity(
    metadata: Mapping[str, Any],
) -> tuple[dict[str, str], str]:
    """Validate the exact source/binary identity used by the collector."""

    raw_aggregate = metadata.get("collection_implementation_sha256")
    aggregate = _sha256(raw_aggregate)
    if not isinstance(raw_aggregate, str) or aggregate is None:
        raise ValueError(
            "Formal episode metadata requires a valid "
            "collection_implementation_sha256."
        )
    raw_files = metadata.get("collection_implementation_files_sha256")
    if not isinstance(raw_files, Mapping) or not raw_files:
        raise ValueError(
            "Formal episode metadata requires a non-empty "
            "collection_implementation_files_sha256 mapping."
        )
    invalid_paths = [
        path
        for path, digest in raw_files.items()
        if not isinstance(path, str)
        or not path.strip()
        or not isinstance(digest, str)
        or _sha256(digest) is None
    ]
    if invalid_paths:
        raise ValueError(
            "collection_implementation_files_sha256 contains empty paths or "
            f"invalid SHA-256 values: {invalid_paths}."
        )
    # Preserve the exact stored path/value strings. The producer defines the
    # aggregate over this canonical JSON object; silently normalizing either
    # would let metadata tampering escape the declared provenance identity.
    files = {path: str(digest) for path, digest in raw_files.items()}
    canonical_json = json.dumps(
        files,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    expected = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    if expected != aggregate:
        raise ValueError(
            "collection_implementation_sha256 does not match the canonical "
            "collection_implementation_files_sha256 mapping."
        )
    return dict(sorted(files.items())), aggregate


def _build_behavior_contract(
    metadata: Mapping[str, Any],
    *,
    obs_shape: Sequence[int],
    caar_checkpoint_sha256: str | None,
    caar_config_sha256: str | None,
) -> tuple[dict[str, Any], str, dict[str, str], str]:
    """Verify, without rewriting, the declared fixed-policy contract."""

    declared = metadata.get("behavior_contract")
    if not isinstance(declared, Mapping):
        raise ValueError(
            "Formal caar_pe_mc_v1 episodes must declare behavior_contract."
        )
    # Round-trip through canonical JSON both validates serializability and
    # ensures the digest is computed from detached JSON data exactly as stored.
    try:
        canonical_json = json.dumps(
            dict(declared),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        contract = json.loads(canonical_json)
    except (TypeError, ValueError) as exc:
        raise ValueError("behavior_contract must be canonical JSON data.") from exc
    forbidden = sorted({"branch", "sampling_seed"} & set(contract))
    if forbidden:
        raise ValueError(
            "behavior_contract must be branch-independent and must not include "
            "the per-dataset base sampling seed; forbidden fields: "
            + ", ".join(forbidden)
        )
    implementation_files, implementation_sha256 = (
        _collection_implementation_identity(metadata)
    )

    observed_shape = [int(value) for value in obs_shape]
    declared_shape = contract.get("obs_shape")
    try:
        normalized_shape = [int(value) for value in declared_shape]
    except (TypeError, ValueError) as exc:
        raise ValueError("behavior_contract.obs_shape is invalid.") from exc
    if normalized_shape != observed_shape:
        raise ValueError(
            "behavior_contract.obs_shape does not match stored samples."
        )
    identity_expectations = {
        "caar_checkpoint_sha256": caar_checkpoint_sha256,
        "caar_config_sha256": caar_config_sha256,
        "collection_implementation_sha256": implementation_sha256,
    }
    for key, expected in identity_expectations.items():
        if contract.get(key) != expected:
            raise ValueError(
                f"behavior_contract.{key} does not match episode metadata."
            )
    metadata_checks = {
        "gamma": lambda value: _coerce_contract_value(value, float),
        "sample_fraction": lambda value: _coerce_contract_value(value, float),
        "sampling": lambda value: _string(value, ""),
        "sampling_seed_strategy": lambda value: _string(value, ""),
        "plan_use_best_move": _strict_bool,
        "plan_max_steps": lambda value: _coerce_contract_value(value, int),
    }
    for key, normalize in metadata_checks.items():
        if key not in metadata:
            continue
        if normalize(metadata[key]) != normalize(contract.get(key)):
            raise ValueError(
                f"behavior_contract.{key} does not match episode metadata."
            )
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    declared_digest = metadata.get("behavior_contract_sha256")
    normalized = _sha256(declared_digest)
    if normalized is None or normalized != digest:
        raise ValueError(
            "behavior_contract_sha256 does not match canonical contract."
        )
    return contract, digest, implementation_files, implementation_sha256


def _map_name(metadata: Mapping[str, Any], scenario: str) -> str:
    if metadata.get("actual_map_name") not in (None, ""):
        return _string(metadata["actual_map_name"], scenario)
    for key in ("map", "map_name"):
        if metadata.get(key) not in (None, ""):
            return _string(metadata[key], scenario)
    grid = metadata.get("grid_config")
    if isinstance(grid, Mapping):
        for key in ("map_name", "map"):
            if grid.get(key) not in (None, ""):
                return _string(grid[key], scenario)
    return scenario


class _ArrayDataset(Dataset):
    def __init__(self, arrays: Mapping[str, np.ndarray]):
        self.arrays = {
            key: np.ascontiguousarray(value)
            for key, value in arrays.items()
        }

    def __len__(self) -> int:
        return len(self.arrays["mc_return"])

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        return {
            key: value[index]
            for key, value in self.arrays.items()
        }


@dataclass(frozen=True)
class _Segment:
    dataset: Dataset
    start: int
    stop: int
    branch: str
    split: str | None
    scenario: str
    map_name: str
    source: str
    source_kind: str
    static_map_sha256: str | None = None
    initial_instance_sha256: str | None = None
    episode_config_sha256: str | None = None
    caar_checkpoint_sha256: str | None = None
    caar_config_sha256: str | None = None
    collection_implementation_files_sha256: Mapping[str, str] | None = None
    collection_implementation_sha256: str | None = None
    behavior_contract: Mapping[str, Any] | None = None
    behavior_contract_sha256: str | None = None
    horizon: int | None = None
    num_agents: int | None = None
    indices: np.ndarray | None = None

    def __len__(self) -> int:
        return (
            int(len(self.indices))
            if self.indices is not None
            else int(self.stop - self.start)
        )

    def base_index(self, local_index: int) -> int:
        if self.indices is not None:
            return int(self.indices[local_index])
        return self.start + local_index

    def group(self, mode: str) -> str:
        if mode == "scenario":
            return self.scenario
        if self.static_map_sha256 is not None:
            return f"sha256:{self.static_map_sha256}"
        name = self.map_name if self.map_name != "unknown" else self.scenario
        return f"name:{name}"


class _SegmentedDataset(Dataset):
    """Dataset view made of whole episodes/groups, with metric labels."""

    def __init__(self, segments: Iterable[_Segment]):
        self.segments = tuple(segment for segment in segments if len(segment))
        self._ends: list[int] = []
        total = 0
        for segment in self.segments:
            total += len(segment)
            self._ends.append(total)

    def __len__(self) -> int:
        return self._ends[-1] if self._ends else 0

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        segment_index = bisect.bisect_right(self._ends, index)
        segment = self.segments[segment_index]
        begin = 0 if segment_index == 0 else self._ends[segment_index - 1]
        sample = segment.dataset[segment.base_index(index - begin)]
        result = {
            "obs": np.asarray(sample["obs"], dtype=np.float32),
            "xy": np.asarray(sample["xy"], dtype=np.float32),
            "target_xy": np.asarray(sample["target_xy"], dtype=np.float32),
            "mc_return": np.float32(sample["mc_return"]),
            "scenario": segment.scenario,
            "map_name": segment.map_name,
        }
        return result


@dataclass
class _DataCatalog:
    segments: list[_Segment]
    sources: list[dict[str, Any]]


def _discover_sources(paths: Sequence[str | Path]) -> list[Path]:
    found: set[Path] = set()
    for raw in paths:
        path = Path(raw).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_file():
            if path.name == "manifest.json" or path.suffix.lower() == ".npz":
                found.add(path)
            else:
                raise ValueError(f"Unsupported data file: {path}")
            continue
        own_manifest = path / "manifest.json"
        if own_manifest.is_file():
            found.add(own_manifest.resolve())
            continue
        manifests = sorted(path.rglob("manifest.json"))
        if manifests:
            found.update(value.resolve() for value in manifests)
            continue
        shards = sorted(path.rglob("*.npz"))
        if not shards:
            raise ValueError(f"No manifest.json or NPZ shards below {path}.")
        found.update(value.resolve() for value in shards)
    if not found:
        raise ValueError("No training data sources were discovered.")
    return sorted(found)


def _load_manifest_source(path: Path, verify_hashes: bool) -> _DataCatalog:
    dataset = NpzShardDataset(path, verify_hashes=verify_hashes)
    manifest = dataset.manifest
    if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError(f"Unsupported dataset schema in {path}.")
    dataset_metadata = manifest.get("dataset_metadata") or {}
    if not isinstance(dataset_metadata, Mapping):
        raise ValueError(f"dataset_metadata must be a mapping in {path}.")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"Manifest has no episode metadata: {path}")
    if len(dataset) < 1:
        raise ValueError(f"Manifest dataset is empty: {path}")

    episode_row_counts = []
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise ValueError(f"Invalid episode metadata in {path}.")
        rows = int(episode.get("sampled_row_count", -1))
        if rows < 0:
            raise ValueError(f"Episode is missing sampled_row_count in {path}.")
        episode_row_counts.append(rows)
    if sum(episode_row_counts) != len(dataset):
        raise ValueError(
            f"Episode row counts ({sum(episode_row_counts)}) do not match "
            f"manifest rows ({len(dataset)}) in {path}."
        )

    # The episode-index integrity pass must already decompress every shard.
    # Retain the four supervised arrays from that same pass so randomized
    # DataLoader access never bounces between and repeatedly inflates NPZs.
    preload_keys = ("obs", "xy", "target_xy", "mc_return")
    preload_parts: dict[str, list[np.ndarray]] = {
        key: [] for key in preload_keys
    }
    episode_ends = np.cumsum(episode_row_counts, dtype=np.int64)
    global_offset = 0
    for shard in dataset.shards:
        arrays, _metadata = load_npz_shard(
            dataset.root / str(shard["path"])
        )
        actual = np.asarray(arrays["episode_index"], dtype=np.int64)
        positions = global_offset + np.arange(len(actual), dtype=np.int64)
        expected = np.searchsorted(episode_ends, positions, side="right")
        if not np.array_equal(actual, expected):
            mismatch = int(np.flatnonzero(actual != expected)[0])
            raise ValueError(
                "Manifest episode order does not match shard episode_index at "
                f"{shard['path']} row {mismatch}."
            )
        for key in preload_keys:
            preload_parts[key].append(np.asarray(arrays[key]))
        global_offset += len(actual)
    if global_offset != len(dataset):
        raise ValueError("Shard rows do not cover the complete manifest dataset.")
    preloaded_arrays = {
        key: np.concatenate(parts, axis=0)
        for key, parts in preload_parts.items()
    }
    preloaded_bytes = sum(
        int(value.nbytes) for value in preloaded_arrays.values()
    )
    dataset = _ArrayDataset(preloaded_arrays)
    manifest_obs_shape = tuple(
        int(value) for value in preloaded_arrays["obs"].shape[1:]
    )

    segments: list[_Segment] = []
    offset = 0
    for episode in episodes:
        if not isinstance(episode, Mapping):
            raise ValueError(f"Invalid episode metadata in {path}.")
        required_episode_provenance = (
            "collection_implementation_files_sha256",
            "collection_implementation_sha256",
            "behavior_contract",
            "behavior_contract_sha256",
        )
        missing_episode_provenance = [
            key for key in required_episode_provenance if key not in episode
        ]
        if missing_episode_provenance:
            raise ValueError(
                "Formal episode metadata is missing collection provenance "
                f"fields in {path}: {missing_episode_provenance}."
            )
        rows = int(episode.get("sampled_row_count", -1))
        if rows < 0:
            raise ValueError(f"Episode is missing sampled_row_count in {path}.")
        branch_value = episode.get("branch", dataset_metadata.get("branch"))
        if branch_value is None:
            raise ValueError(f"Episode is missing branch metadata in {path}.")
        scenario = _string(
            episode.get(
                "scenario_id",
                dataset_metadata.get("scenario_id", f"episode-{len(segments)}"),
            )
        )
        combined = dict(dataset_metadata)
        combined.update(episode)
        split = _normalize_split(combined.get("split"))
        static_map_sha256 = _identity_hash(
            combined,
            "static_map_sha256",
            "map_sha256",
        )
        initial_instance_sha256 = _identity_hash(
            combined,
            "initial_instance_sha256",
            "instance_sha256",
        )
        caar_checkpoint_sha256 = _identity_hash(
            combined,
            "caar_checkpoint_sha256",
            "checkpoint_sha256",
        )
        caar_config_sha256 = _identity_hash(
            combined,
            "caar_config_sha256",
            "config_sha256",
        )
        (
            behavior_contract,
            behavior_contract_sha256,
            collection_implementation_files_sha256,
            collection_implementation_sha256,
        ) = _build_behavior_contract(
            combined,
            obs_shape=manifest_obs_shape,
            caar_checkpoint_sha256=caar_checkpoint_sha256,
            caar_config_sha256=caar_config_sha256,
        )
        grid_config = combined.get("grid_config")
        grid_config = grid_config if isinstance(grid_config, Mapping) else {}
        horizon = _coerce_contract_value(
            grid_config.get(
                "max_episode_steps",
                combined.get("horizon"),
            ),
            int,
        )
        num_agents = _coerce_contract_value(
            grid_config.get(
                "num_agents",
                combined.get("num_agents"),
            ),
            int,
        )
        episode_config_sha256 = (
            _canonical_digest(
                {
                    "grid_config": dict(grid_config),
                    "horizon": horizon,
                    "num_agents": num_agents,
                }
            )
            if grid_config
            else None
        )
        segments.append(
            _Segment(
                dataset=dataset,
                start=offset,
                stop=offset + rows,
                branch=_normalize_branch(branch_value),
                split=split,
                scenario=scenario,
                map_name=_map_name(combined, scenario),
                source=str(path),
                source_kind="manifest",
                static_map_sha256=static_map_sha256,
                initial_instance_sha256=initial_instance_sha256,
                episode_config_sha256=episode_config_sha256,
                caar_checkpoint_sha256=caar_checkpoint_sha256,
                caar_config_sha256=caar_config_sha256,
                collection_implementation_files_sha256=(
                    collection_implementation_files_sha256
                ),
                collection_implementation_sha256=(
                    collection_implementation_sha256
                ),
                behavior_contract=behavior_contract,
                behavior_contract_sha256=behavior_contract_sha256,
                horizon=horizon,
                num_agents=num_agents,
            )
        )
        offset += rows
    if offset != len(dataset):
        raise ValueError(
            f"Episode row counts ({offset}) do not match manifest rows "
            f"({len(dataset)}) in {path}."
        )
    return _DataCatalog(
        segments=segments,
        sources=[
            {
                "path": str(path),
                "kind": "manifest",
                "schema_version": manifest.get("schema_version"),
                "sha256": sha256_file(path),
                "rows": len(dataset),
                "data_loading": "preloaded_once_per_shard",
                "preloaded_bytes": preloaded_bytes,
                "dataset_metadata": dict(dataset_metadata),
            }
        ],
    )


def _json_metadata(archive: Mapping[str, Any]) -> dict[str, Any]:
    if "metadata_json" not in archive:
        return {}
    raw = np.asarray(archive["metadata_json"])
    if raw.size != 1:
        raise ValueError("metadata_json must be scalar.")
    try:
        value = json.loads(str(raw.item()))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid metadata_json in NPZ shard.") from exc
    return value if isinstance(value, dict) else {}


def _metadata_vector(
    archive: Mapping[str, Any],
    key: str,
    rows: int,
    fallback: Any | None,
    *,
    aliases: Sequence[str] = (),
) -> np.ndarray:
    array_key = next(
        (candidate for candidate in (key, *aliases) if candidate in archive),
        None,
    )
    if array_key is not None:
        value = np.asarray(archive[array_key])
    elif fallback is not None:
        value = np.asarray(fallback)
    else:
        value = np.asarray("")
    if value.ndim == 0 or value.size == 1:
        return np.full(rows, value.reshape(-1)[0], dtype=value.dtype)
    value = value.reshape(-1)
    if len(value) != rows:
        raise ValueError(f"{key} must be scalar or have one value per row.")
    return value


def _load_plain_npz(path: Path) -> _DataCatalog:
    with np.load(path, allow_pickle=False) as archive:
        required = ("obs", "xy", "target_xy", "mc_return")
        missing = [key for key in required if key not in archive]
        if missing:
            raise ValueError(f"NPZ shard {path} is missing fields {missing}.")
        arrays = {key: np.asarray(archive[key]).copy() for key in required}
        rows = len(arrays["mc_return"])
        if arrays["obs"].ndim != 4 or arrays["obs"].shape[0] != rows:
            raise ValueError("obs must have shape (rows, 3, height, width).")
        if arrays["xy"].shape != (rows, 2):
            raise ValueError("xy must have shape (rows, 2).")
        if arrays["target_xy"].shape != (rows, 2):
            raise ValueError("target_xy must have shape (rows, 2).")
        if np.asarray(arrays["mc_return"]).shape != (rows,):
            raise ValueError("mc_return must have shape (rows,).")
        if not np.all(np.isfinite(arrays["mc_return"])):
            raise ValueError("mc_return contains non-finite targets.")
        metadata = _json_metadata(archive)
        if metadata.get("schema_version") == DATASET_SCHEMA_VERSION:
            raise ValueError(
                "A caar_pe_mc_v1 shard cannot be trained in isolation because "
                "episode/map identity metadata lives in its manifest; pass the "
                "dataset directory or manifest.json instead."
            )
        dataset_metadata = metadata.get("dataset_metadata") or metadata
        if not isinstance(dataset_metadata, Mapping):
            dataset_metadata = {}
        branches = _metadata_vector(
            archive,
            "branch",
            rows,
            dataset_metadata.get("branch"),
        )
        splits = _metadata_vector(
            archive,
            "split",
            rows,
            dataset_metadata.get("split"),
        )
        scenarios = _metadata_vector(
            archive,
            "scenario_id",
            rows,
            dataset_metadata.get(
                "scenario_id", dataset_metadata.get("scenario", path.stem)
            ),
            aliases=("scenario",),
        )
        maps = _metadata_vector(
            archive,
            "actual_map_name",
            rows,
            dataset_metadata.get(
                "actual_map_name",
                dataset_metadata.get("map_name", dataset_metadata.get("map")),
            ),
            aliases=("map_name", "map"),
        )

    dataset = _ArrayDataset(arrays)
    normalized = []
    for index in range(rows):
        branch = _normalize_branch(branches[index])
        split = _normalize_split(splits[index])
        scenario = _string(scenarios[index], path.stem)
        map_name = _string(maps[index], scenario)
        normalized.append((branch, split, scenario, map_name))
    segments = []
    for key in sorted(set(normalized), key=repr):
        indices = np.flatnonzero(
            np.asarray([value == key for value in normalized], dtype=np.bool_)
        ).astype(np.int64)
        branch, split, scenario, map_name = key
        segments.append(
            _Segment(
                dataset=dataset,
                start=0,
                stop=0,
                indices=indices,
                branch=branch,
                split=split,
                scenario=scenario,
                map_name=map_name,
                source=str(path),
                source_kind="plain_npz",
            )
        )
    return _DataCatalog(
        segments=segments,
        sources=[
            {
                "path": str(path),
                "kind": "npz",
                "sha256": sha256_file(path),
                "rows": rows,
                "dataset_metadata": dict(dataset_metadata),
            }
        ],
    )


def load_catalog(
    paths: Sequence[str | Path],
    *,
    verify_hashes: bool = False,
) -> _DataCatalog:
    segments: list[_Segment] = []
    sources: list[dict[str, Any]] = []
    for path in _discover_sources(paths):
        catalog = (
            _load_manifest_source(path, verify_hashes)
            if path.name == "manifest.json"
            else _load_plain_npz(path)
        )
        segments.extend(catalog.segments)
        sources.extend(catalog.sources)
    for branch in BRANCHES:
        if not any(segment.branch == branch for segment in segments):
            raise ValueError(f"No {branch!r} samples were found.")
    return _DataCatalog(segments, sources)


def _map_content_audit(catalog: _DataCatalog) -> dict[str, Any]:
    hash_to_names: dict[str, set[str]] = {}
    name_to_hashes: dict[str, set[str]] = {}
    fallback_names = set()
    for segment in catalog.segments:
        name = segment.map_name
        digest = segment.static_map_sha256
        if digest is None:
            fallback_names.add(name)
            continue
        hash_to_names.setdefault(digest, set()).add(name)
        name_to_hashes.setdefault(name, set()).add(digest)
    return {
        "group_semantics": "static_map_sha256_else_audited_name_fallback",
        "static_hash_to_actual_map_names": {
            digest: sorted(names) for digest, names in sorted(hash_to_names.items())
        },
        "actual_map_name_to_static_hashes": {
            name: sorted(digests) for name, digests in sorted(name_to_hashes.items())
        },
        "name_fallbacks": sorted(fallback_names),
        "static_content_group_count": len(hash_to_names),
        "name_fallback_group_count": len(fallback_names),
    }


def _validate_behavior_contract(segment: _Segment) -> None:
    contract = segment.behavior_contract
    required = (
        "schema_version",
        "dataset_schema_version",
        "gamma",
        "sample_fraction",
        "sampling",
        "caar_checkpoint_sha256",
        "caar_config_sha256",
        "collection_implementation_sha256",
        "plan_use_best_move",
        "plan_max_steps",
        "obs_shape",
        "sampling_seed_strategy",
    )
    if not isinstance(contract, Mapping):
        raise ValueError(
            f"Manifest episode {segment.scenario!r} lacks behavior_contract."
        )
    missing = [key for key in required if contract.get(key) is None]
    if missing:
        raise ValueError(
            f"Manifest episode {segment.scenario!r} has incomplete behavior "
            f"contract fields: {missing}."
        )
    if contract["schema_version"] != "caar_ao_behavior_contract_v1":
        raise ValueError("Unsupported behavior contract schema.")
    if contract["dataset_schema_version"] != DATASET_SCHEMA_VERSION:
        raise ValueError("Behavior contract uses an unsupported dataset schema.")
    gamma = float(contract["gamma"])
    fraction = float(contract["sample_fraction"])
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("behavior_contract.gamma must be finite and in [0,1].")
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError(
            "behavior_contract.sample_fraction must be finite and in (0,1]."
        )
    if not str(contract["sampling"]).strip():
        raise ValueError("behavior_contract.sampling must be non-empty.")
    if not isinstance(contract["plan_use_best_move"], bool):
        raise ValueError("behavior_contract.plan_use_best_move must be boolean.")
    if int(contract["plan_max_steps"]) < 1:
        raise ValueError("behavior_contract.plan_max_steps must be positive.")
    shape = tuple(int(value) for value in contract["obs_shape"])
    if len(shape) != 3 or shape[0] != 3 or min(shape[1:]) < 1:
        raise ValueError("behavior_contract.obs_shape must be (3,H,W).")
    if contract["sampling_seed_strategy"] != "sha256(base_seed,scenario_id)":
        raise ValueError(
            "behavior_contract.sampling_seed_strategy must be exactly "
            "'sha256(base_seed,scenario_id)'."
        )
    if "sampling_seed" in contract or "branch" in contract:
        raise ValueError(
            "behavior_contract must not include sampling_seed or branch."
        )
    if _sha256(contract["caar_checkpoint_sha256"]) is None:
        raise ValueError("Invalid CAAR checkpoint SHA-256 in behavior contract.")
    if _sha256(contract["caar_config_sha256"]) is None:
        raise ValueError("Invalid CAAR config SHA-256 in behavior contract.")
    if (
        _sha256(contract["collection_implementation_sha256"])
        != segment.collection_implementation_sha256
    ):
        raise ValueError(
            "Behavior contract collection implementation identity does not "
            "match episode provenance metadata."
        )
    if _canonical_digest(contract) != segment.behavior_contract_sha256:
        raise ValueError("Stored behavior contract digest is inconsistent.")


def _base_sampling_seed_audit(catalog: _DataCatalog) -> dict[str, Any]:
    """Validate the dataset-level seed outside the per-episode contract."""

    manifests = [
        source for source in catalog.sources if source.get("kind") == "manifest"
    ]
    if not manifests:
        return {
            "status": "not_available_plain_npz",
            "base_sampling_seed": None,
            "manifest_count": 0,
            "sources": [],
        }
    records = []
    invalid = []
    for source in manifests:
        metadata = source.get("dataset_metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        raw = metadata.get("sampling_seed")
        if isinstance(raw, bool) or not isinstance(raw, int):
            invalid.append(str(source.get("path")))
            continue
        records.append(
            {
                "path": str(source.get("path")),
                "branch": (
                    _normalize_branch(metadata["branch"])
                    if metadata.get("branch") is not None
                    else None
                ),
                "sampling_seed": int(raw),
            }
        )
    if invalid:
        raise ValueError(
            "Every formal dataset manifest must declare one integer "
            "dataset_metadata.sampling_seed; invalid sources: "
            + ", ".join(sorted(invalid))
        )
    seeds = sorted({record["sampling_seed"] for record in records})
    if len(seeds) != 1:
        raise ValueError(
            "Paired formal datasets must use one identical base sampling_seed, "
            f"got {seeds}."
        )
    return {
        "status": "verified",
        "base_sampling_seed": seeds[0],
        "manifest_count": len(manifests),
        "sources": sorted(records, key=lambda value: value["path"]),
    }


def audit_catalog(
    catalog: _DataCatalog,
    *,
    overfit_small_data: bool,
) -> dict[str, Any]:
    """Audit content isolation, paired episodes, and frozen-policy identity."""

    manifest_segments = [
        segment for segment in catalog.segments if segment.source_kind == "manifest"
    ]
    plain_segments = [
        segment for segment in catalog.segments if segment.source_kind == "plain_npz"
    ]
    map_audit = _map_content_audit(catalog)
    sampling_seed_audit = _base_sampling_seed_audit(catalog)
    if plain_segments and not overfit_small_data:
        raise ValueError(
            "Plain NPZ inputs cannot prove paired policy identity or behavior "
            "contracts and are allowed only with --overfit-small-data for a "
            "non-deployable smoke test."
        )

    checkpoint_hashes = sorted(
        {
            value
            for value in (
                segment.caar_checkpoint_sha256 for segment in manifest_segments
            )
            if value is not None
        }
    )
    config_hashes = sorted(
        {
            value
            for value in (
                segment.caar_config_sha256 for segment in manifest_segments
            )
            if value is not None
        }
    )
    contract_hashes = sorted(
        {
            value
            for value in (
                segment.behavior_contract_sha256 for segment in manifest_segments
            )
            if value is not None
        }
    )
    implementation_hashes = sorted(
        {
            value
            for value in (
                segment.collection_implementation_sha256
                for segment in manifest_segments
            )
            if value is not None
        }
    )
    horizons = sorted(
        {segment.horizon for segment in manifest_segments if segment.horizon is not None}
    )
    agent_counts = sorted(
        {
            segment.num_agents
            for segment in manifest_segments
            if segment.num_agents is not None
        }
    )

    if overfit_small_data:
        identity = {
            "caar_checkpoint_sha256": (
                checkpoint_hashes[0] if len(checkpoint_hashes) == 1 else None
            ),
            "caar_config_sha256": (
                config_hashes[0] if len(config_hashes) == 1 else None
            ),
            "observed_checkpoint_hashes": checkpoint_hashes,
            "observed_config_hashes": config_hashes,
            "globally_consistent": (
                len(checkpoint_hashes) <= 1 and len(config_hashes) <= 1
            ),
        }
        return {
            "mode": "degraded_explicit_small_data_overfit",
            "deployable": False,
            "formal_manifest_episode_count": len(manifest_segments),
            "plain_npz_segment_count": len(plain_segments),
            "map_content": map_audit,
            "base_sampling_seed": sampling_seed_audit,
            "source_policy_identity": identity,
            "behavior_contract": (
                dict(manifest_segments[0].behavior_contract)
                if len(contract_hashes) == 1 and manifest_segments
                else None
            ),
            "behavior_contract_sha256": (
                contract_hashes[0] if len(contract_hashes) == 1 else None
            ),
            "observed_behavior_contract_hashes": contract_hashes,
            "collection_implementation_identity": (
                {
                    "collection_implementation_files_sha256": dict(
                        manifest_segments[0].collection_implementation_files_sha256
                        or {}
                    ),
                    "collection_implementation_sha256": implementation_hashes[0],
                }
                if len(implementation_hashes) == 1 and manifest_segments
                else None
            ),
            "observed_collection_implementation_hashes": implementation_hashes,
            "pairing": {
                "status": "not_enforced_in_explicit_overfit_mode",
                "pair_count": 0,
            },
            "training_horizons": horizons,
            "training_agent_counts": agent_counts,
        }

    for segment in manifest_segments:
        missing = []
        if segment.static_map_sha256 is None:
            missing.append("static_map_sha256")
        if segment.initial_instance_sha256 is None:
            missing.append("initial_instance_sha256")
        if segment.episode_config_sha256 is None:
            missing.append("grid_config")
        if segment.caar_checkpoint_sha256 is None:
            missing.append("caar_checkpoint_sha256")
        if segment.caar_config_sha256 is None:
            missing.append("caar_config_sha256")
        if segment.collection_implementation_sha256 is None:
            missing.append("collection_implementation_sha256")
        if not segment.collection_implementation_files_sha256:
            missing.append("collection_implementation_files_sha256")
        if segment.horizon is None or segment.horizon < 1:
            missing.append("max_episode_steps")
        if segment.num_agents is None or segment.num_agents < 1:
            missing.append("num_agents")
        if missing:
            raise ValueError(
                f"Formal manifest episode {segment.scenario!r} has missing or "
                f"invalid identity/config fields: {missing}."
            )
        _validate_behavior_contract(segment)

    if manifest_segments:
        if len(checkpoint_hashes) != 1:
            raise ValueError(
                "Formal datasets must use one globally consistent CAAR "
                f"checkpoint_sha256, got {checkpoint_hashes}."
            )
        if len(config_hashes) != 1:
            raise ValueError(
                "Formal datasets must use one globally consistent CAAR "
                f"config_sha256, got {config_hashes}."
            )
        if len(contract_hashes) != 1:
            raise ValueError(
                "Formal datasets must use one common behavior_contract_sha256, "
                f"got {contract_hashes}."
            )
        if len(implementation_hashes) != 1:
            raise ValueError(
                "Formal datasets must use one common collection "
                "implementation identity, got "
                f"{implementation_hashes}."
            )

    buckets: dict[tuple[str, str, str, str], dict[str, list[_Segment]]] = {}
    for segment in manifest_segments:
        key = (
            segment.scenario,
            str(segment.initial_instance_sha256),
            str(segment.static_map_sha256),
            str(segment.episode_config_sha256),
        )
        bucket = buckets.setdefault(key, {branch: [] for branch in BRANCHES})
        bucket[segment.branch].append(segment)
    invalid_pairs = []
    for key, bucket in buckets.items():
        counts = {branch: len(bucket[branch]) for branch in BRANCHES}
        if any(counts[branch] != 1 for branch in BRANCHES):
            invalid_pairs.append({"key": list(key), "counts": counts})
        elif bucket["caar"][0].split != bucket["ao_safe"][0].split:
            invalid_pairs.append(
                {
                    "key": list(key),
                    "counts": counts,
                    "reason": "paired branches use different split labels",
                }
            )
    if invalid_pairs:
        preview = invalid_pairs[:5]
        raise ValueError(
            "CAAR/AO-safe episode pairing is incomplete or duplicated: "
            + json.dumps(preview, sort_keys=True)
        )
    paired_payload = [
        {
            "scenario_id": key[0],
            "initial_instance_sha256": key[1],
            "static_map_sha256": key[2],
            "episode_config_sha256": key[3],
        }
        for key in sorted(buckets)
    ]
    paired_digest = hashlib.sha256(
        json.dumps(
            paired_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if manifest_segments and plain_segments:
        mode = "strict_manifest_with_audited_plain_npz_degradation"
    elif manifest_segments:
        mode = "strict_paired_manifest"
    else:
        mode = "audited_plain_npz_degradation"
    checkpoint_hash = checkpoint_hashes[0] if len(checkpoint_hashes) == 1 else None
    config_hash = config_hashes[0] if len(config_hashes) == 1 else None
    contract_hash = contract_hashes[0] if len(contract_hashes) == 1 else None
    return {
        "mode": mode,
        "deployable": bool(manifest_segments and not plain_segments),
        "formal_manifest_episode_count": len(manifest_segments),
        "plain_npz_segment_count": len(plain_segments),
        "map_content": map_audit,
        "base_sampling_seed": sampling_seed_audit,
        "source_policy_identity": {
            "caar_checkpoint_sha256": checkpoint_hash,
            "caar_config_sha256": config_hash,
            "observed_checkpoint_hashes": checkpoint_hashes,
            "observed_config_hashes": config_hashes,
            "globally_consistent": bool(
                not manifest_segments
                or (len(checkpoint_hashes) == 1 and len(config_hashes) == 1)
            ),
        },
        "behavior_contract": (
            dict(manifest_segments[0].behavior_contract)
            if contract_hash is not None and manifest_segments
            else None
        ),
        "behavior_contract_sha256": contract_hash,
        "observed_behavior_contract_hashes": contract_hashes,
        "collection_implementation_identity": (
            {
                "collection_implementation_files_sha256": dict(
                    manifest_segments[0].collection_implementation_files_sha256
                    or {}
                ),
                "collection_implementation_sha256": implementation_hashes[0],
            }
            if len(implementation_hashes) == 1 and manifest_segments
            else None
        ),
        "observed_collection_implementation_hashes": implementation_hashes,
        "pairing": {
            "status": "verified" if manifest_segments else "not_available_plain_npz",
            "pair_count": len(buckets),
            "key_fields": [
                "scenario_id",
                "initial_instance_sha256",
                "static_map_sha256",
                "episode_config_sha256",
            ],
            "paired_keys_sha256": paired_digest,
            "duplicates_or_missing": 0,
        },
        "training_horizons": horizons,
        "training_agent_counts": agent_counts,
        "episode_config_count": len(
            {
                segment.episode_config_sha256
                for segment in manifest_segments
                if segment.episode_config_sha256 is not None
            }
        ),
    }


def split_catalog(
    catalog: _DataCatalog,
    *,
    validation_fraction: float,
    split_seed: int,
    split_group: str,
    overfit_small_data: bool,
) -> tuple[
    dict[str, _SegmentedDataset],
    dict[str, _SegmentedDataset],
    dict[str, _SegmentedDataset],
    dict[str, Any],
]:
    usable = [segment for segment in catalog.segments if segment.split != "test"]
    test_segments = [segment for segment in catalog.segments if segment.split == "test"]
    test = {
        branch: _SegmentedDataset(
            segment for segment in test_segments if segment.branch == branch
        )
        for branch in BRANCHES
    }
    if test_segments and any(len(test[branch]) == 0 for branch in BRANCHES):
        raise ValueError(
            "An explicit test split must contain paired samples for both branches."
        )
    test_groups = {segment.group(split_group) for segment in test_segments}
    map_content = _map_content_audit(catalog)
    if not overfit_small_data:
        missing_static_hash = [
            segment.scenario
            for segment in usable
            if segment.source_kind == "manifest"
            and segment.static_map_sha256 is None
        ]
        if missing_static_hash:
            raise ValueError(
                "Formal manifest map splitting requires a valid 64-character "
                "static_map_sha256 for every episode; invalid scenarios: "
                + ", ".join(sorted(set(missing_static_hash)))
            )
    if overfit_small_data:
        train = {
            branch: _SegmentedDataset(
                segment for segment in usable if segment.branch == branch
            )
            for branch in BRANCHES
        }
        validation = {
            branch: _SegmentedDataset(
                segment for segment in usable if segment.branch == branch
            )
            for branch in BRANCHES
        }
        audit = {
            "mode": "explicit_small_data_overfit",
            "group_key": split_group,
            "train_groups": sorted(
                {segment.group(split_group) for segment in usable}
            ),
            "validation_groups": sorted(
                {segment.group(split_group) for segment in usable}
            ),
            "test_groups": sorted(test_groups),
            "groups_disjoint": False,
            "train_validation_overlap": sorted(
                {segment.group(split_group) for segment in usable}
            ),
            "train_test_overlap": sorted(
                {segment.group(split_group) for segment in usable} & test_groups
            ),
            "validation_test_overlap": sorted(
                {segment.group(split_group) for segment in usable} & test_groups
            ),
            "test_isolation_enforced": False,
            "map_content": map_content,
            "test_samples": {branch: len(test[branch]) for branch in BRANCHES},
        }
        return train, validation, test, audit

    explicit = [segment.split for segment in usable]
    has_explicit = any(value is not None for value in explicit)
    if has_explicit and any(value is None for value in explicit):
        raise ValueError(
            "Split metadata is only partially specified; label every episode "
            "or remove split metadata and use automatic grouped splitting."
        )

    if has_explicit:
        train_segments = [segment for segment in usable if segment.split == "train"]
        validation_segments = [
            segment for segment in usable if segment.split == "validation"
        ]
    else:
        groups = sorted({segment.group(split_group) for segment in usable})
        if len(groups) < 2:
            raise ValueError(
                "Grouped validation needs at least two maps/scenarios; use "
                "--overfit-small-data only for a smoke test."
            )
        rng = np.random.default_rng(int(split_seed))
        shuffled = list(groups)
        rng.shuffle(shuffled)
        validation_count = max(
            1,
            min(len(groups) - 1, int(round(len(groups) * validation_fraction))),
        )
        validation_groups = set(shuffled[:validation_count])
        train_segments = [
            segment
            for segment in usable
            if segment.group(split_group) not in validation_groups
        ]
        validation_segments = [
            segment
            for segment in usable
            if segment.group(split_group) in validation_groups
        ]

    train_groups = {segment.group(split_group) for segment in train_segments}
    validation_groups = {
        segment.group(split_group) for segment in validation_segments
    }
    overlap = train_groups & validation_groups
    if overlap:
        raise ValueError(
            "Train/validation group leakage detected: " + ", ".join(sorted(overlap))
        )
    train_test_overlap = train_groups & test_groups
    validation_test_overlap = validation_groups & test_groups
    if train_test_overlap or validation_test_overlap:
        raise ValueError(
            "Explicit test static-content groups must be isolated from both "
            "train and validation; train/test overlap="
            f"{sorted(train_test_overlap)}, validation/test overlap="
            f"{sorted(validation_test_overlap)}."
        )

    train = {
        branch: _SegmentedDataset(
            segment for segment in train_segments if segment.branch == branch
        )
        for branch in BRANCHES
    }
    validation = {
        branch: _SegmentedDataset(
            segment for segment in validation_segments if segment.branch == branch
        )
        for branch in BRANCHES
    }
    for branch in BRANCHES:
        if len(train[branch]) == 0 or len(validation[branch]) == 0:
            raise ValueError(
                f"Branch {branch!r} needs non-empty train and validation data."
            )
    audit = {
        "mode": "metadata" if has_explicit else "automatic_grouped",
        "group_key": split_group,
        "train_groups": sorted(train_groups),
        "validation_groups": sorted(validation_groups),
        "test_groups": sorted(test_groups),
        "groups_disjoint": True,
        "train_validation_overlap": [],
        "train_test_overlap": [],
        "validation_test_overlap": [],
        "test_isolation_enforced": True,
        "map_content": map_content,
        "test_samples": {branch: len(test[branch]) for branch in BRANCHES},
    }
    return train, validation, test, audit


def _infer_obs_shape(datasets: Iterable[_SegmentedDataset]) -> tuple[int, int, int]:
    shapes = set()
    for dataset in datasets:
        for segment in dataset.segments:
            sample = segment.dataset[segment.base_index(0)]
            shapes.add(tuple(int(value) for value in np.asarray(sample["obs"]).shape))
    if len(shapes) != 1:
        raise ValueError(f"All estimator samples must share one obs shape, got {shapes}.")
    shape = next(iter(shapes))
    if len(shape) != 3 or shape[0] != 4:
        raise ValueError(f"Estimator obs must have shape (4,H,W), got {shape}.")
    return shape


def _model_batch(batch: Mapping[str, Any], device: torch.device):
    observations = {
        key: torch.as_tensor(batch[key], device=device, dtype=torch.float32)
        for key in ("obs", "xy", "target_xy")
    }
    target = torch.as_tensor(
        batch["mc_return"], device=device, dtype=torch.float32
    ).reshape(-1)
    return observations, target


def _metric_result(sums: Mapping[str, float], counts: Mapping[str, int]):
    return {
        key: {
            "mse": float(sums[key] / counts[key]),
            "samples": int(counts[key]),
        }
        for key in sorted(sums)
    }


def evaluate(
    model: PolicyEstimationModel,
    dataset: Dataset,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    total_squared = 0.0
    total_count = 0
    scenario_sums: dict[str, float] = {}
    scenario_counts: dict[str, int] = {}
    map_sums: dict[str, float] = {}
    map_counts: dict[str, int] = {}
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            observations, target = _model_batch(batch, device)
            prediction = model(observations)
            squared = (prediction - target).square().detach().cpu().numpy()
            total_squared += float(squared.sum(dtype=np.float64))
            total_count += int(len(squared))
            for value, scenario, map_name in zip(
                squared,
                batch["scenario"],
                batch["map_name"],
            ):
                scenario = str(scenario)
                map_name = str(map_name)
                scenario_sums[scenario] = scenario_sums.get(scenario, 0.0) + float(value)
                scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
                map_sums[map_name] = map_sums.get(map_name, 0.0) + float(value)
                map_counts[map_name] = map_counts.get(map_name, 0) + 1
    if total_count == 0:
        raise ValueError("Cannot evaluate an empty dataset.")
    return {
        "mse": float(total_squared / total_count),
        "samples": total_count,
        "by_scenario": _metric_result(scenario_sums, scenario_counts),
        "by_map": _metric_result(map_sums, map_counts),
    }


def train_branch(
    branch: str,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    test_dataset: Dataset,
    *,
    model_config: PolicyEstimationModelConfig,
    device: torch.device,
    num_trials: int,
    epochs_per_trial: int,
    batch_size: int,
    learning_rate: float,
    num_workers: int,
    seed: int,
    partition_seed: int,
) -> tuple[PolicyEstimationModel, dict[str, Any]]:
    """Fit one model from only its own branch's raw MC targets."""

    branch = _normalize_branch(branch)
    torch.manual_seed(int(seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    model = PolicyEstimationModel(model_config).to(device)
    criterion = nn.MSELoss()
    if len(train_dataset) < num_trials:
        raise ValueError(
            f"Branch {branch!r} has {len(train_dataset)} training rows, fewer "
            f"than num_trials={num_trials}."
        )
    partition_rng = np.random.default_rng(int(partition_seed))
    shuffled_indices = partition_rng.permutation(len(train_dataset))
    trial_indices = [
        np.asarray(values, dtype=np.int64)
        for values in np.array_split(shuffled_indices, num_trials)
    ]
    partition_digest = hashlib.sha256(
        b"".join(
            np.asarray(values, dtype="<i8").tobytes() for values in trial_indices
        )
    ).hexdigest()
    if sum(len(values) for values in trial_indices) != len(train_dataset):
        raise RuntimeError("Trial partition does not cover the training dataset.")
    if len(np.unique(np.concatenate(trial_indices))) != len(train_dataset):
        raise RuntimeError("Trial training subsets are not mutually exclusive.")
    history = []
    best_score = math.inf
    best_trial = -1
    best_epoch = -1
    best_global_epoch = -1
    best_state = None
    global_epoch = 0
    for trial_index, indices in enumerate(trial_indices, start=1):
        # Match the official implementation: model weights persist, while a
        # fresh Adam optimizer is constructed for each newly collected trial.
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed) + trial_index)
        loader = DataLoader(
            Subset(train_dataset, indices.tolist()),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
        )
        for epoch_in_trial in range(1, epochs_per_trial + 1):
            global_epoch += 1
            model.train()
            squared_sum = 0.0
            sample_count = 0
            for batch in loader:
                observations, target = _model_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(observations)
                loss = criterion(prediction, target)
                loss.backward()
                optimizer.step()
                squared_sum += float(
                    (prediction.detach() - target).square().sum().cpu()
                )
                sample_count += int(len(target))
            validation = evaluate(
                model,
                validation_dataset,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            epoch_metrics = {
                "trial": trial_index,
                "epoch_in_trial": epoch_in_trial,
                "global_epoch": global_epoch,
                "trial_samples": len(indices),
                "train_mse": float(squared_sum / sample_count),
                "validation_mse": validation["mse"],
            }
            selection_score = float(validation["mse"])
            history.append(epoch_metrics)
            if selection_score < best_score:
                best_score = selection_score
                best_trial = trial_index
                best_epoch = epoch_in_trial
                best_global_epoch = global_epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
    if best_state is None:
        raise RuntimeError(f"Training branch {branch} produced no checkpoint.")
    model.load_state_dict(best_state, strict=True)
    model.to(device).eval()
    metrics = {
        "branch": branch,
        "objective": "mse_on_raw_mc_return",
        "num_trials": num_trials,
        "epochs_per_trial": epochs_per_trial,
        "total_trial_epochs": num_trials * epochs_per_trial,
        "equivalent_full_dataset_passes": epochs_per_trial,
        "trial_sample_counts": [len(values) for values in trial_indices],
        "trial_partition_sha256": partition_digest,
        "trial_partition_seed": int(partition_seed),
        "validation_selection": "sample_mse",
        "best_validation_selection_score": best_score,
        "best_trial": best_trial,
        "best_epoch": best_epoch,
        "best_global_epoch": best_global_epoch,
        "history": history,
        "train": evaluate(
            model,
            train_dataset,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
        "validation": evaluate(
            model,
            validation_dataset,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
        "test": (
            evaluate(
                model,
                test_dataset,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            if len(test_dataset)
            else {"samples": 0}
        ),
        "test_samples": len(test_dataset),
    }
    return model, metrics


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def train_estimators(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "caar": output_dir / "caar_estimator.pth",
        "ao_safe": output_dir / "ao_estimator.pth",
    }
    reserved = [*outputs.values(), output_dir / "manifest.json", output_dir / "metrics.json"]
    if not args.overwrite:
        occupied = [path for path in reserved if path.exists()]
        if occupied:
            raise FileExistsError(
                "Refusing to overwrite estimator outputs: "
                + ", ".join(str(path) for path in occupied)
            )

    catalog = load_catalog(args.data, verify_hashes=args.verify_hashes)
    catalog_audit = audit_catalog(
        catalog,
        overfit_small_data=args.overfit_small_data,
    )
    train_data, validation_data, test_data, split_audit = split_catalog(
        catalog,
        validation_fraction=args.validation_fraction,
        split_seed=args.split_seed,
        split_group=args.split_group,
        overfit_small_data=args.overfit_small_data,
    )
    obs_shape = _infer_obs_shape(
        [*train_data.values(), *validation_data.values(), *test_data.values()]
    )
    model_config = PolicyEstimationModelConfig(
        obs_shape=obs_shape,
        encoder_num_filters=args.encoder_filters,
        encoder_num_res_blocks=args.encoder_res_blocks,
        coordinate_encoding="absolute_v1",
    )
    device = resolve_device(args.device)

    metrics = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "deployable": bool(catalog_audit["deployable"]),
        "test_samples": {
            branch: len(test_data[branch]) for branch in BRANCHES
        },
        "branches": {},
    }
    checkpoint_records = {}
    identity = catalog_audit["source_policy_identity"]
    base_sampling_seed_audit = catalog_audit["base_sampling_seed"]
    collection_identity = catalog_audit[
        "collection_implementation_identity"
    ]
    for offset, branch in enumerate(BRANCHES):
        branch_seed = int(args.seed) + offset
        model, branch_metrics = train_branch(
            branch,
            train_data[branch],
            validation_data[branch],
            test_data[branch],
            model_config=model_config,
            device=device,
            num_trials=args.num_trials,
            epochs_per_trial=args.epochs_per_trial,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_workers=args.num_workers,
            seed=branch_seed,
            partition_seed=int(args.seed),
        )
        metrics["branches"][branch] = branch_metrics
        checkpoint_path = save_policy_return_checkpoint(
            outputs[branch],
            model,
            branch=branch,
            training_metadata={
                "training_schema_version": TRAINING_SCHEMA_VERSION,
                "objective": "mse_on_raw_mc_return",
                "behavior_branch": branch,
                "seed": branch_seed,
                "partition_seed": int(args.seed),
                "num_trials": int(args.num_trials),
                "epochs_per_trial": int(args.epochs_per_trial),
                "total_trial_epochs": int(
                    args.num_trials * args.epochs_per_trial
                ),
                "equivalent_full_dataset_passes": int(
                    args.epochs_per_trial
                ),
                "trial_partition_sha256": branch_metrics[
                    "trial_partition_sha256"
                ],
                "best_epoch": branch_metrics["best_epoch"],
                "best_trial": branch_metrics["best_trial"],
                "best_global_epoch": branch_metrics["best_global_epoch"],
                "best_validation_mse": branch_metrics["validation"]["mse"],
                "validation_selection": branch_metrics[
                    "validation_selection"
                ],
                "best_validation_selection_score": branch_metrics[
                    "best_validation_selection_score"
                ],
                "train_samples": len(train_data[branch]),
                "validation_samples": len(validation_data[branch]),
                "test_samples": len(test_data[branch]),
                "overfit_small_data": bool(args.overfit_small_data),
                "deployable": bool(catalog_audit["deployable"]),
                "caar_checkpoint_sha256": identity[
                    "caar_checkpoint_sha256"
                ],
                "caar_config_sha256": identity["caar_config_sha256"],
                "behavior_contract": catalog_audit["behavior_contract"],
                "behavior_contract_sha256": catalog_audit[
                    "behavior_contract_sha256"
                ],
                "collection_implementation_files_sha256": (
                    collection_identity[
                        "collection_implementation_files_sha256"
                    ]
                    if collection_identity is not None
                    else None
                ),
                "collection_implementation_sha256": (
                    collection_identity["collection_implementation_sha256"]
                    if collection_identity is not None
                    else None
                ),
                "base_sampling_seed": base_sampling_seed_audit[
                    "base_sampling_seed"
                ],
                "base_sampling_seed_audit": base_sampling_seed_audit,
                "pairing_audit": catalog_audit["pairing"],
                "map_content_audit": catalog_audit["map_content"],
                "training_horizons": catalog_audit["training_horizons"],
                "training_agent_counts": catalog_audit[
                    "training_agent_counts"
                ],
                "coordinate_encoding": model_config.coordinate_encoding,
            },
        )
        checkpoint_records[branch] = {
            "path": checkpoint_path.name,
            "sha256": sha256_file(checkpoint_path),
            "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        }

    partition_digests = {
        metrics["branches"][branch]["trial_partition_sha256"]
        for branch in BRANCHES
    }
    partitions_identical = len(partition_digests) == 1

    manifest = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "method": "two_independent_absolute_raw_mc_return_estimators",
        "branches": list(BRANCHES),
        "deployable": bool(catalog_audit["deployable"]),
        "model_config": model_config.to_dict(),
        "training": {
            "device": str(device),
            "schedule": "official_style_continuous_model_disjoint_trial_subsets",
            "num_trials": int(args.num_trials),
            "epochs_per_trial": int(args.epochs_per_trial),
            "total_trial_epochs": int(
                args.num_trials * args.epochs_per_trial
            ),
            "equivalent_full_dataset_passes": int(args.epochs_per_trial),
            "optimizer_recreated_each_trial": True,
            "model_persists_across_trials": True,
            "validation_selection": "global_best_validation_mse",
            "validation_selection_metric": "sample_mse",
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "optimizer": "Adam",
            "loss": "MSE",
            "coordinate_encoding": model_config.coordinate_encoding,
            "seed": int(args.seed),
            "trial_partition_seed": int(args.seed),
            "paired_trial_partition_sha256": (
                next(iter(partition_digests))
                if partitions_identical
                else None
            ),
            "paired_trial_partition_status": (
                "identical_local_row_partition"
                if partitions_identical
                else "same_seed_branch_local_partitions_differ_in_row_count"
            ),
            "paired_trial_partition_note": (
                "Both branches use the same partition seed. Digests can differ "
                "only when fixed policies yield different trajectory/sample "
                "counts; each branch still has disjoint complete coverage."
            ),
            "train_samples": {
                branch: len(train_data[branch]) for branch in BRANCHES
            },
            "validation_samples": {
                branch: len(validation_data[branch]) for branch in BRANCHES
            },
            "test_samples": {
                branch: len(test_data[branch]) for branch in BRANCHES
            },
        },
        "source_policy_identity": identity,
        "behavior_contract": catalog_audit["behavior_contract"],
        "behavior_contract_sha256": catalog_audit[
            "behavior_contract_sha256"
        ],
        "collection_implementation_identity": collection_identity,
        "base_sampling_seed_audit": base_sampling_seed_audit,
        "pairing_audit": catalog_audit["pairing"],
        "training_horizons": catalog_audit["training_horizons"],
        "training_agent_counts": catalog_audit["training_agent_counts"],
        "catalog_audit": catalog_audit,
        "split_audit": split_audit,
        "sources": catalog.sources,
        "checkpoints": checkpoint_records,
    }
    _atomic_json(output_dir / "metrics.json", metrics)
    _atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        nargs="+",
        required=True,
        help="Dataset parent, caar_pe_mc_v1 manifest/directory, or NPZ shards",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--num-trials",
        type=int,
        default=7,
        help="Official-style independently partitioned collection trials",
    )
    parser.add_argument(
        "--epochs-per-trial",
        type=int,
        default=3,
        help="Dataset passes over each mutually exclusive trial subset",
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--split-seed", type=int, default=20260731)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument(
        "--split-group",
        choices=("map", "scenario"),
        default="map",
    )
    parser.add_argument("--encoder-filters", type=int, default=64)
    parser.add_argument("--encoder-res-blocks", type=int, default=3)
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument(
        "--overfit-small-data",
        action="store_true",
        help="Use the same tiny dataset for train and validation smoke tests",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.num_trials < 1:
        parser.error("--num-trials must be positive")
    if args.epochs_per_trial < 1:
        parser.error("--epochs-per-trial must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be positive and finite")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if not 0.0 < args.validation_fraction < 1.0:
        parser.error("--validation-fraction must be in (0, 1)")
    if args.encoder_filters < 1:
        parser.error("--encoder-filters must be positive")
    if args.encoder_res_blocks < 0:
        parser.error("--encoder-res-blocks must be non-negative")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    manifest = train_estimators(parse_args(argv))
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
