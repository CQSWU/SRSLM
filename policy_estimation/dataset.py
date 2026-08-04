"""Deterministic Monte-Carlo datasets stored as auditable NPZ shards."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


DATASET_SCHEMA_VERSION = "caar_pe_mc_v1"

SAMPLE_ARRAY_NAMES = (
    "obs",
    "xy",
    "target_xy",
    "mc_return",
    "reward",
    "caar_action",
    "plan_action",
    "executed_action",
    "plan_valid",
    "plan_reverse",
    "plan_selected",
    "planner_committed",
    "agent_id",
    "timestep",
    "terminated",
    "truncated",
)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate JSON serializability and detach caller-owned dictionaries."""

    return json.loads(_canonical_json(value))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discounted_returns(
    rewards: Sequence[float] | np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Return ``G_t = r_t + gamma * G_(t+1)`` in temporal order."""

    gamma = float(gamma)
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be finite and in [0, 1].")
    values = np.asarray(rewards, dtype=np.float64).reshape(-1)
    if not bool(np.all(np.isfinite(values))):
        raise ValueError("rewards must be finite.")
    result = np.empty(values.shape, dtype=np.float32)
    running = 0.0
    for index in range(len(values) - 1, -1, -1):
        running = float(values[index]) + gamma * running
        result[index] = running
    return result


def deterministic_subsample_indices(
    size: int,
    *,
    fraction: float = 0.2,
    seed: int,
) -> np.ndarray:
    """Choose a fixed fraction without replacement using an episode seed.

    A non-empty episode always contributes at least one sample.  Returned
    indices are sorted so shard contents do not depend on RNG draw order.
    """

    size = int(size)
    fraction = float(fraction)
    if size < 0:
        raise ValueError("size must be non-negative.")
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be finite and in (0, 1].")
    if size == 0:
        return np.empty(0, dtype=np.int64)
    count = min(size, max(1, int(math.floor(size * fraction))))
    rng = np.random.default_rng(int(seed))
    return np.sort(
        rng.choice(size, size=count, replace=False).astype(np.int64)
    )


@dataclass
class EpisodeSamples:
    """Subsampled, correctly aligned ``(o_t, r_t, G_t)`` rows.

    ``plan_action`` uses ``-1`` for a missing raw Plan proposal.  Metadata is
    episode-level and must be JSON serializable.
    """

    obs: np.ndarray
    xy: np.ndarray
    target_xy: np.ndarray
    mc_return: np.ndarray
    reward: np.ndarray
    caar_action: np.ndarray
    plan_action: np.ndarray
    executed_action: np.ndarray
    plan_valid: np.ndarray
    plan_reverse: np.ndarray
    plan_selected: np.ndarray
    planner_committed: np.ndarray
    agent_id: np.ndarray
    timestep: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        conversions = {
            "obs": np.uint8,
            "xy": np.int32,
            "target_xy": np.int32,
            "mc_return": np.float32,
            "reward": np.float32,
            "caar_action": np.int16,
            "plan_action": np.int16,
            "executed_action": np.int16,
            "plan_valid": np.bool_,
            "plan_reverse": np.bool_,
            "plan_selected": np.bool_,
            "planner_committed": np.bool_,
            "agent_id": np.int32,
            "timestep": np.int32,
            "terminated": np.bool_,
            "truncated": np.bool_,
        }
        for name, dtype in conversions.items():
            value = np.asarray(getattr(self, name), dtype=dtype)
            setattr(self, name, np.ascontiguousarray(value))

        if self.obs.ndim != 4:
            raise ValueError(
                "obs must have shape (samples, channels, height, width)."
            )
        if self.xy.shape != (len(self), 2):
            raise ValueError("xy must have shape (samples, 2).")
        if self.target_xy.shape != (len(self), 2):
            raise ValueError("target_xy must have shape (samples, 2).")
        for name in SAMPLE_ARRAY_NAMES[3:]:
            value = getattr(self, name)
            if value.shape != (len(self),):
                raise ValueError(f"{name} must have shape (samples,), got {value.shape}.")
        if not bool(np.all(np.isfinite(self.mc_return))):
            raise ValueError("mc_return must be finite.")
        if not bool(np.all(np.isfinite(self.reward))):
            raise ValueError("reward must be finite.")
        if bool(np.any(self.caar_action < 0)) or bool(
            np.any(self.executed_action < 0)
        ):
            raise ValueError("CAAR and executed actions must be non-negative.")
        if bool(np.any(self.plan_action < -1)):
            raise ValueError("plan_action may only use -1 for a missing proposal.")
        self.metadata = _json_copy(self.metadata)

    def __len__(self) -> int:
        return int(self.obs.shape[0])

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in SAMPLE_ARRAY_NAMES}

    def take(self, indices: Sequence[int] | np.ndarray) -> "EpisodeSamples":
        index = np.asarray(indices, dtype=np.int64)
        return EpisodeSamples(
            **{name: value[index] for name, value in self.as_arrays().items()},
            metadata=self.metadata,
        )


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def load_npz_shard(
    path: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load one shard without permitting pickled Python objects."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in SAMPLE_ARRAY_NAMES if name not in archive]
        if "episode_index" not in archive:
            missing.append("episode_index")
        if "metadata_json" not in archive:
            missing.append("metadata_json")
        if missing:
            raise ValueError(f"NPZ shard is missing fields: {missing}")
        arrays = {
            name: np.asarray(archive[name]).copy()
            for name in (*SAMPLE_ARRAY_NAMES, "episode_index")
        }
        metadata = json.loads(str(np.asarray(archive["metadata_json"]).item()))
    if metadata.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported dataset schema: {metadata.get('schema_version')!r}."
        )
    row_count = int(metadata.get("row_count", -1))
    if row_count < 0 or any(len(array) != row_count for array in arrays.values()):
        raise ValueError("NPZ shard row count does not match its metadata.")
    return arrays, metadata


class NpzShardWriter:
    """Append episode samples and emit deterministic compressed shards."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        max_rows_per_shard: int = 100_000,
        dataset_metadata: Mapping[str, Any] | None = None,
    ):
        self.output_dir = Path(output_dir).resolve()
        self.max_rows_per_shard = int(max_rows_per_shard)
        if self.max_rows_per_shard < 1:
            raise ValueError("max_rows_per_shard must be positive.")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        occupied = [
            path
            for pattern in ("manifest.json", "shard-*.npz", "shard-*.json")
            for path in self.output_dir.glob(pattern)
        ]
        if occupied:
            raise FileExistsError(
                "Refusing to mix with an existing dataset: "
                + ", ".join(str(path) for path in sorted(occupied))
            )
        self.dataset_metadata = _json_copy(dataset_metadata or {})
        self._buffers: dict[str, list[np.ndarray]] = {
            name: [] for name in (*SAMPLE_ARRAY_NAMES, "episode_index")
        }
        self._buffer_rows = 0
        self._sample_signature: tuple[tuple[int, ...], ...] | None = None
        self._episodes: list[dict[str, Any]] = []
        self._shards: list[dict[str, Any]] = []
        self._total_rows = 0
        self._closed = False

    @staticmethod
    def _signature(samples: EpisodeSamples) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(value.shape[1:]) for value in samples.as_arrays().values()
        )

    def append(self, samples: EpisodeSamples) -> None:
        if self._closed:
            raise RuntimeError("Cannot append to a closed shard writer.")
        signature = self._signature(samples)
        if self._sample_signature is None:
            self._sample_signature = signature
        elif signature != self._sample_signature:
            raise ValueError(
                "All samples in one dataset must use the same observation schema."
            )

        episode_index = len(self._episodes)
        episode_metadata = _json_copy(samples.metadata)
        episode_metadata.update(
            {
                "episode_index": episode_index,
                "sampled_row_count": len(samples),
            }
        )
        self._episodes.append(episode_metadata)

        arrays = samples.as_arrays()
        offset = 0
        while offset < len(samples):
            capacity = self.max_rows_per_shard - self._buffer_rows
            take = min(capacity, len(samples) - offset)
            stop = offset + take
            for name, value in arrays.items():
                self._buffers[name].append(value[offset:stop])
            self._buffers["episode_index"].append(
                np.full(take, episode_index, dtype=np.int32)
            )
            self._buffer_rows += take
            self._total_rows += take
            offset = stop
            if self._buffer_rows == self.max_rows_per_shard:
                self._flush()

    def _flush(self) -> None:
        if self._buffer_rows == 0:
            return
        index = len(self._shards)
        stem = f"shard-{index:05d}"
        npz_path = self.output_dir / f"{stem}.npz"
        json_path = self.output_dir / f"{stem}.json"
        arrays = {
            name: np.concatenate(parts, axis=0)
            for name, parts in self._buffers.items()
        }
        episode_indices = sorted(
            int(value) for value in np.unique(arrays["episode_index"])
        )
        metadata = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "shard_index": index,
            "row_count": self._buffer_rows,
            "episode_indices": episode_indices,
            "arrays": {
                name: {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                }
                for name, value in arrays.items()
            },
            "dataset_metadata": self.dataset_metadata,
        }
        metadata_json = _canonical_json(metadata)
        temporary = npz_path.with_name(npz_path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                **arrays,
                metadata_json=np.asarray(metadata_json),
            )
        os.replace(temporary, npz_path)
        digest = sha256_file(npz_path)
        sidecar = dict(metadata)
        sidecar["npz_sha256"] = digest
        _atomic_write_text(
            json_path,
            json.dumps(sidecar, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n",
        )
        self._shards.append(
            {
                "path": npz_path.name,
                "metadata_path": json_path.name,
                "row_count": self._buffer_rows,
                "sha256": digest,
            }
        )
        for parts in self._buffers.values():
            parts.clear()
        self._buffer_rows = 0

    def close(self) -> dict[str, Any]:
        if self._closed:
            manifest_path = self.output_dir / "manifest.json"
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        self._flush()
        manifest = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "row_count": self._total_rows,
            "episode_count": len(self._episodes),
            "shard_count": len(self._shards),
            "dataset_metadata": self.dataset_metadata,
            "episodes": self._episodes,
            "shards": self._shards,
        }
        _atomic_write_text(
            self.output_dir / "manifest.json",
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
        )
        self._closed = True
        return manifest

    def __enter__(self) -> "NpzShardWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if exc_type is None:
            self.close()
        return False


class NpzShardDataset:
    """A small lazy, NumPy-backed Dataset compatible with DataLoader."""

    def __init__(
        self,
        path: str | Path,
        *,
        verify_hashes: bool = False,
    ):
        path = Path(path).resolve()
        self.manifest_path = path / "manifest.json" if path.is_dir() else path
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported dataset schema: {manifest.get('schema_version')!r}."
            )
        self.root = self.manifest_path.parent
        self.manifest = manifest
        self.shards = list(manifest.get("shards", []))
        self._ends: list[int] = []
        total = 0
        for shard in self.shards:
            shard_path = self.root / str(shard["path"])
            if not shard_path.is_file():
                raise FileNotFoundError(shard_path)
            if verify_hashes and sha256_file(shard_path) != shard.get("sha256"):
                raise ValueError(f"Shard checksum mismatch: {shard_path}")
            total += int(shard["row_count"])
            self._ends.append(total)
        if total != int(manifest.get("row_count", -1)):
            raise ValueError("Manifest row count does not match its shards.")
        self._cache_index: int | None = None
        self._cache_arrays: dict[str, np.ndarray] | None = None

    def __len__(self) -> int:
        return self._ends[-1] if self._ends else 0

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self._ends, index)
        start = 0 if shard_index == 0 else self._ends[shard_index - 1]
        if self._cache_index != shard_index:
            arrays, _metadata = load_npz_shard(
                self.root / str(self.shards[shard_index]["path"])
            )
            self._cache_index = shard_index
            self._cache_arrays = arrays
        assert self._cache_arrays is not None
        local = index - start
        result = {
            name: value[local] for name, value in self._cache_arrays.items()
        }
        # A conventional alias keeps the supervised training loop simple.
        result["target"] = result["mc_return"]
        return result

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_cache_index"] = None
        state["_cache_arrays"] = None
        return state


__all__ = [
    "DATASET_SCHEMA_VERSION",
    "EpisodeSamples",
    "NpzShardDataset",
    "NpzShardWriter",
    "SAMPLE_ARRAY_NAMES",
    "deterministic_subsample_indices",
    "discounted_returns",
    "load_npz_shard",
    "sha256_file",
]
