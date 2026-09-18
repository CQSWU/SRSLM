#!/usr/bin/env python3
"""Small shared helpers for checkpoint and JSON artifact verification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

try:
    from scripts.switcher_checkpoint_identity import policy_model_sha256
except ModuleNotFoundError:  # Direct ``python scripts/...py`` execution.
    from switcher_checkpoint_identity import policy_model_sha256


_REGULAR_CHECKPOINT = re.compile(
    r"^checkpoint_(?P<train_step>[0-9]+)_(?P<env_steps>[0-9]+)\.pth$"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def same_training_certificate(saved: dict, rebuilt: dict) -> bool:
    """Compare immutable evidence, allowing only the retired branch display name."""
    if saved == rebuilt:
        return True
    old = saved.get("network_contract")
    new = rebuilt.get("network_contract")
    if not isinstance(old, dict) or not isinstance(new, dict):
        return False
    if old.get("branch_0") != "CAAR" or new.get("branch_0") != "ARPE":
        return False
    if old.get("branch_1") != "AORePlan" or new.get("branch_1") != "AORePlan":
        return False
    # Only this leaf changes; checkpoint/config hashes and all other data stay exact.
    return {**saved, "network_contract": {**old, "branch_0": "ARPE"}} == rebuilt


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


def checkpoint_identity(path: Path | str) -> dict[str, Any]:
    path = Path(path).resolve()
    require(path.is_file() and path.stat().st_size > 0, f"Checkpoint is missing: {path}")
    match = _REGULAR_CHECKPOINT.fullmatch(path.name)
    require(match is not None, f"Not a regular checkpoint filename: {path.name}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    require(isinstance(checkpoint, Mapping), f"Checkpoint is not a mapping: {path}")
    env_steps = checkpoint.get("env_steps")
    train_step = checkpoint.get("train_step")
    require(
        isinstance(env_steps, int) and not isinstance(env_steps, bool),
        f"Checkpoint has no integer env_steps: {path}",
    )
    require(
        isinstance(train_step, int) and not isinstance(train_step, bool),
        f"Checkpoint has no integer train_step: {path}",
    )
    require(int(env_steps) == int(match.group("env_steps")), f"Checkpoint filename/env_steps mismatch: {path}")
    require(int(train_step) == int(match.group("train_step")), f"Checkpoint filename/train_step mismatch: {path}")
    require(env_steps >= 0 and train_step >= 0, "Negative checkpoint counters")
    return {
        "path": str(path),
        "filename": path.name,
        "env_steps": int(env_steps),
        "train_step": int(train_step),
        "checkpoint_sha256": sha256_file(path),
        "policy_model_sha256": policy_model_sha256(path),
    }


def latest_regular_checkpoint(weights_dir: Path | str) -> dict[str, Any]:
    checkpoint_dir = Path(weights_dir).resolve() / "checkpoint_p0"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?")
    parser.add_argument("--weights-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.checkpoint not in (None, "checkpoint"):
        parser.error("the only command is 'checkpoint'")
    print(json.dumps(latest_regular_checkpoint(args.weights_dir), sort_keys=True))


if __name__ == "__main__":
    main()
