#!/usr/bin/env python3
"""Stable identity helpers for the policy parameters inside a checkpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch


def policy_model_sha256(path: Path | str) -> str:
    checkpoint = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint.get("model")
    if not isinstance(state, dict) or not state:
        raise ValueError(f"Checkpoint has no non-empty model state: {path}")
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not torch.is_tensor(value):
            raise ValueError(f"Model entry {key!r} is not a tensor: {path}")
        tensor = value.detach().cpu().contiguous()
        for field in (
            key.encode("utf-8"),
            str(tensor.dtype).encode("ascii"),
            repr(tuple(tensor.shape)).encode("ascii"),
            tensor.view(torch.uint8).numpy().tobytes(),
        ):
            digest.update(len(field).to_bytes(8, "little"))
            digest.update(field)
    return digest.hexdigest()


__all__ = ["policy_model_sha256"]
