#!/usr/bin/env python3
"""Build the fixed large-WC3 evaluation registry from MovingAI map files."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml


MAPS = (
    ("wc3-128x64-DustwallowKeys", "dustwallowkeys.map", 179_479,
     "67f0824637bc30dc5ae58df4d10ff637a41db53026d4a4c4584e318406fbee76",
     5_660),
    ("wc3-128x64-Icecrown", "icecrown.map", 176_566,
     "0472c18028f818234c31a12fd9669e4a630f723b957da30681f245bf33a5111b",
     5_563),
    ("wc3-128x64-PlunderIsle", "plunderisle.map", 171_069,
     "ac3cf8f348de7688c33fd420dee40d20f6f8e5f4555706fe9e48321fbec4e8e0",
     5_694),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def load_map(path: Path, expected_states: int, resized_states: int) -> str:
    lines = path.read_text(encoding="ascii").splitlines()
    if lines[:4] != ["type octile", "height 512", "width 512", "map"]:
        raise RuntimeError(f"Unexpected MovingAI header: {path}")
    rows = lines[4:]
    if len(rows) != 512 or any(len(row) != 512 for row in rows):
        raise RuntimeError(f"Unexpected dimensions: {path}")
    if set("".join(rows)) - set(".GS@OTW"):
        raise RuntimeError(f"Unexpected terrain symbol: {path}")
    states = sum(symbol in ".GS" for row in rows for symbol in row)
    if states != expected_states:
        raise RuntimeError(f"State count differs for {path}: {states}")
    walkable = set(".GS")
    resized = []
    for target_row in range(64):
        output_row = []
        for target_column in range(128):
            free = sum(
                rows[source_row][source_column] in walkable
                for source_row in range(target_row * 8, (target_row + 1) * 8)
                for source_column in range(target_column * 4, (target_column + 1) * 4)
            )
            output_row.append("." if free >= 16 else "#")
        resized.append("".join(output_row))
    actual_resized_states = sum(row.count(".") for row in resized)
    if actual_resized_states != resized_states:
        raise RuntimeError(f"Resized state count differs for {path}: {actual_resized_states}")
    return "\n".join(resized)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    payload = {}
    for registry_name, filename, states, expected_sha, resized_states in MAPS:
        path = args.source_dir / filename
        if sha256(path) != expected_sha:
            raise RuntimeError(f"MovingAI source hash differs: {path}")
        payload[registry_name] = load_map(path, states, resized_states)
    args.destination.write_text(
        yaml.safe_dump(payload, sort_keys=False, width=1_000_000),
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
