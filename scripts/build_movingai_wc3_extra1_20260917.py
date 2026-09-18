#!/usr/bin/env python3
"""Build the fourth MovingAI WC3 evaluation map with the frozen resize rule."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import yaml


REGISTRY_NAME = "wc3-128x64-TimbermawHold"
FILENAME = "timbermawhold.map"
SOURCE_STATES = 149_744
SOURCE_SHA256 = "280c25f71638e1ca330a430ba7f76ed4a54152956deab5b58bc5c61f07c2cd3e"
RESIZED_STATES = 5_101


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resize(path: Path) -> str:
    lines = path.read_text(encoding="ascii").splitlines()
    if lines[:4] != ["type octile", "height 512", "width 512", "map"]:
        raise RuntimeError(f"Unexpected MovingAI header: {path}")
    rows = lines[4:]
    if len(rows) != 512 or any(len(row) != 512 for row in rows):
        raise RuntimeError(f"Unexpected dimensions: {path}")
    if set("".join(rows)) - set(".GS@OTW"):
        raise RuntimeError(f"Unexpected terrain symbol: {path}")
    states = sum(symbol in ".GS" for row in rows for symbol in row)
    if states != SOURCE_STATES:
        raise RuntimeError(f"Source state count differs: {states}")

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
    actual = sum(row.count(".") for row in resized)
    if actual != RESIZED_STATES:
        raise RuntimeError(f"Resized state count differs: {actual}")

    remaining = {
        (row, column)
        for row, line in enumerate(resized)
        for column, symbol in enumerate(line)
        if symbol == "."
    }
    frontier = [remaining.pop()]
    connected = 0
    while frontier:
        row, column = frontier.pop()
        connected += 1
        for delta_row, delta_column in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbor = row + delta_row, column + delta_column
            if neighbor in remaining:
                remaining.remove(neighbor)
                frontier.append(neighbor)
    if connected != RESIZED_STATES or remaining:
        raise RuntimeError("Resized free space is not one connected component")
    return "\n".join(resized)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    source = args.source_dir / FILENAME
    if sha256(source) != SOURCE_SHA256:
        raise RuntimeError(f"MovingAI source hash differs: {source}")
    args.destination.write_text(
        yaml.safe_dump(
            {REGISTRY_NAME: resize(source)},
            sort_keys=False,
            width=1_000_000,
        ),
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
