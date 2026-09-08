"""Check the published B counts and aggregates; no private files or ML packages needed."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

METHODS = ("EPOM-L", "CAAR", "SRSLM")
RULES = ("block_both", "soft")
MAPS = ("mazes-s40_wc4_od30", "mazes-s45_wc4_od55", "random-s40_d0.15",
        "random-s44_d0.35", "sc1-TheFrozenSea", "sc1-Turbo",
        "street-Shanghai_0", "street-Sydney_0")
POPS, SEEDS = (200, 600), (0, 42, 123)
KEY = ("method", "collision_system", "map_name", "num_agents", "seed")
COUNTS = tuple(f"window_{i}_goals" for i in range(1, 9))
EPISODE_FIELDS = KEY + COUNTS + ("completed_targets", "avg_throughput")
AGGREGATES = ("B_window_throughput.csv", "B_window_by_population.csv",
              "B_tail_comparison.csv", "B_tail_by_population.csv")
DATA_FILES = ("B_episode_windows.csv",) + AGGREGATES
FIGURES = ("B_window_throughput_both_soft.png", "B_window_throughput_both_soft.pdf")
COLLECTION_SHA = "a747ffb56051ce607eca6fc5a6dc0318957cbb7d786986796ff28e8310fc7a85"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(value, expected):
    return math.isfinite(float(value)) and math.isclose(float(value), expected, rel_tol=1e-12, abs_tol=1e-12)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames), "Duplicate/absent CSV headers")
        rows = list(reader)
        require(all(None not in r and None not in r.values() for r in rows), "Ragged CSV row")
        return tuple(reader.fieldnames), rows


def load_episodes(path):
    headers, rows = read_csv(path)
    require(headers == EPISODE_FIELDS, "Unexpected episode columns")
    for row in rows:
        for key in ("num_agents", "seed") + COUNTS + ("completed_targets",):
            value = row[key]
            require(value.isdigit() and str(int(value)) == value, f"Non-canonical integer in {key}")
            row[key] = int(value)
        require(sum(row[k] for k in COUNTS) == row["completed_targets"], "Window/episode goal-count mismatch")
        require(close(row["avg_throughput"], row["completed_targets"] / 4096), "Wrong episode throughput")
        row["avg_throughput"] = float(row["avg_throughput"])
    keys = [tuple(row[k] for k in KEY) for row in rows]
    expected = set(itertools.product(METHODS, RULES, MAPS, POPS, SEEDS))
    require(len(keys) == len(set(keys)) == 288 and set(keys) == expected, "Incomplete/duplicate/wrong 288-episode grid")
    return rows


def aggregate(rows, population_specific=False):
    windows, tails = [], []
    for method, rule in itertools.product(METHODS, RULES):
        for population in POPS if population_specific else (None,):
            group = [r for r in rows if r["method"] == method and r["collision_system"] == rule
                     and (population is None or r["num_agents"] == population)]
            fields = dict(method=method, collision_system=rule)
            if population_specific:
                fields["num_agents"] = population
            means = [sum(r[k] for r in group) / (512 * len(group)) for k in COUNTS]
            for i, mean in enumerate(means):
                windows.append(dict(**fields, window=i + 1, start_step=512 * i + 1,
                                    end_step=512 * (i + 1), episodes=len(group), mean_window_throughput=mean))
            differences = [r[COUNTS[-1]] - r[COUNTS[0]] for r in group]
            tails.append(dict(**fields, episodes=len(group), first_window=means[0], last_window=means[-1],
                              last_minus_first=means[-1] - means[0],
                              relative_change_percent=100 * (means[-1] / means[0] - 1),
                              decreased_episodes=sum(d < 0 for d in differences),
                              unchanged_episodes=sum(d == 0 for d in differences),
                              increased_episodes=sum(d > 0 for d in differences)))
    return windows, tails


def compare_csv(path, expected):
    headers, actual = read_csv(path)
    require(headers == tuple(expected[0]), f"Wrong aggregate columns: {path.name}")
    require(len(actual) == len(expected), f"Wrong aggregate row count: {path.name}")
    identity = ("method", "collision_system") + (("num_agents",) if "num_agents" in headers else ())
    identity += ("window",) if "window" in headers else ()
    key = lambda row: tuple(str(row[k]) for k in identity)
    require(len({key(r) for r in actual}) == len(actual), f"Duplicate aggregate: {path.name}")
    actual = {key(r): r for r in actual}
    require(set(actual) == {key(r) for r in expected}, f"Wrong aggregate identities: {path.name}")
    for row in expected:
        observed = actual[key(row)]
        for field, value in row.items():
            valid = observed[field] == str(value) if isinstance(value, (str, int)) else close(observed[field], value)
            require(valid, f"Recomputed aggregate differs: {path.name} {key(row)} {field}")


def verify(data_dir):
    data_dir = Path(data_dir).resolve()
    report = json.loads((data_dir / "PROVENANCE.json").read_text(encoding="utf-8"))
    require(report.get("schema") == "public_long_horizon_counts_v1", "Wrong provenance schema")
    require(report.get("original_collection_sha256") == COLLECTION_SHA, "Wrong historical collection identity")
    expected_protocol = dict(methods=list(METHODS), collision_systems=list(RULES), maps=list(MAPS),
                             populations=list(POPS), seeds=list(SEEDS), max_steps=4096, window_steps=512,
                             obs_radius=5, on_target="restart", episodes=288, episode_windows=2304,
                             weights_trained_under="block_both", caar_entropy_gate=True,
                             srslm_static_step_occupancy_check="enabled_under_both_rules")
    require(report.get("protocol") == expected_protocol, "Published protocol changed")
    # Paths are fixed here: the manifest cannot request arbitrary filesystem reads.
    require(set(report.get("data_sha256", {})) == set(DATA_FILES), "Incomplete data hash manifest")
    for name in DATA_FILES:
        require(sha(data_dir / name) == report["data_sha256"][name], f"Data hash mismatch: {name}")
    require(set(report.get("figure_sha256", {})) == set(FIGURES), "Incomplete figure hash manifest")
    for name in FIGURES:
        require(sha(data_dir.parent.parent / "assets" / name) == report["figure_sha256"][name], f"Figure hash mismatch: {name}")
    rows = load_episodes(data_dir / DATA_FILES[0])
    overall, tail = aggregate(rows)
    population, population_tail = aggregate(rows, True)
    for name, expected in zip(AGGREGATES, (overall, population, tail, population_tail)):
        compare_csv(data_dir / name, expected)
    means = []
    for method, rule in itertools.product(METHODS, RULES):
        group = [r for r in rows if r["method"] == method and r["collision_system"] == rule]
        means.append(dict(method=method, collision_system=rule, episodes=len(group),
                          mean_throughput=sum(r["completed_targets"] for r in group) / (4096 * len(group))))
    require(report.get("means") == means, "Published means differ from goal counts")
    return dict(validated=True, episodes=len(rows), episode_windows=len(rows) * 8,
                aggregate_rows=dict(zip(AGGREGATES, map(len, (overall, population, tail, population_tail)))),
                means=means,
                scope="Published counts, aggregate arithmetic and release hashes only; private journals/runtime provenance are not re-audited.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "docs/data/long_horizon_20260908")
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.data_dir), indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError, TypeError, ZeroDivisionError) as exc:
        raise SystemExit(f"Release verification failed: {exc}")


if __name__ == "__main__":
    main()
