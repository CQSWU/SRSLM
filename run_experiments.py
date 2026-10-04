import argparse

import json

import logging

import multiprocessing

import os

import random

import time

from concurrent.futures import ProcessPoolExecutor, as_completed

from contextlib import suppress

from datetime import datetime

from pathlib import Path

import numpy as np
import torch

from agents.utils_agents import ResultsHolder
from pogema.svg_animation.animation_wrapper import (
    AnimationConfig,
    AnimationMonitor,
)
from pomapf_env.env import make_pomapf
from pomapf_env.pomapf_config import POMAPFConfig

DEFAULT_MAPS = {
    "mazes": "mazes-s0_wc8_od55",
    "random": "random-s0_d0.15",
    "sc1": "sc1-AcrosstheCape",
    "street": "street-Berlin_0",
    "wc3": "wc3-Battleground",
}

SUPPORTED_ALGORITHMS = ("AORePlan", "ARPE", "SRSLM")
DEFAULT_ALGORITHMS = ("AORePlan",)
ALGORITHM_ALIASES = {
    "aoreplan": "AORePlan",
    "arpe": "ARPE",
    "srslm": "SRSLM",
}

ALGORITHM_COLUMN_WIDTH = max(
    13, *(len(algorithm) for algorithm in SUPPORTED_ALGORITHMS)
)

_worker_algo_cache = {}


def canonical_algorithm_name(value):

    return ALGORITHM_ALIASES.get(value.strip().lower())


def quiet_model_logs():

    logging.getLogger("rl").setLevel(logging.ERROR)

    with suppress(Exception):
        from sample_factory.utils.utils import log

        log.setLevel(logging.ERROR)


def _has_config(path):

    return (path / "config.json").exists()


def _has_checkpoints(path):

    for d in path.glob("checkpoint_p*"):
        if d.is_dir() and any(d.glob("*.pth")):
            return True

    return False


def _find_switcher_weights(main_dir):
    root = Path(main_dir).resolve()
    candidate = root / "weights" / "SRSLM-Switcher-Final-1B"
    if not _has_config(candidate) or not _has_checkpoints(candidate):
        raise FileNotFoundError(
            "The selected final 1B SRSLM Switcher checkpoint is missing."
        )
    return str(candidate)


def _project_path(main_dir, value):
    path = Path(value)
    if not path.is_absolute():
        path = Path(main_dir) / path
    return path.resolve()


def _load_arpe_candidate_artifact(main_dir, manifest_path):

    if manifest_path is None:
        manifest_path = str(
            Path(main_dir).resolve() / "configs" / "arpe_final_candidate.json"
        )
    path = _project_path(main_dir, manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"ARPE candidate manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ARPE candidate manifest must be a JSON object.")
    from agents.arpe import ArpeCandidateArtifact

    artifact = ArpeCandidateArtifact.from_mapping(
        payload,
        Path(main_dir).resolve(),
    )
    return artifact


_EPISODE_FRESH_ALGORITHMS = frozenset(("ARPE", "SRSLM"))


def should_cache_algorithm(algorithm, requested):
    canonical = canonical_algorithm_name(algorithm) or algorithm
    return bool(requested) and canonical not in _EPISODE_FRESH_ALGORITHMS


def _ao_replan_cfg(
    seed,
    max_planning_steps=10000,
):

    from agents.ao_replan import AORePlanConfig

    return AORePlanConfig(
        name="AORePlan",
        max_planning_steps=max_planning_steps,
        seed=seed,
    )


def build_algorithm(
    algo_name,
    main_dir,
    seed,
    arpe_candidate_manifest=None,
    switcher_weights_path=None,
):

    algo_name = canonical_algorithm_name(algo_name) or algo_name
    if algo_name not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unsupported public algorithm: {algo_name}")

    if algo_name == "AORePlan":
        from agents.ao_replan import AORePlan

        return AORePlan(_ao_replan_cfg(seed))

    if algo_name == "SRSLM":
        from agents.srslm import SRSLM, SRSLMConfig
        from agents.switcher import SwitcherConfig
        from agents.arpe import ARPEConfig

        artifact = _load_arpe_candidate_artifact(main_dir, arpe_candidate_manifest)
        candidate = ARPEConfig(
            path_to_weights=str(artifact.weights_path),
            milestone_checkpoint=str(artifact.checkpoint_path),
            base_weights_path=str(artifact.base_weights_path),
            base_checkpoint_path=str(artifact.base_checkpoint_path),
            inference=artifact.inference.as_dict() if artifact.inference else None,
        )

        policy = SRSLM(
            SRSLMConfig(
                candidate=candidate,
                switcher=SwitcherConfig(
                    path_to_weights=str(
                        _project_path(
                            main_dir,
                            switcher_weights_path or _find_switcher_weights(main_dir),
                        )
                    ),
                    checkpoint_kind="auto",
                    device="auto",
                    deterministic=False,
                ),
                seed=seed,
            ),
            project_root=Path(main_dir).resolve(),
        )
        return policy

    if algo_name == "ARPE":
        from agents.arpe import ARPE

        manifest = arpe_candidate_manifest or str(
            Path(main_dir).resolve() / "configs" / "arpe_final_candidate.json"
        )
        artifact = _load_arpe_candidate_artifact(main_dir, manifest)
        return ARPE.load(
            artifact,
            seed=int(seed),
            device="auto",
            action_sampling="direct_numpy",
        )

    raise ValueError(f"Unsupported algorithm: {algo_name}")


class _MoveFailureTracker:
    METRIC_VERSION = "submitted_nonwait_no_position_change_v1"
    CONTENTION_METRIC_VERSION = "agent_contention_participation_v1"
    VERTEX_FLOW_METRIC_VERSION = "submitted_one_step_vertex_flow_pairs_v1"

    def __init__(self, moves, obstacle_mask=None):
        self.moves = tuple(tuple(int(value) for value in move) for move in moves)
        self.obstacle_mask = (
            None
            if obstacle_mask is None
            else np.asarray(obstacle_mask, dtype=bool).copy()
        )
        self.environment_step_count = 0
        self.active_agent_step_count = 0
        self.wait_action_count = 0
        self.move_attempt_count = 0
        self.successful_move_count = 0
        self.conflict_count = 0
        self.agent_conflict_count = 0
        self.other_or_unattributed_conflict_count = 0
        self.conflict_step_count = 0
        self.contention_participant_count = 0
        self.contention_step_count = 0
        self.vertex_flow_pair_count = 0
        self.vertex_flow_move_denominator = 0
        self.vertex_flow_contested_destination_count = 0
        self.vertex_flow_step_count = 0
        self.vertex_flow_max_inflow = 0

    @staticmethod
    def _point(observation):
        value = observation["xy"]
        return int(value[0]), int(value[1])

    def _is_valid_flow_target(self, target):
        if self.obstacle_mask is None:
            return True
        x, y = target
        height, width = self.obstacle_mask.shape
        return 0 <= x < height and 0 <= y < width and not bool(self.obstacle_mask[x, y])

    def capture(
        self,
        actions,
        observations,
        dones,
        infos,
        global_positions=None,
    ):
        positions = (
            [tuple(int(value) for value in position) for position in global_positions]
            if global_positions is not None
            else [self._point(observation) for observation in observations]
        )
        active = [
            not bool(dones[index]) and bool(infos[index].get("is_active", True))
            for index in range(len(observations))
        ]
        attempts = []
        flow_attempts = []
        waits = 0
        for index, is_active in enumerate(active):
            if not is_active:
                continue
            try:
                action = int(actions[index])
                delta = self.moves[action]
            except (IndexError, TypeError, ValueError):
                waits += 1
                continue
            if delta == (0, 0):
                waits += 1
                continue
            position = positions[index]
            target = position[0] + delta[0], position[1] + delta[1]
            attempts.append((index, position, target))
            if self._is_valid_flow_target(target):
                flow_attempts.append((index, position, target))
        return {
            "positions": positions,
            "active": active,
            "active_count": sum(active),
            "wait_count": waits,
            "attempts": attempts,
            "flow_attempts": flow_attempts,
        }

    def commit(self, pending, observations, global_positions=None):
        after_positions = (
            [tuple(int(value) for value in position) for position in global_positions]
            if global_positions is not None
            else [self._point(observation) for observation in observations]
        )
        occupied = {}
        for index, position in enumerate(pending["positions"]):
            occupied.setdefault(position, set()).add(index)
        target_counts = {}
        for _, _, target in pending["attempts"]:
            target_counts[target] = target_counts.get(target, 0) + 1

        attempts_by_agent = {
            index: (before, target) for index, before, target in pending["attempts"]
        }
        contention_participants = set()

        target_groups = {}
        for index, _, target in pending["attempts"]:
            target_groups.setdefault(target, []).append(index)
        for group in target_groups.values():
            if len(group) > 1:
                contention_participants.update(group)

        flow_target_counts = {}
        for _, _, target in pending["flow_attempts"]:
            flow_target_counts[target] = flow_target_counts.get(target, 0) + 1
        vertex_flow_pairs = sum(
            count * (count - 1) // 2 for count in flow_target_counts.values()
        )
        contested_flow_destinations = sum(
            count > 1 for count in flow_target_counts.values()
        )
        max_inflow = max(flow_target_counts.values(), default=0)

        position_to_agents = {}
        for index, position in enumerate(pending["positions"]):
            position_to_agents.setdefault(position, []).append(index)

        for index, before, target in pending["attempts"]:
            for other in position_to_agents.get(target, ()):
                if other == index:
                    continue
                other_attempt = attempts_by_agent.get(other)
                if other_attempt is not None and other_attempt[1] == before:
                    contention_participants.add(index)
                    if pending["active"][other]:
                        contention_participants.add(other)
                if after_positions[other] == pending["positions"][other]:
                    contention_participants.add(index)
                    if pending["active"][other]:
                        contention_participants.add(other)

        failed_this_step = 0
        agent_conflicts = 0
        successes = 0
        for index, before, target in pending["attempts"]:
            if after_positions[index] != before:
                successes += 1
                continue
            failed_this_step += 1
            occupied_by_other = any(
                other != index for other in occupied.get(target, ())
            )
            if occupied_by_other or target_counts[target] > 1:
                agent_conflicts += 1

        self.environment_step_count += 1
        self.active_agent_step_count += pending["active_count"]
        self.wait_action_count += pending["wait_count"]
        self.move_attempt_count += len(pending["attempts"])
        self.successful_move_count += successes
        self.conflict_count += failed_this_step
        self.agent_conflict_count += agent_conflicts
        self.other_or_unattributed_conflict_count += failed_this_step - agent_conflicts
        if failed_this_step:
            self.conflict_step_count += 1
        self.contention_participant_count += len(contention_participants)
        if contention_participants:
            self.contention_step_count += 1
        self.vertex_flow_pair_count += vertex_flow_pairs
        self.vertex_flow_move_denominator += len(pending["flow_attempts"])
        self.vertex_flow_contested_destination_count += contested_flow_destinations
        if vertex_flow_pairs:
            self.vertex_flow_step_count += 1
        self.vertex_flow_max_inflow = max(
            self.vertex_flow_max_inflow,
            max_inflow,
        )

    @staticmethod
    def _rate(numerator, denominator):
        return float(numerator / denominator) if denominator else 0.0

    def metrics(self):
        return {
            "congestion_metric_version": self.METRIC_VERSION,
            "environment_step_count_observed": self.environment_step_count,
            "active_agent_step_count": self.active_agent_step_count,
            "wait_action_count": self.wait_action_count,
            "move_attempt_count": self.move_attempt_count,
            "successful_move_count": self.successful_move_count,
            "conflict_count": self.conflict_count,
            "move_failure_count": self.conflict_count,
            "agent_conflict_count": self.agent_conflict_count,
            "other_or_unattributed_conflict_count": (
                self.other_or_unattributed_conflict_count
            ),
            "conflict_step_count": self.conflict_step_count,
            "contention_metric_version": self.CONTENTION_METRIC_VERSION,
            "contention_participant_count": self.contention_participant_count,
            "contention_step_count": self.contention_step_count,
            "contention_participation_rate": self._rate(
                self.contention_participant_count,
                self.active_agent_step_count,
            ),
            "contention_step_rate": self._rate(
                self.contention_step_count,
                self.environment_step_count,
            ),
            "vertex_flow_metric_version": self.VERTEX_FLOW_METRIC_VERSION,
            "vertex_flow_pair_count": self.vertex_flow_pair_count,
            "vertex_flow_move_denominator": self.vertex_flow_move_denominator,
            "vertex_flow_pair_cost_per_move": self._rate(
                self.vertex_flow_pair_count,
                self.vertex_flow_move_denominator,
            ),
            "vertex_flow_contested_destination_count": (
                self.vertex_flow_contested_destination_count
            ),
            "vertex_flow_step_count": self.vertex_flow_step_count,
            "vertex_flow_max_inflow": self.vertex_flow_max_inflow,
            "congestion_rate": self._rate(
                self.conflict_count,
                self.move_attempt_count,
            ),
            "agent_conflict_rate": self._rate(
                self.agent_conflict_count,
                self.move_attempt_count,
            ),
            "conflict_agent_step_rate": self._rate(
                self.conflict_count,
                self.active_agent_step_count,
            ),
            "conflict_step_rate": self._rate(
                self.conflict_step_count,
                self.environment_step_count,
            ),
        }


def run_algorithm(
    algo,
    *,
    map_name,
    max_episode_steps,
    seed,
    num_agents,
    obs_radius,
    animate,
    on_target,
    collision_system,
    map_text,
    agents_xy=None,
    targets_xy=None,
):

    gc_kwargs = {
        "max_episode_steps": max_episode_steps,
        "seed": seed,
        "num_agents": num_agents,
        "on_target": on_target,
    }
    if obs_radius is not None:
        gc_kwargs["obs_radius"] = obs_radius
    if collision_system is not None:
        gc_kwargs["collision_system"] = collision_system
    if map_text is not None:
        gc_kwargs["map"] = map_text
        gc_kwargs["map_name"] = None
    else:
        gc_kwargs["map_name"] = map_name

    if (agents_xy is None) != (targets_xy is None):
        raise ValueError("Explicit placements require both agents_xy and targets_xy")
    if agents_xy is not None:
        if len(agents_xy) != num_agents or len(targets_xy) != num_agents:
            raise ValueError("Explicit placement count must equal num_agents")
        gc_kwargs["agents_xy"] = [list(position) for position in agents_xy]
        gc_kwargs["targets_xy"] = [list(position) for position in targets_xy]

    grid_config = POMAPFConfig(**gc_kwargs)

    env = make_pomapf(
        grid_config=grid_config,
        with_animations=False,
        auto_reset=False,
    )
    if animate:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        map_short = map_name.split("-")[-1] if "-" in map_name else map_name
        directory = Path("renders") / (
            f"{map_short}_{type(algo).__name__}_{num_agents}agents_{timestamp}"
        )
        env = AnimationMonitor(
            env,
            AnimationConfig(directory=str(directory)),
        )

    def environment_grid():
        current = env
        while current is not None:
            grid = getattr(current, "grid", None)
            if grid is not None:
                return grid
            current = getattr(current, "env", None)
        raise RuntimeError("Could not locate the environment grid")

    def global_positions():
        grid = environment_grid()
        positions = getattr(grid, "positions_xy", None)
        if positions is None and hasattr(grid, "get_agents_xy"):
            positions = grid.get_agents_xy()
        if positions is not None:
            return np.asarray(positions, dtype=np.int64).copy()
        raise RuntimeError(
            "Contention participation requires global agent positions from "
            "the environment grid."
        )

    def global_obstacles():
        grid = environment_grid()
        obstacles = getattr(grid, "obstacles", None)
        if obstacles is not None:
            return np.asarray(obstacles, dtype=bool).copy()
        raise RuntimeError(
            "Vertex flow evaluation requires the environment obstacle grid."
        )

    try:
        observations, _ = env.reset()
        if agents_xy is not None:
            grid = environment_grid()
            expected_agents = np.asarray(agents_xy, dtype=np.int64)
            if (
                targets_xy
                and isinstance(targets_xy[0], (list, tuple))
                and targets_xy[0]
                and isinstance(targets_xy[0][0], (list, tuple))
            ):
                expected_targets = np.asarray(
                    [sequence[0] for sequence in targets_xy],
                    dtype=np.int64,
                )
            else:
                expected_targets = np.asarray(targets_xy, dtype=np.int64)
            actual_agents = np.asarray(
                grid.get_agents_xy(ignore_borders=True),
                dtype=np.int64,
            )
            actual_targets = np.asarray(
                grid.get_targets_xy(ignore_borders=True),
                dtype=np.int64,
            )
            expected_obstacles = np.asarray(
                [
                    [character == "#" for character in row]
                    for row in map_text.splitlines()
                ],
                dtype=bool,
            )
            actual_obstacles = np.asarray(
                grid.get_obstacles(ignore_borders=True),
                dtype=bool,
            )
            if not np.array_equal(actual_agents, expected_agents):
                raise RuntimeError("Environment changed the explicit start coordinates")
            if not np.array_equal(actual_targets, expected_targets):
                raise RuntimeError("Environment changed the explicit goal coordinates")
            if not np.array_equal(actual_obstacles, expected_obstacles):
                raise RuntimeError("Environment changed the explicit obstacle grid")
            padding = int(grid_config.obs_radius)
            if not np.array_equal(
                np.asarray(grid.positions_xy, dtype=np.int64),
                expected_agents + padding,
            ):
                raise RuntimeError("Unexpected padded start coordinates")
            if not np.array_equal(
                np.asarray(grid.finishes_xy, dtype=np.int64),
                expected_targets + padding,
            ):
                raise RuntimeError("Unexpected padded goal coordinates")
        tracker = _MoveFailureTracker(
            grid_config.MOVES,
            obstacle_mask=global_obstacles(),
        )
        algo.after_reset()
        if hasattr(algo, "set_grid_config"):
            algo.set_grid_config(env.grid_config)
        if hasattr(algo, "set_env"):
            algo.set_env(env)
        results_holder = ResultsHolder()
        dones = [False for _ in observations]
        infos = [{"is_active": True} for _ in observations]
        rewards = [0 for _ in observations]
        decision_seconds = 0.0
        decision_calls = 0
        completed_targets = 0
        previous_segment_targets = 0
        previous_segment_end = 0
        throughput_segments = []
        with torch.no_grad():
            while True:
                decision_started = time.perf_counter()
                actions = algo.act(observations, rewards, dones, infos)
                decision_seconds += time.perf_counter() - decision_started
                decision_calls += 1
                pending = tracker.capture(
                    actions,
                    observations,
                    dones,
                    infos,
                    global_positions=global_positions(),
                )
                observations, rewards, terminated, truncated, infos = env.step(actions)

                if on_target == "restart":
                    completed_targets += int(sum(env.unwrapped.was_on_goal))
                tracker.commit(
                    pending,
                    observations,
                    global_positions=global_positions(),
                )
                dones = [
                    terminated_value or truncated_value
                    for terminated_value, truncated_value in zip(
                        terminated,
                        truncated,
                    )
                ]
                results_holder.after_step(infos)
                algo.after_step(dones)
                if on_target == "restart" and (decision_calls % 512 == 0 or all(dones)):
                    segment_steps = decision_calls - previous_segment_end
                    segment_targets = completed_targets - previous_segment_targets
                    throughput_segments.append(
                        {
                            "start_step": previous_segment_end + 1,
                            "end_step": decision_calls,
                            "step_count": segment_steps,
                            "completed_targets": segment_targets,
                            "throughput": segment_targets / segment_steps,
                            "cumulative_throughput": completed_targets / decision_calls,
                        }
                    )
                    previous_segment_end = decision_calls
                    previous_segment_targets = completed_targets
                if all(dones):
                    break
        results = results_holder.get_final()
        results.update(tracker.metrics())
        results["policy_decision_seconds"] = decision_seconds
        results["policy_decision_calls"] = decision_calls
        results["policy_decision_ms_per_joint_action"] = (
            1000.0 * decision_seconds / decision_calls if decision_calls else None
        )
        results["policy_decision_timing_scope"] = "algo.act_only_perf_counter_v1"
        results["throughput_segments"] = throughput_segments
        results["completed_targets_observed"] = (
            completed_targets if on_target == "restart" else None
        )
        if on_target == "restart":
            reported_targets = float(results["avg_throughput"]) * max_episode_steps
            if (
                not np.isfinite(reported_targets)
                or abs(reported_targets - completed_targets) > 1e-6
            ):
                raise RuntimeError(
                    "Segment goal events disagree with POGEMA throughput"
                )
        results["algorithm"] = type(algo).__name__
        return results
    finally:
        env.close()


def run_single_experiment(task):

    quiet_model_logs()

    algo_name = canonical_algorithm_name(task["algorithm"]) or task["algorithm"]

    main_dir = task["main_dir"]

    seed = task["seed"]

    random.seed(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)

    use_cache = should_cache_algorithm(
        algo_name,
        task.get("cache_algorithms", False),
    )

    cache_key = (
        algo_name,
        str(Path(main_dir).resolve()),
        seed,
        task.get("arpe_candidate_manifest"),
        task.get("switcher_weights_path"),
    )

    try:
        if use_cache and cache_key in _worker_algo_cache:
            algo = _worker_algo_cache[cache_key]
        else:
            algo = build_algorithm(
                algo_name,
                main_dir,
                seed,
                arpe_candidate_manifest=task.get("arpe_candidate_manifest"),
                switcher_weights_path=task.get("switcher_weights_path"),
            )
            if use_cache:
                _worker_algo_cache[cache_key] = algo

        start = time.time()

        result = run_algorithm(
            algo,
            map_name=task["map_name"],
            max_episode_steps=task["max_steps"],
            seed=seed,
            num_agents=task["num_agents"],
            obs_radius=task.get("obs_radius"),
            animate=task["animate"],
            on_target=task.get("on_target", "restart"),
            collision_system=task.get("collision_system"),
            map_text=task.get("map_text"),
            agents_xy=task.get("agents_xy"),
            targets_xy=task.get("targets_xy"),
        )

        run_time = time.time() - start

        on_target = task.get("on_target", "restart")
        is_restart = on_target == "restart"
        is_replan = algo_name == "AORePlan"

        if hasattr(algo, "get_switch_stats"):
            hybrid_stats = algo.get_switch_stats()
        else:
            hybrid_stats = {}
        result_record = {
            "algorithm": algo_name,
            "map_name": task["map_name"],
            "num_agents": task["num_agents"],
            "max_steps": task["max_steps"],
            "seed": seed,
            "on_target": on_target,
            "run_time_seconds": run_time,
        }
        if task.get("task_id") is not None:
            result_record.update(
                {
                    "task_id": task["task_id"],
                    "family_id": task["family_id"],
                    "density_percent": task["density_percent"],
                }
            )
        result_record.update(
            {key: value for key, value in result.items() if key != "algorithm"}
        )

        result_record.update(hybrid_stats)

        if is_restart and is_replan:
            for key in (
                "reverse_action_rate",
                "reverse_action_count",
                "reverse_action_denominator",
                "reverse_metric_version",
                "static_astar_query_count",
                "static_astar_query_denominator",
                "static_astar_query_rate",
                "no_path_fallback_count",
            ):
                result_record[key] = getattr(algo, key, None)

        return result_record

    except Exception as exc:
        import traceback

        error_record = {
            "algorithm": algo_name,
            "map_name": task["map_name"],
            "num_agents": task["num_agents"],
            "max_steps": task["max_steps"],
            "seed": seed,
            "on_target": task.get("on_target", "restart"),
            "avg_throughput": None,
            "run_time_seconds": 0.0,
            "policy_decision_seconds": None,
            "policy_decision_calls": 0,
            "policy_decision_ms_per_joint_action": None,
            "policy_decision_timing_scope": "algo.act_only_perf_counter_v1",
            "throughput_segments": [],
            "completed_targets_observed": None,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        if task.get("task_id") is not None:
            error_record.update(
                {
                    "task_id": task["task_id"],
                    "family_id": task["family_id"],
                    "density_percent": task["density_percent"],
                }
            )
        return error_record


def parse_algorithms(value):

    if value.strip().lower() == "all":
        return list(DEFAULT_ALGORITHMS)

    raw_algorithms = [item.strip() for item in value.split(",") if item.strip()]

    algorithms = []

    seen = set()

    unknown = []

    for item in raw_algorithms:
        canonical = canonical_algorithm_name(item)

        if canonical is None:
            unknown.append(item)

        else:
            if canonical not in seen:
                algorithms.append(canonical)

                seen.add(canonical)

    if unknown:
        choices = ", ".join(SUPPORTED_ALGORITHMS)

        raise argparse.ArgumentTypeError(
            f"Unknown algorithm(s): {unknown}. Choices: {choices}"
        )

    if not algorithms:
        raise argparse.ArgumentTypeError("No algorithms selected")

    return algorithms


def parse_agent_counts(args):

    if args.agents:
        counts = [int(item.strip()) for item in args.agents.split(",") if item.strip()]

    else:
        if args.agent_step <= 0:
            raise ValueError("--agent-step must be positive")

        counts = list(range(args.agent_start, args.agent_stop + 1, args.agent_step))

    counts = sorted(set(counts))

    if not counts:
        raise ValueError("No agent counts selected")

    if any(count <= 0 for count in counts):
        raise ValueError("Agent counts must all be positive")

    return counts


def parse_seeds(args):

    if args.seeds:
        seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]

        seeds = sorted(set(seeds))

        if not seeds:
            raise ValueError("No seeds selected")

        return seeds

    return [args.seed]


def parse_maps(map_types_value, map_overrides):

    if map_types_value.strip().lower() == "custom":
        maps = {}

        for item in map_overrides:
            map_name = item.strip()

            if not map_name:
                continue

            maps[map_name] = map_name

        if not maps:
            raise ValueError(
                "When --map-types=custom, --map must provide at least one map name"
            )

        return maps

    maps = dict(DEFAULT_MAPS)

    for item in map_overrides:
        if "=" not in item:
            raise ValueError("--map must use the format map_type=map_name")

        map_type, map_name = item.split("=", 1)

        map_type = map_type.strip()

        map_name = map_name.strip()

        if not map_type or not map_name:
            raise ValueError("--map must use non-empty map_type=map_name values")

        maps[map_type] = map_name

    if map_types_value.strip().lower() == "all":
        selected_types = list(DEFAULT_MAPS.keys())

    else:
        selected_types = [
            item.strip() for item in map_types_value.split(",") if item.strip()
        ]

    unknown = [item for item in selected_types if item not in maps]

    if unknown:
        raise ValueError(f"Unknown map type(s): {unknown}. Available: {sorted(maps)}")

    if not selected_types:
        raise ValueError("No map types selected")

    return {map_type: maps[map_type] for map_type in selected_types}


def _looks_like_movingai_map(lines):

    if not lines:
        return False

    head = [line.strip().lower() for line in lines[:4]]

    return "map" in head and any(line.startswith("type ") for line in head)


def _translate_map_rows(rows):

    trans = {".": ".", "G": ".", "S": ".", "W": "#", "T": "#", "@": "#", "O": "#"}

    return [
        "".join(trans.get(ch, "#") for ch in row.rstrip())
        for row in rows
        if row.strip()
    ]


def load_map_text(path_or_url, trim_border=False):

    path_or_url = str(path_or_url)

    raw_text = Path(path_or_url).read_text()

    source = str(Path(path_or_url).resolve())

    lines = raw_text.splitlines()

    if _looks_like_movingai_map(lines):
        map_start = (
            next(i for i, line in enumerate(lines) if line.strip().lower() == "map") + 1
        )

        rows = _translate_map_rows(lines[map_start:])

    else:
        rows = [line.rstrip() for line in lines if line.strip()]

    if trim_border and len(rows) >= 3 and len(rows[0]) >= 3:
        rows = [row[1:-1] for row in rows[1:-1]]

    if not rows:
        raise ValueError(f"Map source produced no rows: {path_or_url}")

    width = len(rows[0])

    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError(
            f"Map source must contain a non-empty rectangular grid: {path_or_url}"
        )

    label = Path(path_or_url).name

    return {
        "map_name": label or "custom-map",
        "map_text": "\n".join(rows),
        "map_source": source,
        "map_size": [len(rows[0]), len(rows)],
    }


def load_map_list_snapshot(path, registry_path=None):

    import yaml

    path = Path(path).resolve()

    payload = path.read_bytes()

    try:
        data = yaml.safe_load(payload.decode("utf-8"))

    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError(f"Could not parse map list {path}: {error}") from error

    if not isinstance(data, dict) or not data:
        raise ValueError("--map-list must contain a non-empty YAML mapping")

    if any(not isinstance(name, str) or not name for name in data):
        raise ValueError("--map-list keys must be non-empty strings")

    registry = data

    if any(not isinstance(value, str) or not value.strip() for value in data.values()):
        if registry_path is None:
            raise ValueError("--map-list entries without grid text require a registry")

        registry_path = Path(registry_path).resolve()

        registry_payload = registry_path.read_bytes()

        try:
            registry = yaml.safe_load(registry_payload.decode("utf-8"))

        except (UnicodeDecodeError, yaml.YAMLError) as error:
            raise ValueError(
                f"Could not parse map registry {registry_path}: {error}"
            ) from error

        if not isinstance(registry, dict) or not registry:
            raise ValueError(
                f"Map registry must be a non-empty mapping: {registry_path}"
            )

    map_texts = {}

    for name, selected_value in data.items():
        value = (
            selected_value
            if isinstance(selected_value, str) and selected_value.strip()
            else registry.get(name)
        )

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"No grid text found for map {name!r}")

        rows = value.splitlines()

        if (
            not rows
            or len({len(row) for row in rows}) != 1
            or not rows[0]
            or any(set(row) - {".", "#"} for row in rows)
        ):
            raise ValueError(f"Map {name!r} must be a non-empty rectangular .# grid")

        map_texts[name] = value

    return {name: name for name in data}, map_texts


def build_tasks(
    algorithms,
    maps,
    agent_counts,
    seeds,
    args,
    custom_map=None,
    map_texts=None,
):

    map_items = (
        [
            (
                "custom",
                custom_map["map_name"],
                custom_map["map_text"],
                custom_map["map_source"],
            )
        ]
        if custom_map is not None
        else [
            (
                map_type,
                map_name,
                map_texts.get(map_name) if map_texts is not None else None,
                None,
            )
            for map_type, map_name in maps.items()
        ]
    )

    return [
        {
            "algorithm": algorithm,
            "map_type": map_type,
            "map_name": map_name,
            "map_text": map_text,
            "map_source": map_source,
            "num_agents": num_agents,
            "obs_radius": args.obs_radius,
            "max_steps": args.max_steps,
            "seed": seed,
            "animate": args.animate,
            "main_dir": args.main_dir,
            "on_target": args.on_target,
            "collision_system": args.collision_system,
            "arpe_candidate_manifest": getattr(args, "arpe_candidate_manifest", None),
            "switcher_weights_path": args.switcher_weights_path,
            "cache_algorithms": should_cache_algorithm(
                algorithm,
                args.cache_algorithms,
            ),
        }
        for algorithm in algorithms
        for map_type, map_name, map_text, map_source in map_items
        for num_agents in agent_counts
        for seed in seeds
    ]


def format_duration(seconds):

    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes, rem = divmod(seconds, 60)

    if minutes < 60:
        return f"{int(minutes)}m{int(rem):02d}s"

    hours, minutes = divmod(minutes, 60)

    return f"{int(hours)}h{int(minutes):02d}m"


def run_experiments(tasks, workers):
    results = []
    total = len(tasks)
    start_time = time.time()
    print(f"Starting experiments: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"Total: {total} | Workers: {workers}")

    print("-" * 110, flush=True)

    if not tasks:
        return results, 0.0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_single_experiment, task) for task in tasks]

        for index, future in enumerate(as_completed(futures), start=len(results) + 1):
            result = future.result()

            elapsed = time.time() - start_time

            eta = elapsed / index * (total - index) if index else 0.0

            result["completed_index"] = index

            result["total_experiments"] = total

            result["elapsed_since_start_seconds"] = elapsed

            result["eta_after_result_seconds"] = eta

            result["finished_at"] = datetime.now().isoformat(timespec="seconds")

            results.append(result)

            if result.get("error"):
                status = f"ERROR: {result['error']}"

            else:
                on_target = result.get("on_target", "restart")

                diag_str = ""

                if result.get("congestion_rate") is not None:
                    diag_str += f" congestion={result['congestion_rate']:.1%}"

                if on_target != "restart":
                    if result.get("ep_length") is not None:
                        diag_str += f" ep_len={result['ep_length']:.1f}"

                    isr = result.get("ISR")

                    csr = result.get("CSR")

                    status = (
                        f"isr={(0.0 if isr is None else isr):.1%} "
                        f"csr={(0.0 if csr is None else csr):.1%}{diag_str} "
                        f"run={format_duration(result['run_time_seconds'])}"
                    )

                else:
                    if result.get("reverse_action_rate") is not None:
                        diag_str += f" rev={result['reverse_action_rate']:.1%}"

                    status = (
                        f"throughput={result['avg_throughput']:.4f}{diag_str} "
                        f"run={format_duration(result['run_time_seconds'])}"
                    )

            print(
                f"[{index:>3}/{total:<3}] {result['algorithm']:<{ALGORITHM_COLUMN_WIDTH}} | "
                f"{result['map_name']:<22} | {result['num_agents']:>3} agents | "
                f"{status:<34}",
                flush=True,
            )

    return results, time.time() - start_time


def save_results(results, metadata, output_dir, filename=None):

    os.makedirs(output_dir, exist_ok=True)

    if filename:
        output_path = Path(output_dir) / filename

    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_path = Path(output_dir) / f"experiments_{timestamp}.json"

    payload = {
        "metadata": metadata,
        "results": results,
    }

    temporary_path = output_path.with_name(f".{output_path.name}.tmp")

    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

        f.flush()

        os.fsync(f.fileno())

    temporary_path.replace(output_path)

    print(f"\nResults saved: {output_path}")

    return output_path


def parse_args():

    parser = argparse.ArgumentParser(
        description="Unified Lifelong MAPF experiment runner"
    )

    parser.add_argument(
        "--algorithms",
        type=parse_algorithms,
        default=list(DEFAULT_ALGORITHMS),
        help=(
            "Comma-separated algorithms, or 'all'. "
            f"Choices: {', '.join(SUPPORTED_ALGORITHMS)}"
        ),
    )

    parser.add_argument(
        "--agents",
        type=str,
        default=None,
        help="Comma-separated agent counts, e.g. 50,100,200",
    )

    parser.add_argument(
        "--agent-start",
        type=int,
        default=50,
        help="First agent count when --agents is not set",
    )

    parser.add_argument(
        "--agent-stop",
        type=int,
        default=500,
        help="Last inclusive agent count when --agents is not set",
    )

    parser.add_argument(
        "--agent-step",
        type=int,
        default=50,
        help="Agent count step when --agents is not set",
    )

    parser.add_argument(
        "--workers",
        "--works",
        dest="workers",
        type=int,
        default=8,
        help="Parallel workers",
    )

    parser.add_argument(
        "--obs-radius",
        type=int,
        default=None,
        help="Override the local observation radius (default: environment configuration)",
    )

    parser.add_argument(
        "--cache-algorithms",
        action="store_true",
        help="Reuse algorithm objects inside each worker. Faster, but less isolated between experiment tasks.",
    )

    parser.add_argument(
        "--animate", action="store_true", help="Generate SVG animations"
    )

    parser.add_argument("--max-steps", type=int, default=512, help="Episode length")

    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    parser.add_argument(
        "--seeds", type=str, default=None, help="Comma-separated seeds, e.g. 0,1,2"
    )

    parser.add_argument(
        "--main-dir", type=str, default="./", help="Project root directory"
    )

    parser.add_argument(
        "--map-types",
        type=str,
        default="all",
        help="Comma-separated map types, or 'all'",
    )

    parser.add_argument(
        "--map",
        action="append",
        default=[],
        help="Override one representative map with map_type=map_name. Can be repeated.",
    )

    parser.add_argument(
        "--map-file", type=str, default=None, help="Custom local map file path."
    )

    parser.add_argument(
        "--map-list",
        type=str,
        default=None,
        help="YAML file whose top-level keys are map names (e.g. maps/test.yaml)",
    )

    parser.add_argument(
        "--trim-border",
        dest="trim_border",
        action="store_true",
        help="Trim one-cell border from custom map",
    )

    parser.add_argument(
        "--no-trim-border",
        dest="trim_border",
        action="store_false",
        help="Do not trim border from custom map",
    )

    parser.set_defaults(trim_border=None)

    parser.add_argument(
        "--on-target",
        choices=("restart", "finish", "nothing"),
        default=None,
        help="Override Pogema on_target mode",
    )

    parser.add_argument(
        "--collision-system",
        choices=("soft", "block_both", "priority"),
        default="block_both",
        help="Override collision system",
    )

    parser.add_argument(
        "--arpe-candidate-manifest",
        type=str,
        default=None,
        help=(
            "Optional ARPE path declaration. Defaults to "
            "configs/arpe_final_candidate.json"
        ),
    )

    parser.add_argument(
        "--switcher-weights-path",
        type=str,
        default=None,
        help="Override the Switcher weights directory",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="exp_result",
        help="Directory for JSON results",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (default: experiments_TIMESTAMP.json)",
    )

    parser.add_argument(
        "--save",
        dest="save",
        action="store_true",
        default=True,
        help="Save JSON results",
    )

    parser.add_argument(
        "--no-save", dest="save", action="store_false", help="Do not save JSON results"
    )

    return parser.parse_args()


def main():

    multiprocessing.set_start_method("spawn", force=True)

    quiet_model_logs()

    args = parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    if args.max_steps < 1:
        raise ValueError("--max-steps must be at least 1")

    if args.obs_radius is not None and args.obs_radius < 1:
        raise ValueError("--obs-radius must be at least 1")

    map_sources = sum(bool(value) for value in (args.map_file, args.map_list))

    if map_sources > 1:
        raise ValueError("--map-file and --map-list are mutually exclusive")

    if args.on_target is None:
        args.on_target = "restart"

    if args.collision_system is None:
        args.collision_system = "block_both"

    if args.trim_border is None:
        args.trim_border = False

    algorithms = args.algorithms

    agent_counts = parse_agent_counts(args)

    seeds = parse_seeds(args)

    custom_map = None

    map_texts = None

    if args.map_file:
        custom_map = load_map_text(args.map_file, trim_border=args.trim_border)

        maps = {"custom": custom_map["map_name"]}

    elif args.map_list:
        maps, map_texts = load_map_list_snapshot(
            _project_path(args.main_dir, args.map_list),
            registry_path=_project_path(
                args.main_dir,
                "maps/test.yaml",
            ),
        )

    else:
        maps = parse_maps(args.map_types, args.map)

    tasks = build_tasks(
        algorithms,
        maps,
        agent_counts,
        seeds,
        args,
        custom_map=custom_map,
        map_texts=map_texts,
    )

    metadata = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "algorithms": algorithms,
        "agent_counts": agent_counts,
        "seeds": seeds,
        "maps": maps,
        "workers": args.workers,
        "obs_radius": args.obs_radius,
        "max_steps": args.max_steps,
        "on_target": args.on_target,
        "collision_system": args.collision_system,
    }

    print("Configuration")

    print(f"  algorithms: {', '.join(algorithms)}")

    print(
        f"  agent_counts: {agent_counts[0]}..{agent_counts[-1]} ({len(agent_counts)} values)"
    )

    print(f"  maps: {', '.join(maps.values())}")

    print(
        f"  obs_radius: {args.obs_radius if args.obs_radius is not None else 'default'}"
    )

    print(
        f"  max_steps: {args.max_steps} | seeds: {', '.join(str(seed) for seed in seeds)} | animate: {args.animate}"
    )

    print(f"  on_target: {args.on_target} | collision: {args.collision_system}")

    results, elapsed = run_experiments(
        tasks,
        args.workers,
    )

    metadata["finished_at"] = datetime.now().isoformat(timespec="seconds")

    metadata["total_elapsed_seconds"] = elapsed

    print(f"\nTotal elapsed: {format_duration(elapsed)}")

    if args.save:
        save_results(results, metadata, args.output_dir, args.output)

    failed = [result for result in results if result.get("error")]

    if failed:
        raise RuntimeError(
            f"{len(failed)} of {len(results)} experiments failed; "
            f"first error: {failed[0]['error']}"
        )


if __name__ == "__main__":
    main()
