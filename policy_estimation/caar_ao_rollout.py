"""Fixed-policy rollouts for absolute CAAR and AO-safe return estimation.

Both lanes advance the frozen CAAR policy and the raw dynamic planner on every
environment step.  The CAAR lane always executes CAAR.  The AO-safe lane
executes a valid, non-reverse raw Plan proposal and otherwise executes CAAR.
The raw planner is used directly; this module never imports or constructs any
probe policy.
"""

from __future__ import annotations

import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from policy_estimation.dataset import (
    DATASET_SCHEMA_VERSION,
    EpisodeSamples,
    deterministic_subsample_indices,
    discounted_returns,
    sha256_file,
)


CAAR_LANE = "caar"
AO_SAFE_LANE = "ao_safe"
LANES = (CAAR_LANE, AO_SAFE_LANE)
COLLECTION_BEHAVIOR_SOURCE_FILES = (
    "agents/caar.py",
    "agents/utils_agents.py",
    "learning/caar_actor_critic.py",
    "learning/caar_encoder.py",
    "learning/config.py",
    "learning/encoder.py",
    "learning/grid_memory.py",
    "pomapf_env/custom_maps.py",
    "pomapf_env/env.py",
    "pomapf_env/pomapf_config.py",
    "pomapf_env/stigmergic.py",
    "pomapf_env/wrappers.py",
    "planning/ao_replan_algo.py",
    "planning/raw_aoreplan_candidates.py",
    "planning/planner.cpp",
    "policy_estimation/caar_ao_rollout.py",
    "policy_estimation/dataset.py",
    "scripts/collect_caar_ao_returns.py",
    "train.py",
)
COLLECTION_BEHAVIOR_BINARY_GLOBS = (
    "planning/planner*.so",
    "planning/planner*.pyd",
)
PAIRED_IDENTITY_FIELDS = (
    "scenario_id",
    "initial_instance_sha256",
    "static_map_sha256",
    "actual_map_name",
    "map_family",
    "num_agents",
    "grid_config",
    "horizon",
    "gamma",
    "sample_fraction",
    "sample_seed",
    "sampling",
    "sampling_seed_strategy",
    "plan_use_best_move",
    "plan_max_steps",
    "obs_shape",
    "caar_checkpoint_sha256",
    "caar_config_sha256",
    "collection_implementation_files_sha256",
    "collection_implementation_sha256",
    "behavior_contract",
    "behavior_contract_sha256",
)


@dataclass(frozen=True)
class EpisodeSpec:
    """One reproducible environment instance."""

    scenario_id: str
    grid_config: Mapping[str, Any]

    def __post_init__(self) -> None:
        scenario_id = str(self.scenario_id).strip()
        if not scenario_id:
            raise ValueError("scenario_id must be non-empty.")
        config = json.loads(
            json.dumps(dict(self.grid_config), sort_keys=True)
        )
        if config.get("seed") is None:
            raise ValueError("Every episode must declare a fixed grid seed.")
        if int(config.get("num_agents", 0)) < 1:
            raise ValueError("grid_config.num_agents must be positive.")
        if int(config.get("max_episode_steps", 0)) < 1:
            raise ValueError("grid_config.max_episode_steps must be positive.")
        object.__setattr__(self, "scenario_id", scenario_id)
        object.__setattr__(self, "grid_config", config)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EpisodeSpec":
        scenario_id = value.get("scenario_id", value.get("id"))
        grid_config = value.get("grid_config")
        if scenario_id is None or not isinstance(grid_config, Mapping):
            raise ValueError(
                "Each scenario needs scenario_id (or id) and grid_config."
            )
        return cls(str(scenario_id), grid_config)


@dataclass(frozen=True)
class RolloutJob:
    """Pickle-safe settings for one production rollout worker."""

    episode: EpisodeSpec
    lane: str
    gamma: float = 0.99
    sample_fraction: float = 0.2
    sample_seed: int = 0
    caar_path_to_weights: str = "weights/CAAR/radius_ablation/R5"
    caar_checkpoint_kind: str = "auto"
    caar_device: str = "cpu"
    plan_use_best_move: bool = True
    plan_max_steps: int = 10_000
    torch_num_threads: int = 1

    def __post_init__(self) -> None:
        if self.lane not in LANES:
            raise ValueError(f"lane must be one of {LANES}, got {self.lane!r}.")
        if not math.isfinite(float(self.gamma)) or not 0.0 <= float(self.gamma) <= 1.0:
            raise ValueError("gamma must be finite and in [0, 1].")
        if not math.isfinite(float(self.sample_fraction)) or not (
            0.0 < float(self.sample_fraction) <= 1.0
        ):
            raise ValueError("sample_fraction must be finite and in (0, 1].")
        if int(self.plan_max_steps) < 1:
            raise ValueError("plan_max_steps must be positive.")
        if int(self.torch_num_threads) < 1:
            raise ValueError("torch_num_threads must be positive.")


@dataclass(frozen=True)
class LaneDecision:
    """Candidate and final joint actions for one environment step."""

    actions: tuple[int, ...]
    caar_actions: tuple[int, ...]
    plan_batch: Any
    plan_valid_mask: tuple[bool, ...]
    plan_selected_mask: tuple[bool, ...]
    planner_commit_mask: tuple[bool, ...]


def derive_episode_sample_seed(
    base_seed: int,
    scenario_id: str,
    lane: str | None = None,
) -> int:
    """Derive one shared paired seed without Python's randomized ``hash()``.

    ``lane`` is retained for call-site compatibility but deliberately does not
    enter the digest: CAAR and AO-safe use the same sampling seed for one
    scenario.
    """

    if lane is not None and lane not in LANES:
        raise ValueError(f"lane must be one of {LANES}, got {lane!r}.")
    payload = f"{int(base_seed)}\0{scenario_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


class FixedBehaviorLane:
    """Execute exactly one fixed behavior while updating both candidates."""

    def __init__(
        self,
        lane: str,
        caar_policy,
        plan_candidates,
        *,
        action_count: int | None = None,
    ):
        if lane not in LANES:
            raise ValueError(f"lane must be one of {LANES}, got {lane!r}.")
        self.lane = lane
        self.caar_policy = caar_policy
        self.plan_candidates = plan_candidates
        if action_count is None:
            actor = getattr(caar_policy, "ppo", None)
            action_space = getattr(actor, "action_space", None)
            action_count = int(getattr(action_space, "n", 5))
        self.action_count = int(action_count)
        if self.action_count < 1:
            raise ValueError("action_count must be positive.")
        self.last_decision: LaneDecision | None = None

    def after_reset(self, *, grid_config=None, env=None) -> None:
        self.plan_candidates.reset()
        self.caar_policy.after_reset()
        set_grid_config = getattr(self.caar_policy, "set_grid_config", None)
        if grid_config is not None and callable(set_grid_config):
            set_grid_config(grid_config)
        set_env = getattr(self.caar_policy, "set_env", None)
        if env is not None and callable(set_env):
            set_env(env)
        self.last_decision = None

    @staticmethod
    def _cancel_pending(plan_candidates, count: int) -> None:
        if getattr(plan_candidates, "pending", None) is not None:
            plan_candidates.commit([False] * count)

    def decide(
        self,
        observations: Sequence[Mapping[str, Any]],
        rewards=None,
        dones=None,
        infos=None,
    ) -> LaneDecision:
        count = len(observations)
        raw_caar = self.caar_policy.act(
            observations,
            rewards,
            dones,
            infos,
        )
        caar_actions = tuple(
            int(value)
            for value in np.asarray(raw_caar, dtype=np.int64).reshape(-1)
        )
        if len(caar_actions) != count:
            raise RuntimeError("CAAR returned the wrong action count.")
        if any(not 0 <= action < self.action_count for action in caar_actions):
            raise RuntimeError("CAAR returned an invalid environment action.")

        plan_batch = self.plan_candidates.propose(observations)
        try:
            if not (
                len(plan_batch.actions)
                == len(plan_batch.planned_mask)
                == len(plan_batch.reverse_mask)
                == count
            ):
                raise RuntimeError("Raw Plan returned the wrong action count.")
            valid = tuple(
                bool(
                    planned
                    and action is not None
                    and 0 <= int(action) < self.action_count
                )
                for action, planned in zip(
                    plan_batch.actions,
                    plan_batch.planned_mask,
                )
            )
            if self.lane == AO_SAFE_LANE:
                selected = tuple(
                    bool(is_valid and not reverse)
                    for is_valid, reverse in zip(valid, plan_batch.reverse_mask)
                )
            else:
                selected = (False,) * count
            actions = tuple(
                int(plan_batch.actions[index])
                if selected[index]
                else caar_actions[index]
                for index in range(count)
            )
            # Advance raw Plan only when its non-reverse proposal is exactly
            # the physical action. This also keeps it synchronized in the
            # CAAR lane when both candidates happen to agree.
            commit = tuple(
                bool(
                    planned
                    and not reverse
                    and plan_action is not None
                    and int(plan_action) == int(final_action)
                )
                for plan_action, planned, reverse, final_action in zip(
                    plan_batch.actions,
                    plan_batch.planned_mask,
                    plan_batch.reverse_mask,
                    actions,
                )
            )
            self.plan_candidates.commit(commit)
        except Exception:
            self._cancel_pending(self.plan_candidates, count)
            raise

        decision = LaneDecision(
            actions=actions,
            caar_actions=caar_actions,
            plan_batch=plan_batch,
            plan_valid_mask=valid,
            plan_selected_mask=selected,
            planner_commit_mask=commit,
        )
        self.last_decision = decision
        return decision

    def act(self, observations, rewards=None, dones=None, infos=None):
        return list(
            self.decide(observations, rewards, dones, infos).actions
        )

    def after_step(self, dones: Sequence[bool]) -> None:
        self.caar_policy.after_step(dones)


def _normalize_reset(result):
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, {}


def _normalize_step(result, count: int):
    if not isinstance(result, tuple):
        raise TypeError("env.step() must return a tuple.")
    if len(result) == 5:
        observations, rewards, terminated, truncated, infos = result
    elif len(result) == 4:
        observations, rewards, done, infos = result
        terminated = done
        truncated = [False] * count
    else:
        raise ValueError("env.step() must return four or five values.")
    rewards = np.asarray(rewards, dtype=np.float64).reshape(-1)
    terminated = np.asarray(terminated, dtype=np.bool_).reshape(-1)
    truncated = np.asarray(truncated, dtype=np.bool_).reshape(-1)
    if not (
        len(observations)
        == len(rewards)
        == len(terminated)
        == len(truncated)
        == count
    ):
        raise RuntimeError("Environment returned the wrong agent count.")
    if not bool(np.all(np.isfinite(rewards))):
        raise RuntimeError("Environment returned a non-finite reward.")
    return observations, rewards, terminated, truncated, infos


def _active_mask(
    dones: Sequence[bool],
    infos,
    count: int,
) -> tuple[bool, ...]:
    result = []
    per_agent_infos = infos if isinstance(infos, (list, tuple)) else None
    for index in range(count):
        active = not bool(dones[index])
        if per_agent_infos is not None and index < len(per_agent_infos):
            info = per_agent_infos[index]
            if isinstance(info, Mapping):
                active = active and bool(info.get("is_active", True))
        result.append(active)
    return tuple(result)


def _default_matrix_converter(observations):
    from pomapf_env.wrappers import MatrixObservationWrapper

    return MatrixObservationWrapper.to_matrix(observations)


def _canonical_array_digest(digest, label: str, value, dtype) -> None:
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    digest.update(label.encode("utf-8") + b"\0")
    digest.update(json.dumps(list(array.shape)).encode("ascii") + b"\0")
    digest.update(array.tobytes(order="C"))


def _static_obstacle_matrix(env):
    grid = getattr(env, "grid", None)
    obstacles = getattr(grid, "obstacles", None) if grid is not None else None
    if obstacles is None:
        getter = getattr(env, "get_obstacles", None)
        if callable(getter):
            try:
                obstacles = getter(ignore_borders=False)
            except TypeError:
                obstacles = getter()
    if obstacles is None:
        raise RuntimeError(
            "Return collection requires the global static obstacle matrix "
            "after reset()."
        )
    array = np.asarray(obstacles, dtype=np.uint8)
    if array.ndim != 2:
        raise RuntimeError(
            f"Static obstacle matrix must be two-dimensional, got {array.shape}."
        )
    return array


def static_map_sha256(env) -> str:
    """Hash only the global static obstacle matrix and its shape."""

    digest = hashlib.sha256()
    _canonical_array_digest(
        digest,
        "obstacles",
        _static_obstacle_matrix(env),
        np.uint8,
    )
    return digest.hexdigest()


def initial_instance_sha256(env, observations) -> str:
    """Hash the post-reset static map, starts, and initial targets."""

    grid = getattr(env, "grid", None)
    positions = getattr(grid, "positions_xy", None) if grid is not None else None
    if positions is None and grid is not None and hasattr(grid, "get_agents_xy"):
        positions = grid.get_agents_xy()
    targets = getattr(grid, "finishes_xy", None) if grid is not None else None
    if targets is None and grid is not None and hasattr(grid, "get_targets_xy"):
        targets = grid.get_targets_xy()
    if positions is None:
        positions = [observation["xy"] for observation in observations]
    if targets is None:
        targets = [observation["target_xy"] for observation in observations]

    digest = hashlib.sha256()
    _canonical_array_digest(
        digest,
        "obstacles",
        _static_obstacle_matrix(env),
        np.uint8,
    )
    _canonical_array_digest(digest, "positions", positions, "<i8")
    _canonical_array_digest(digest, "targets", targets, "<i8")
    return digest.hexdigest()


def _actual_map_metadata(env) -> tuple[str | None, str | None]:
    grid_config = getattr(env, "grid_config", None)
    map_name = getattr(grid_config, "map_name", None)
    if map_name is None:
        return None, None
    map_name = str(map_name)
    family = map_name.split("-", 1)[0].lower()
    return map_name, family


def _is_sha256(value) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _canonical_sha256_mapping(values: Mapping[str, str]) -> str:
    payload = json.dumps(
        dict(values),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def collection_implementation_identity(
    project_root: str | Path | None = None,
    *,
    required: bool = True,
) -> dict[str, Any]:
    """Hash every repository file that can change collection behavior.

    Production workers call this inside their own process.  The aggregate is
    the SHA-256 of a canonical ``relative path -> file SHA-256`` mapping, so a
    change to either source or the actually loaded native planner changes the
    behavior contract.
    """

    root = (
        Path(__file__).resolve().parents[1]
        if project_root is None
        else Path(project_root).resolve()
    )
    files: dict[str, str] = {}
    missing = []
    for relative in COLLECTION_BEHAVIOR_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        files[relative] = sha256_file(path)

    native_paths = {
        path.resolve()
        for pattern in COLLECTION_BEHAVIOR_BINARY_GLOBS
        for path in root.glob(pattern)
        if path.is_file()
    }
    if not native_paths:
        missing.append("planning/planner*.{so,pyd}")
    for path in sorted(native_paths, key=lambda value: value.as_posix()):
        relative = path.relative_to(root).as_posix()
        files[relative] = sha256_file(path)

    if required and missing:
        raise RuntimeError(
            "Production collection implementation identity is incomplete; "
            f"missing={missing}."
        )
    normalized = dict(sorted(files.items()))
    return {
        "collection_implementation_files_sha256": normalized,
        "collection_implementation_sha256": _canonical_sha256_mapping(
            normalized
        ),
    }


def _normalize_collection_identity(
    identity: Mapping[str, Any] | None,
    *,
    required: bool,
) -> dict[str, Any]:
    if identity is None:
        if required:
            raise RuntimeError(
                "Production collection implementation identity is required."
            )
        return {
            "collection_implementation_files_sha256": {},
            "collection_implementation_sha256": None,
        }
    if not isinstance(identity, Mapping):
        raise TypeError("collection_identity must be a mapping.")
    raw_files = identity.get("collection_implementation_files_sha256")
    if not isinstance(raw_files, Mapping):
        raise RuntimeError(
            "collection_identity lacks its per-file SHA-256 mapping."
        )
    files = {
        str(path): str(digest).lower()
        for path, digest in raw_files.items()
    }
    invalid = [
        path
        for path, digest in files.items()
        if not path or not _is_sha256(digest)
    ]
    if invalid or (required and not files):
        raise RuntimeError(
            "Collection implementation file identity is invalid; "
            f"invalid={invalid}, empty={not files}."
        )
    files = dict(sorted(files.items()))
    expected = _canonical_sha256_mapping(files)
    declared = identity.get("collection_implementation_sha256")
    if not _is_sha256(declared) or str(declared).lower() != expected:
        raise RuntimeError(
            "collection_implementation_sha256 does not match its per-file "
            "mapping."
        )
    return {
        "collection_implementation_files_sha256": files,
        "collection_implementation_sha256": expected,
    }


def caar_artifact_identity(caar_policy, *, required: bool) -> dict[str, str | None]:
    """Read the exact frozen CAAR config/checkpoint identity from the policy."""

    values = {
        "caar_checkpoint_sha256": getattr(
            caar_policy, "checkpoint_sha256", None
        ),
        "caar_config_sha256": getattr(caar_policy, "config_sha256", None),
        "caar_checkpoint_path": getattr(caar_policy, "checkpoint_path", None),
        "caar_config_path": getattr(caar_policy, "config_path", None),
    }
    normalized = {
        name: None if value is None else str(value)
        for name, value in values.items()
    }
    if required:
        missing = [name for name, value in normalized.items() if not value]
        invalid_hashes = [
            name
            for name in ("caar_checkpoint_sha256", "caar_config_sha256")
            if not _is_sha256(normalized[name])
        ]
        if missing or invalid_hashes:
            raise RuntimeError(
                "Production CAAR artifact identity is incomplete; "
                f"missing={missing}, invalid_hashes={invalid_hashes}."
            )
    return normalized


def validate_paired_episode_samples(
    caar_samples: EpisodeSamples,
    ao_safe_samples: EpisodeSamples,
) -> None:
    """Reject branch results that do not describe the exact same instance."""

    if caar_samples.metadata.get("branch") != CAAR_LANE:
        raise ValueError("The first paired sample must be the CAAR branch.")
    if ao_safe_samples.metadata.get("branch") != AO_SAFE_LANE:
        raise ValueError("The second paired sample must be the AO-safe branch.")
    mismatches = []
    for field in PAIRED_IDENTITY_FIELDS:
        left = caar_samples.metadata.get(field)
        right = ao_safe_samples.metadata.get(field)
        if isinstance(left, Mapping) or isinstance(right, Mapping):
            left = json.dumps(left, ensure_ascii=False, sort_keys=True)
            right = json.dumps(right, ensure_ascii=False, sort_keys=True)
        if left != right:
            mismatches.append(field)
    if mismatches:
        scenario = caar_samples.metadata.get("scenario_id")
        raise ValueError(
            f"Paired branch identity mismatch for {scenario!r}: {mismatches}."
        )


def _allocate_transition_buffers(
    capacity: int,
    obs_shape: Sequence[int],
) -> dict[str, np.ndarray]:
    """Allocate the compact upper bound for one fixed-horizon episode."""

    capacity = int(capacity)
    shape = tuple(int(value) for value in obs_shape)
    if capacity < 1:
        raise ValueError("transition buffer capacity must be positive.")
    if len(shape) != 3 or min(shape) < 1:
        raise RuntimeError(
            f"Matrix observations must have shape (C,H,W), got {shape}."
        )
    return {
        "obs": np.empty((capacity, *shape), dtype=np.uint8),
        "xy": np.empty((capacity, 2), dtype=np.int32),
        "target_xy": np.empty((capacity, 2), dtype=np.int32),
        "reward": np.empty(capacity, dtype=np.float32),
        "caar_action": np.empty(capacity, dtype=np.int16),
        "plan_action": np.empty(capacity, dtype=np.int16),
        "executed_action": np.empty(capacity, dtype=np.int16),
        "plan_valid": np.empty(capacity, dtype=np.bool_),
        "plan_reverse": np.empty(capacity, dtype=np.bool_),
        "plan_selected": np.empty(capacity, dtype=np.bool_),
        "planner_committed": np.empty(capacity, dtype=np.bool_),
        "agent_id": np.empty(capacity, dtype=np.int32),
        "timestep": np.empty(capacity, dtype=np.int32),
        "terminated": np.empty(capacity, dtype=np.bool_),
        "truncated": np.empty(capacity, dtype=np.bool_),
    }


def collect_episode(
    env,
    behavior: FixedBehaviorLane,
    *,
    episode: EpisodeSpec,
    gamma: float = 0.99,
    sample_fraction: float = 0.2,
    sample_seed: int,
    require_caar_artifact_identity: bool = False,
    collection_identity: Mapping[str, Any] | None = None,
    require_collection_identity: bool = False,
    matrix_converter: Callable[[Sequence[Mapping[str, Any]]], Sequence[Mapping[str, Any]]] = _default_matrix_converter,
) -> EpisodeSamples:
    """Collect one episode and align every target with its pre-action state."""

    implementation_identity = _normalize_collection_identity(
        collection_identity,
        required=bool(require_collection_identity),
    )
    observations, reset_info = _normalize_reset(env.reset())
    count = len(observations)
    expected_agents = int(episode.grid_config["num_agents"])
    if count != expected_agents:
        raise RuntimeError(
            f"Expected {expected_agents} agents, environment returned {count}."
        )
    static_digest = static_map_sha256(env)
    instance_digest = initial_instance_sha256(env, observations)
    actual_map_name, map_family = _actual_map_metadata(env)
    artifact_identity = caar_artifact_identity(
        behavior.caar_policy,
        required=bool(require_caar_artifact_identity),
    )
    behavior.after_reset(
        grid_config=getattr(env, "grid_config", None),
        env=env,
    )
    max_steps = int(episode.grid_config["max_episode_steps"])
    previous_rewards = np.zeros(count, dtype=np.float32)
    previous_dones = np.zeros(count, dtype=np.bool_)
    infos = (
        reset_info
        if isinstance(reset_info, (list, tuple))
        else [{"is_active": True} for _ in range(count)]
    )

    capacity = count * max_steps
    buffers: dict[str, np.ndarray] | None = None
    row_count = 0
    episode_steps = 0
    for timestep in range(max_steps):
        active = _active_mask(previous_dones, infos, count)
        # Snapshot o_t before either candidate or the environment can mutate
        # any observation-owned array.
        matrix_observations = matrix_converter(observations)
        if len(matrix_observations) != count:
            raise RuntimeError("Matrix converter returned the wrong agent count.")
        if buffers is None:
            first_obs_shape = np.asarray(matrix_observations[0]["obs"]).shape
            buffers = _allocate_transition_buffers(capacity, first_obs_shape)
        active_indices = np.flatnonzero(
            np.asarray(active, dtype=np.bool_)
        )
        if row_count + len(active_indices) > capacity:
            raise RuntimeError(
                "Active transition count exceeded the fixed episode capacity."
            )
        # Copy o_t into its final compact storage before either policy or the
        # environment can mutate arrays owned by the observation.
        for offset, index_value in enumerate(active_indices):
            index = int(index_value)
            row = row_count + offset
            obs = np.asarray(matrix_observations[index]["obs"])
            xy = np.asarray(matrix_observations[index]["xy"])
            target_xy = np.asarray(matrix_observations[index]["target_xy"])
            if obs.shape != buffers["obs"].shape[1:]:
                raise RuntimeError(
                    "Matrix observation shape changed during the episode: "
                    f"expected {buffers['obs'].shape[1:]}, got {obs.shape}."
                )
            if xy.shape != (2,) or target_xy.shape != (2,):
                raise RuntimeError("xy and target_xy must each have shape (2,).")
            buffers["obs"][row] = obs
            buffers["xy"][row] = xy
            buffers["target_xy"][row] = target_xy
            buffers["agent_id"][row] = index
            buffers["timestep"][row] = timestep
        decision = behavior.decide(
            observations,
            previous_rewards,
            previous_dones,
            infos,
        )
        step_result = _normalize_step(env.step(list(decision.actions)), count)
        (
            next_observations,
            rewards,
            terminated,
            truncated,
            next_infos,
        ) = step_result
        forced_horizon = timestep + 1 >= max_steps
        if forced_horizon:
            unfinished = np.logical_and(
                np.asarray(active, dtype=np.bool_),
                np.logical_not(np.logical_or(terminated, truncated)),
            )
            truncated = np.logical_or(
                truncated,
                unfinished,
            )
        dones = np.logical_or(terminated, truncated)

        for offset, index_value in enumerate(active_indices):
            index = int(index_value)
            row = row_count + offset
            plan_action = decision.plan_batch.actions[index]
            buffers["reward"][row] = rewards[index]
            buffers["caar_action"][row] = decision.caar_actions[index]
            buffers["plan_action"][row] = (
                -1 if plan_action is None else int(plan_action)
            )
            buffers["executed_action"][row] = decision.actions[index]
            buffers["plan_valid"][row] = decision.plan_valid_mask[index]
            buffers["plan_reverse"][row] = (
                decision.plan_batch.reverse_mask[index]
            )
            buffers["plan_selected"][row] = (
                decision.plan_selected_mask[index]
            )
            buffers["planner_committed"][row] = (
                decision.planner_commit_mask[index]
            )
            buffers["terminated"][row] = terminated[index]
            buffers["truncated"][row] = truncated[index]
        row_count += len(active_indices)

        behavior.after_step(dones.tolist())
        episode_steps = timestep + 1
        if bool(np.all(dones)):
            break
        observations = next_observations
        previous_rewards = rewards
        previous_dones = dones
        infos = next_infos

    full_row_count = row_count
    if full_row_count == 0:
        raise RuntimeError("Episode produced no active agent transitions.")
    if buffers is None:
        raise RuntimeError("Episode transition buffers were not initialized.")
    agent_ids = buffers["agent_id"][:full_row_count]
    rewards_array = buffers["reward"][:full_row_count]
    returns = np.empty(full_row_count, dtype=np.float32)
    for agent_index in range(count):
        agent_rows = np.flatnonzero(agent_ids == agent_index)
        if len(agent_rows):
            returns[agent_rows] = discounted_returns(
                rewards_array[agent_rows],
                gamma,
            )

    selected = deterministic_subsample_indices(
        full_row_count,
        fraction=sample_fraction,
        seed=sample_seed,
    )
    selected_digest = hashlib.sha256(
        np.asarray(selected, dtype="<i8").tobytes()
    ).hexdigest()
    arrays = {
        name: np.ascontiguousarray(values[:full_row_count][selected])
        for name, values in buffers.items()
    }
    metadata = {
        "scenario_id": episode.scenario_id,
        "grid_config": dict(episode.grid_config),
        "actual_map_name": actual_map_name,
        "map_family": map_family,
        "static_map_sha256": static_digest,
        "initial_instance_sha256": instance_digest,
        "branch": behavior.lane,
        "lane": behavior.lane,
        "gamma": float(gamma),
        "sample_fraction": float(sample_fraction),
        "sample_seed": int(sample_seed),
        "sampling": "fixed_size_without_replacement",
        "sampling_seed_strategy": "sha256(base_seed,scenario_id)",
        "sampled_indices_sha256": selected_digest,
        "full_row_count": full_row_count,
        "episode_steps": episode_steps,
        "num_agents": count,
        **artifact_identity,
        **implementation_identity,
    }
    obs_shape = list(buffers["obs"].shape[1:])
    plan_use_best_move = getattr(
        behavior.plan_candidates,
        "use_best_move",
        None,
    )
    plan_max_steps = getattr(behavior.plan_candidates, "max_steps", None)
    if require_caar_artifact_identity and (
        plan_use_best_move is None or plan_max_steps is None
    ):
        raise RuntimeError(
            "Production raw Plan configuration is missing use_best_move "
            "or max_steps."
        )
    metadata.update(
        {
            "horizon": max_steps,
            "plan_use_best_move": (
                None
                if plan_use_best_move is None
                else bool(plan_use_best_move)
            ),
            "plan_max_steps": (
                None if plan_max_steps is None else int(plan_max_steps)
            ),
            "obs_shape": obs_shape,
        }
    )
    behavior_contract = {
        "schema_version": "caar_ao_behavior_contract_v1",
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "gamma": float(gamma),
        "sample_fraction": float(sample_fraction),
        "sampling": "fixed_size_without_replacement",
        "sampling_seed_strategy": "sha256(base_seed,scenario_id)",
        "caar_checkpoint_sha256": artifact_identity[
            "caar_checkpoint_sha256"
        ],
        "caar_config_sha256": artifact_identity["caar_config_sha256"],
        "collection_implementation_sha256": implementation_identity[
            "collection_implementation_sha256"
        ],
        "plan_use_best_move": metadata["plan_use_best_move"],
        "plan_max_steps": metadata["plan_max_steps"],
        "obs_shape": obs_shape,
    }
    behavior_contract_json = json.dumps(
        behavior_contract,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    metadata["behavior_contract"] = behavior_contract
    metadata["behavior_contract_sha256"] = hashlib.sha256(
        behavior_contract_json.encode("utf-8")
    ).hexdigest()
    return EpisodeSamples(
        obs=arrays.pop("obs"),
        xy=arrays.pop("xy"),
        target_xy=arrays.pop("target_xy"),
        mc_return=returns[selected],
        metadata=metadata,
        **arrays,
    )


def create_production_components(job: RolloutJob):
    """Construct worker-local environment and policies lazily."""

    import torch

    # Each rollout process loads its own frozen CAAR. Limiting intra-op pools
    # prevents a multi-process collector from oversubscribing all host cores.
    torch.set_num_threads(int(job.torch_num_threads))
    set_interop = getattr(torch, "set_num_interop_threads", None)
    if callable(set_interop):
        try:
            set_interop(1)
        except RuntimeError:
            # PyTorch permits this setting only once per process and may have
            # initialized the pool during an earlier injected smoke call.
            pass
    from agents.caar import CAAR, CAARConfig
    from planning.raw_aoreplan_candidates import RawAORePlanCandidates
    from pomapf_env.env import make_pomapf
    from pomapf_env.pomapf_config import POMAPFConfig

    grid_config = POMAPFConfig(**dict(job.episode.grid_config))
    env = make_pomapf(grid_config=grid_config)
    caar = CAAR(
        CAARConfig(
            path_to_weights=job.caar_path_to_weights,
            checkpoint_kind=job.caar_checkpoint_kind,
            device=job.caar_device,
            seed=int(grid_config.seed),
        )
    )
    actor = getattr(caar, "ppo", None)
    if actor is not None:
        # Standalone CAAR intentionally remains in training mode during
        # inference so its observation normalizer keeps the same online
        # update behavior. Freeze only learnable parameters here.
        train = getattr(actor, "train", None)
        if callable(train):
            train(True)
        parameters = getattr(actor, "parameters", None)
        if callable(parameters):
            for parameter in parameters():
                parameter.requires_grad_(False)
    planner = RawAORePlanCandidates(
        use_best_move=job.plan_use_best_move,
        max_steps=job.plan_max_steps,
        seed=int(grid_config.seed),
    )
    return env, FixedBehaviorLane(job.lane, caar, planner)


def collect_rollout_job(
    job: RolloutJob,
    *,
    component_factory: Callable[[RolloutJob], tuple[Any, FixedBehaviorLane]] = create_production_components,
) -> EpisodeSamples:
    """Collect one job with an injectable component factory."""

    production = component_factory is create_production_components
    implementation_identity = (
        collection_implementation_identity(required=True)
        if production
        else None
    )
    env, behavior = component_factory(job)
    try:
        samples = collect_episode(
            env,
            behavior,
            episode=job.episode,
            gamma=job.gamma,
            sample_fraction=job.sample_fraction,
            sample_seed=job.sample_seed,
            require_caar_artifact_identity=production,
            collection_identity=implementation_identity,
            require_collection_identity=production,
        )
        samples.metadata["job"] = asdict(job)
        return samples
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def collect_production_job(job: RolloutJob) -> EpisodeSamples:
    """Top-level pickleable worker used by ``ProcessPoolExecutor``."""

    return collect_rollout_job(job)


def iter_collected_jobs(
    jobs: Iterable[RolloutJob],
    *,
    max_workers: int,
    worker: Callable[[RolloutJob], EpisodeSamples] = collect_production_job,
    executor_factory=ProcessPoolExecutor,
) -> Iterator[EpisodeSamples]:
    """Collect jobs in input order, with injectable worker/executor hooks."""

    max_workers = int(max_workers)
    if max_workers < 1:
        raise ValueError("max_workers must be positive.")
    jobs = tuple(jobs)
    if max_workers == 1:
        for job in jobs:
            yield worker(job)
        return
    with executor_factory(max_workers=max_workers) as executor:
        yield from executor.map(worker, jobs)


__all__ = [
    "AO_SAFE_LANE",
    "CAAR_LANE",
    "COLLECTION_BEHAVIOR_BINARY_GLOBS",
    "COLLECTION_BEHAVIOR_SOURCE_FILES",
    "LANES",
    "PAIRED_IDENTITY_FIELDS",
    "EpisodeSpec",
    "FixedBehaviorLane",
    "LaneDecision",
    "RolloutJob",
    "collect_episode",
    "collect_production_job",
    "collect_rollout_job",
    "collection_implementation_identity",
    "create_production_components",
    "derive_episode_sample_seed",
    "caar_artifact_identity",
    "initial_instance_sha256",
    "static_map_sha256",
    "validate_paired_episode_samples",
    "iter_collected_jobs",
]
