"""Original-style learnable switcher between CAAR and raw AO-RePlan.

The switcher follows the value-evaluation design from *When to Switch*: two
independent estimators predict absolute Monte-Carlo returns on every step.  The
larger prediction selects the nominal branch immediately.  A missing or invalid
raw Plan proposal is replaced for that step by CAAR.  Reverse proposals can
either use the default CAAR safety override or execute exactly as selected by
the predictor.  An optional per-agent reverse cooldown keeps CAAR active for a
fixed number of steps, including the triggering step.  It never constructs or
calls AO-RePlan's Probe.
"""

from __future__ import annotations

import hashlib
import json
from typing import Callable, Literal, Mapping, Sequence

import numpy as np
from pydantic import Extra, Field, validator

from agents.caar import CAAR, CAARConfig
from agents.utils_agents import AlgoBase
from planning.raw_aoreplan_candidates import RawAORePlanCandidates, RawPlanBatch
from pomapf_env.wrappers import MatrixObservationWrapper


CAAR_BRANCH = 0
AO_BRANCH = 1
HYBRID_MODE = "per_step_absolute_return_lswitcher_reverse_to_caar_v1"
REVERSE_COOLDOWN_HYBRID_MODE = (
    "per_step_absolute_return_lswitcher_reverse_to_caar_cooldown_v1"
)
PREDICTOR_ONLY_HYBRID_MODE = (
    "per_step_absolute_return_lswitcher_predictor_only_v1"
)
ROAD_TOPOLOGY_PROVENANCE_VERSION = "caar_ls_road_topology_v1"
DEFAULT_ACTION_COUNT = 5
IDENTITY_FIELDS = (
    "caar_checkpoint_sha256",
    "caar_config_sha256",
)


def _freeze_policy_parameters(policy) -> None:
    """Freeze weights without changing CAAR's deployed normalizer mode."""

    actor = getattr(policy, "ppo", None)
    parameters = getattr(actor, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            parameter.requires_grad_(False)


class CAARLSConfig(AlgoBase, extra=Extra.forbid):
    """Deployment configuration for the absolute-return learnable switcher."""

    name: Literal["CAAR-LS"] = "CAAR-LS"
    hybrid_mode: Literal[
        "per_step_absolute_return_lswitcher_reverse_to_caar_v1",
        "per_step_absolute_return_lswitcher_reverse_to_caar_cooldown_v1",
        "per_step_absolute_return_lswitcher_predictor_only_v1",
    ] = HYBRID_MODE
    caar: CAARConfig = CAARConfig(
        path_to_weights="weights/CAAR/CAAR",
        checkpoint_kind="latest",
    )
    caar_estimator_checkpoint_path: str = "weights/CAAR-LS/caar_estimator.pth"
    ao_estimator_checkpoint_path: str = "weights/CAAR-LS/ao_estimator.pth"
    estimator_device: str = "auto"
    value_margin: float = 0.0
    reverse_caar_override_enabled: bool = True
    reverse_caar_cooldown_steps: int = Field(4, ge=0)
    road_topology_adaptive_cooldown_enabled: bool = False
    road_open4_threshold: float = Field(0.68, ge=0.0, le=1.0)
    road_dense_obstacle_threshold: float = Field(0.70, ge=0.0, le=1.0)
    road_reverse_caar_cooldown_steps: int = Field(8, ge=0)
    road_caar_only_density_threshold: float | None = Field(
        None,
        ge=0.0,
    )
    plan_use_best_move: bool = True
    max_planning_steps: int = Field(10_000, gt=0)

    @validator("value_margin")
    def finite_margin(cls, value):
        if not np.isfinite(value):
            raise ValueError("value_margin must be finite.")
        return float(value)

    @validator("reverse_caar_cooldown_steps")
    def cooldown_requires_reverse_override(cls, value, values):
        if int(value) > 0 and not values.get(
            "reverse_caar_override_enabled", True
        ):
            raise ValueError(
                "reverse CAAR cooldown requires the reverse override"
            )
        return int(value)

    @validator("road_reverse_caar_cooldown_steps")
    def road_cooldown_requires_reverse_override(cls, value, values):
        if (
            values.get("road_topology_adaptive_cooldown_enabled", False)
            and int(value) > 0
            and not values.get("reverse_caar_override_enabled", True)
        ):
            raise ValueError(
                "road reverse CAAR cooldown requires the reverse override"
            )
        return int(value)

    @validator("road_caar_only_density_threshold")
    def finite_optional_density_threshold(cls, value):
        if value is not None and not np.isfinite(value):
            raise ValueError(
                "road CAAR-only density threshold must be finite"
            )
        return None if value is None else float(value)


def _default_estimator_factory(**kwargs):
    """Lazily import the estimator so injected unit tests need no checkpoint."""

    from policy_estimation.model import PolicyReturnEstimator

    return PolicyReturnEstimator(**kwargs)


def _default_collection_identity_factory(*, required: bool):
    """Resolve the collector identity helper lazily for production deploys."""

    from policy_estimation.caar_ao_rollout import (
        collection_implementation_identity,
    )

    return collection_implementation_identity(required=required)


def select_ao_by_absolute_return(
    caar_values,
    ao_values,
    *,
    margin: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return AO choices and a mask of non-finite comparisons.

    AO is selected only for the strict comparison
    ``V_AO > V_CAAR + margin``.  Ties and every comparison containing a NaN or
    infinity stay with CAAR.
    """

    caar = np.asarray(caar_values, dtype=np.float64).reshape(-1)
    ao = np.asarray(ao_values, dtype=np.float64).reshape(-1)
    if caar.shape != ao.shape:
        raise ValueError("CAAR and AO value batches must have equal shapes.")
    if not np.isfinite(margin):
        raise ValueError("value margin must be finite.")
    nonfinite = ~(np.isfinite(caar) & np.isfinite(ao))
    choices = (~nonfinite) & (ao > caar + float(margin))
    return choices.astype(np.int64), nonfinite


class _DeploymentRawAORePlanCandidates(RawAORePlanCandidates):
    """Allow reverse commits only for predictor-only CAAR-LS deployment.

    The shared candidate generator remains identical to the implementation
    used for estimator-data collection.  Keeping this adapter local prevents
    the experimental deployment rule from changing other policies or their
    recorded collection contract.
    """

    def commit(self, executed_mask: Sequence[bool], *, allow_reverse=False):
        if not allow_reverse:
            return super().commit(executed_mask)
        pending = self.pending
        if pending is None:
            raise RuntimeError("propose() must be called before commit().")
        if len(executed_mask) != len(pending.actions):
            raise ValueError(
                "executed_mask and the pending Plan batch must have equal sizes."
            )
        mask = [bool(value) for value in executed_mask]
        if any(
            executed and not planned
            for executed, planned in zip(mask, pending.planned_mask)
        ):
            raise ValueError("A missing raw Plan proposal cannot be committed.")
        self._base.commit_proposals(mask)
        self._pending = None


class CAARLS:
    """Per-step absolute-return switcher over CAAR and raw AO-RePlan."""

    def __init__(
        self,
        cfg: CAARLSConfig,
        *,
        caar_factory: Callable[[CAARConfig], object] = CAAR,
        planner_factory: Callable[..., object] = _DeploymentRawAORePlanCandidates,
        estimator_factory: Callable[..., object] | None = None,
        collection_identity_factory: Callable[..., object] | None = None,
    ):
        self.cfg = cfg
        caar_cfg = cfg.caar.copy(deep=True, update={"seed": cfg.seed})
        self.caar = caar_factory(caar_cfg)
        # Standalone CAAR retains online observation-normalizer updates.  Keep
        # that mode so an all-CAAR switcher is action-for-action equivalent.
        _freeze_policy_parameters(self.caar)
        self.action_count = self._infer_action_count(self.caar)
        self.plan_candidates = planner_factory(
            use_best_move=cfg.plan_use_best_move,
            max_steps=cfg.max_planning_steps,
            seed=cfg.seed,
        )

        factory = estimator_factory or _default_estimator_factory
        common = {"device": cfg.estimator_device}
        self.caar_estimator = factory(
            checkpoint_path=cfg.caar_estimator_checkpoint_path,
            expected_branch="caar",
            **common,
        )
        self.ao_estimator = factory(
            checkpoint_path=cfg.ao_estimator_checkpoint_path,
            expected_branch="ao_safe",
            **common,
        )
        if self.caar_estimator is self.ao_estimator:
            raise ValueError("CAAR-LS requires two independent value estimators.")
        self._collection_identity_factory = (
            collection_identity_factory
            or _default_collection_identity_factory
        )
        self._freeze_estimator(self.caar_estimator)
        self._freeze_estimator(self.ao_estimator)
        self._validate_estimator_policy_identity()

        self.device = getattr(self.caar, "device", cfg.device)
        self.env = None
        self._road_open4_ratio: float | None = None
        self._road_dense_obstacle_ratio: float | None = None
        self._road_open4_count: int | None = None
        self._road_dense_obstacle_count: int | None = None
        self._road_free_cell_count: int | None = None
        self._road_obstacle_cell_count: int | None = None
        self._road_map_shape: tuple[int, int] | None = None
        self._road_agent_density: float | None = None
        self._road_topology_detected = False
        self._road_density_gate_active = False
        self._effective_reverse_caar_cooldown_steps = int(
            cfg.reverse_caar_cooldown_steps
        )
        self.after_reset()

    @staticmethod
    def _infer_action_count(caar) -> int:
        actor = getattr(caar, "ppo", None)
        action_space = getattr(actor, "action_space", None)
        count = getattr(action_space, "n", DEFAULT_ACTION_COUNT)
        return int(count)

    @staticmethod
    def _freeze_estimator(estimator) -> None:
        eval_method = getattr(estimator, "eval", None)
        if callable(eval_method):
            eval_method()
        parameters = getattr(estimator, "parameters", None)
        if callable(parameters):
            for parameter in parameters():
                parameter.requires_grad_(False)

    @staticmethod
    def _identity_metadata(estimator, label: str) -> dict:
        metadata = getattr(estimator, "training_metadata", None)
        if not isinstance(metadata, dict):
            raise ValueError(
                f"The {label} estimator is missing training identity metadata."
            )
        if metadata.get("deployable") is not True:
            raise ValueError(
                f"The {label} estimator is explicitly non-deployable."
            )
        identity = {}
        for field in IDENTITY_FIELDS:
            value = str(metadata.get(field, "")).lower()
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(
                    f"The {label} estimator has an invalid {field}."
                )
            identity[field] = value
        contract = metadata.get("behavior_contract")
        if not isinstance(contract, dict):
            raise ValueError(
                f"The {label} estimator is missing behavior_contract."
            )
        canonical = json.dumps(
            contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        actual_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        declared_digest = str(
            metadata.get("behavior_contract_sha256", "")
        ).lower()
        if declared_digest != actual_digest:
            raise ValueError(
                f"The {label} estimator behavior contract hash is invalid."
            )
        if contract.get("schema_version") != "caar_ao_behavior_contract_v1":
            raise ValueError(
                f"The {label} estimator has an unsupported behavior contract."
            )
        raw_collection_digest = contract.get(
            "collection_implementation_sha256"
        )
        collection_digest = (
            raw_collection_digest.lower()
            if isinstance(raw_collection_digest, str)
            else ""
        )
        if len(collection_digest) != 64 or any(
            character not in "0123456789abcdef"
            for character in collection_digest
        ):
            raise ValueError(
                f"The {label} estimator behavior contract has an invalid "
                "collection_implementation_sha256."
            )
        for field in IDENTITY_FIELDS:
            if str(contract.get(field, "")).lower() != identity[field]:
                raise ValueError(
                    f"The {label} estimator contract disagrees on {field}."
                )
        horizons = metadata.get("training_horizons")
        if not isinstance(horizons, (list, tuple)) or not horizons:
            raise ValueError(
                f"The {label} estimator is missing training_horizons."
            )
        try:
            horizons = tuple(sorted({int(value) for value in horizons}))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"The {label} estimator has invalid training_horizons."
            ) from exc
        if any(value < 1 for value in horizons):
            raise ValueError(
                f"The {label} estimator has invalid training_horizons."
            )
        identity["behavior_contract"] = contract
        identity["behavior_contract_sha256"] = declared_digest
        identity["training_horizons"] = horizons
        coordinate_encoding = str(
            metadata.get("coordinate_encoding", "absolute_v1")
        )
        if coordinate_encoding != "absolute_v1":
            raise ValueError(
                f"The {label} estimator has an invalid coordinate_encoding."
            )
        identity["coordinate_encoding"] = coordinate_encoding
        return identity

    def _validate_estimator_policy_identity(self) -> None:
        """Refuse value comparisons across different frozen CAAR policies."""

        caar_identity = self._identity_metadata(
            self.caar_estimator,
            "CAAR",
        )
        ao_identity = self._identity_metadata(
            self.ao_estimator,
            "AO-safe",
        )
        if caar_identity != ao_identity:
            raise ValueError(
                "The CAAR and AO-safe estimators were trained against "
                "different CAAR policy identities or behavior contracts."
            )

        deployed = {
            "caar_checkpoint_sha256": str(
                getattr(self.caar, "checkpoint_sha256", "")
            ).lower(),
            "caar_config_sha256": str(
                getattr(self.caar, "config_sha256", "")
            ).lower(),
        }
        for field in IDENTITY_FIELDS:
            expected = caar_identity[field]
            if deployed[field] != expected:
                raise ValueError(
                    f"Estimator/deployment CAAR identity mismatch for {field}: "
                    f"trained={expected}, deployed={deployed[field] or 'missing'}."
                )
        contract = caar_identity["behavior_contract"]
        if contract.get("plan_use_best_move") != self.cfg.plan_use_best_move:
            raise ValueError(
                "Estimator/deployment plan_use_best_move mismatch."
            )
        try:
            trained_plan_steps = int(contract.get("plan_max_steps"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "Estimator behavior contract has invalid plan_max_steps."
            ) from exc
        if trained_plan_steps != self.cfg.max_planning_steps:
            raise ValueError(
                "Estimator/deployment max_planning_steps mismatch: "
                f"trained={trained_plan_steps}, "
                f"deployed={self.cfg.max_planning_steps}."
            )

        # Bind deployment to the exact fixed-behavior implementation used to
        # collect the return targets.  The rollout module remains the single
        # owner of file selection and aggregate hashing; this class only
        # validates and compares the aggregate recorded in the contract.
        current_identity = self._collection_identity_factory(required=True)
        if not isinstance(current_identity, Mapping):
            raise ValueError(
                "Current collection implementation identity is invalid."
            )
        raw_current_digest = current_identity.get(
            "collection_implementation_sha256"
        )
        current_digest = (
            raw_current_digest.lower()
            if isinstance(raw_current_digest, str)
            else ""
        )
        if len(current_digest) != 64 or any(
            character not in "0123456789abcdef"
            for character in current_digest
        ):
            raise ValueError(
                "Current collection implementation identity has an invalid "
                "collection_implementation_sha256."
            )
        trained_digest = str(
            contract["collection_implementation_sha256"]
        ).lower()
        if current_digest != trained_digest:
            raise ValueError(
                "Estimator/deployment collection implementation mismatch: "
                f"trained={trained_digest}, deployed={current_digest}."
            )
        self.estimator_policy_identity = {
            field: caar_identity[field] for field in IDENTITY_FIELDS
        }
        self.estimator_behavior_contract = dict(contract)
        self.estimator_behavior_contract_sha256 = caar_identity[
            "behavior_contract_sha256"
        ]
        self.training_horizons = caar_identity["training_horizons"]
        self.estimator_coordinate_encoding = caar_identity[
            "coordinate_encoding"
        ]

    @staticmethod
    def _static_obstacle_mask(map_value) -> np.ndarray:
        """Parse ``GridConfig.map`` without consulting a map name."""

        if isinstance(map_value, str):
            rows = [row.strip() for row in map_value.splitlines() if row.strip()]
            if not rows:
                raise ValueError("grid_config.map is empty.")
            if len({len(row) for row in rows}) != 1:
                raise ValueError("grid_config.map must be rectangular.")
            if any(set(row) - {".", "#"} for row in rows):
                raise ValueError("grid_config.map must contain only '.' and '#'.")
            return np.asarray(
                [[cell == "#" for cell in row] for row in rows],
                dtype=bool,
            )

        array = np.asarray(map_value)
        if array.ndim == 1 and all(
            isinstance(row, str) for row in array.tolist()
        ):
            return CAARLS._static_obstacle_mask("\n".join(array.tolist()))
        if array.ndim != 2 or not array.size:
            raise ValueError("grid_config.map must be a non-empty 2-D grid.")
        if array.dtype.kind in "bui":
            if np.any((array != 0) & (array != 1)):
                raise ValueError("numeric grid_config.map must contain 0/1.")
            return array.astype(bool, copy=False)
        symbols = array.astype(str)
        if np.any((symbols != ".") & (symbols != "#")):
            raise ValueError("grid_config.map must contain only '.' and '#'.")
        return symbols == "#"

    @staticmethod
    def _topology_metrics(obstacles: np.ndarray) -> dict:
        """Return map-only road features using in-bounds four-neighbours."""

        if obstacles.ndim != 2 or not obstacles.size:
            raise ValueError("The static obstacle mask must be non-empty and 2-D.")
        obstacles = np.asarray(obstacles, dtype=bool)
        free = ~obstacles
        free_count = int(np.count_nonzero(free))
        obstacle_count = int(np.count_nonzero(obstacles))

        free4 = np.zeros_like(free)
        if obstacles.shape[0] >= 3 and obstacles.shape[1] >= 3:
            free4[1:-1, 1:-1] = (
                free[1:-1, 1:-1]
                & free[:-2, 1:-1]
                & free[2:, 1:-1]
                & free[1:-1, :-2]
                & free[1:-1, 2:]
            )

        obstacle_neighbours = np.zeros(obstacles.shape, dtype=np.uint8)
        obstacle_neighbours[1:, :] += obstacles[:-1, :]
        obstacle_neighbours[:-1, :] += obstacles[1:, :]
        obstacle_neighbours[:, 1:] += obstacles[:, :-1]
        obstacle_neighbours[:, :-1] += obstacles[:, 1:]
        dense_obstacles = obstacles & (obstacle_neighbours >= 3)

        return {
            "shape": (int(obstacles.shape[0]), int(obstacles.shape[1])),
            "free_cell_count": free_count,
            "obstacle_cell_count": obstacle_count,
            "open4_count": int(np.count_nonzero(free4)),
            "dense_obstacle_count": int(np.count_nonzero(dense_obstacles)),
            "open4_ratio": (
                float(np.count_nonzero(free4)) / free_count
                if free_count
                else 0.0
            ),
            "dense_obstacle_ratio": (
                float(np.count_nonzero(dense_obstacles)) / obstacle_count
                if obstacle_count
                else 0.0
            ),
        }

    def _configure_road_topology(self, grid_config) -> None:
        self._road_open4_ratio = None
        self._road_dense_obstacle_ratio = None
        self._road_open4_count = None
        self._road_dense_obstacle_count = None
        self._road_free_cell_count = None
        self._road_obstacle_cell_count = None
        self._road_map_shape = None
        self._road_agent_density = None
        self._road_topology_detected = False
        self._road_density_gate_active = False
        self._effective_reverse_caar_cooldown_steps = int(
            self.cfg.reverse_caar_cooldown_steps
        )

        feature_requested = bool(
            self.cfg.road_topology_adaptive_cooldown_enabled
            or self.cfg.road_caar_only_density_threshold is not None
        )
        map_value = getattr(grid_config, "map", None)
        if map_value is None:
            if feature_requested:
                raise ValueError(
                    "Road-topology adaptation requires grid_config.map."
                )
            return

        try:
            metrics = self._topology_metrics(
                self._static_obstacle_mask(map_value)
            )
        except (TypeError, ValueError) as exc:
            if feature_requested:
                raise ValueError(
                    "Could not compute road topology from grid_config.map."
                ) from exc
            return

        self._road_open4_ratio = metrics["open4_ratio"]
        self._road_dense_obstacle_ratio = metrics["dense_obstacle_ratio"]
        self._road_open4_count = metrics["open4_count"]
        self._road_dense_obstacle_count = metrics["dense_obstacle_count"]
        self._road_free_cell_count = metrics["free_cell_count"]
        self._road_obstacle_cell_count = metrics["obstacle_cell_count"]
        self._road_map_shape = metrics["shape"]
        self._road_topology_detected = bool(
            self._road_open4_ratio >= self.cfg.road_open4_threshold
            and self._road_dense_obstacle_ratio
            >= self.cfg.road_dense_obstacle_threshold
        )

        num_agents = getattr(grid_config, "num_agents", None)
        if num_agents is not None and self._road_free_cell_count:
            self._road_agent_density = (
                float(num_agents) / self._road_free_cell_count
            )
        elif self.cfg.road_caar_only_density_threshold is not None:
            raise ValueError(
                "Road density gating requires num_agents and free map cells."
            )

        if (
            self.cfg.road_topology_adaptive_cooldown_enabled
            and self._road_topology_detected
        ):
            self._effective_reverse_caar_cooldown_steps = int(
                self.cfg.road_reverse_caar_cooldown_steps
            )
        threshold = self.cfg.road_caar_only_density_threshold
        self._road_density_gate_active = bool(
            threshold is not None
            and self._road_topology_detected
            and self._road_agent_density is not None
            and self._road_agent_density >= threshold
        )

    def set_grid_config(self, grid_config):
        horizon = int(getattr(grid_config, "max_episode_steps"))
        if horizon not in self.training_horizons:
            raise ValueError(
                "CAAR-LS evaluation horizon was absent from estimator "
                f"training: evaluation={horizon}, "
                f"training={list(self.training_horizons)}."
            )
        obs_radius = int(getattr(grid_config, "obs_radius"))
        expected_shape = tuple(
            int(value)
            for value in self.estimator_behavior_contract.get("obs_shape", ())
        )
        # Three matrix channels plus the pre-action Shared Traffic Trace.
        actual_shape = (4, 2 * obs_radius + 1, 2 * obs_radius + 1)
        if expected_shape != actual_shape:
            raise ValueError(
                "CAAR-LS observation shape differs from estimator training: "
                f"evaluation={actual_shape}, training={expected_shape}."
            )
        self._configure_road_topology(grid_config)
        self.caar.set_grid_config(grid_config)

    def set_env(self, env):
        self.env = env
        self.caar.set_env(env)

    def after_reset(self):
        self.caar.after_reset()
        _freeze_policy_parameters(self.caar)
        self.plan_candidates.reset()
        self._nominal_branch: list[int] | None = None
        self._branch_initialized: list[bool] | None = None
        self._reverse_caar_cooldown_remaining: list[int] | None = None

        self.environment_step_count = 0
        self.total_action_count = 0
        self.value_comparison_count = 0
        self.branch_switch_count = 0
        self.nominal_caar_count = 0
        self.nominal_ao_count = 0
        self.executed_caar_count = 0
        self.executed_ao_count = 0
        self.reverse_count = 0
        self.reverse_override_count = 0
        self.reverse_ao_executed_count = 0
        self.none_count = 0
        self.none_override_count = 0
        self.invalid_plan_count = 0
        self.invalid_override_count = 0
        self.nonfinite_value_count = 0
        self.nonfinite_caar_value_count = 0
        self.nonfinite_ao_value_count = 0
        self.plan_commit_count = 0
        self.caar_agreement_commit_count = 0
        self.plan_caar_agreement_count = 0
        self.nominal_ao_agreement_count = 0
        self.forced_caar_count = 0
        self.reverse_caar_cooldown_trigger_count = 0
        self.reverse_caar_cooldown_action_count = 0
        self.reverse_caar_cooldown_followup_action_count = 0
        self.max_reverse_caar_cooldown_remaining = 0
        self.road_density_gate_forced_nominal_count = 0
        self.final_none_action_count = 0
        self.max_concurrent_nominal_ao = 0
        self.max_concurrent_ao_executed = 0

    def _ensure_agent_state(self, count: int) -> None:
        if self._nominal_branch is None:
            self._nominal_branch = [CAAR_BRANCH] * count
            self._branch_initialized = [False] * count
            self._reverse_caar_cooldown_remaining = [0] * count
            return
        if len(self._nominal_branch) != count:
            raise ValueError(
                "CAAR-LS agent count changed without an environment reset."
            )
        if (
            self._reverse_caar_cooldown_remaining is None
            or len(self._reverse_caar_cooldown_remaining) != count
        ):
            raise RuntimeError("CAAR-LS reverse cooldown state is invalid.")

    @staticmethod
    def _as_value_array(estimator, observations, count: int, label: str):
        predict = getattr(estimator, "predict", None)
        if not callable(predict):
            raise TypeError(f"The {label} estimator must expose predict().")
        values = predict(observations)
        detach = getattr(values, "detach", None)
        if callable(detach):
            values = detach()
            cpu = getattr(values, "cpu", None)
            if callable(cpu):
                values = cpu()
        array = np.asarray(values, dtype=np.float64)
        if array.shape == (count, 1):
            array = array[:, 0]
        if array.shape != (count,):
            raise RuntimeError(
                f"The {label} estimator returned shape {array.shape}, "
                f"expected ({count},)."
            )
        return array

    def _select_branches(self, raw_observations) -> None:
        count = len(raw_observations)
        base_observations = MatrixObservationWrapper.to_matrix(
            raw_observations
        )
        get_augmented = getattr(
            self.caar,
            "last_augmented_observations",
            None,
        )
        if not callable(get_augmented):
            raise RuntimeError(
                "Trace-aware CAAR-LS requires "
                "CAAR.last_augmented_observations()."
            )
        augmented_observations = get_augmented()
        if augmented_observations is None or len(augmented_observations) != count:
            raise RuntimeError(
                "CAAR did not expose one trace-augmented observation per agent."
            )
        matrix_observations = []
        for base, augmented in zip(base_observations, augmented_observations):
            base_obs = np.asarray(base["obs"], dtype=np.float32)
            augmented_base = np.asarray(augmented.get("obs"), dtype=np.float32)
            tau = np.asarray(augmented.get("tau"), dtype=np.float32)
            if augmented_base.shape != base_obs.shape or not np.array_equal(
                augmented_base,
                base_obs,
            ):
                raise RuntimeError(
                    "CAAR's Shared Traffic Trace is not aligned with o_t."
                )
            if tau.shape != (1, base_obs.shape[1], base_obs.shape[2]):
                raise RuntimeError(
                    "Shared Traffic Trace must have shape (1, height, width)."
                )
            if not bool(np.all(np.isfinite(tau))):
                raise RuntimeError("Shared Traffic Trace contains non-finite values.")
            matrix_observations.append(
                {
                    "obs": np.concatenate((base_obs, tau), axis=0),
                    "xy": np.asarray(base["xy"], dtype=np.float32),
                    "target_xy": np.asarray(
                        base["target_xy"],
                        dtype=np.float32,
                    ),
                }
            )
        caar_values = self._as_value_array(
            self.caar_estimator,
            matrix_observations,
            count,
            "CAAR",
        )
        ao_values = self._as_value_array(
            self.ao_estimator,
            matrix_observations,
            count,
            "AO-safe",
        )
        choices, nonfinite = select_ao_by_absolute_return(
            caar_values,
            ao_values,
            margin=self.cfg.value_margin,
        )

        for index in range(count):
            previous = self._nominal_branch[index]
            initialized = self._branch_initialized[index]
            cooldown_active = bool(
                self._reverse_caar_cooldown_remaining[index] > 0
            )
            candidate = (
                CAAR_BRANCH
                if cooldown_active or self._road_density_gate_active
                else int(choices[index])
            )
            if self._road_density_gate_active and int(choices[index]) == AO_BRANCH:
                self.road_density_gate_forced_nominal_count += 1
            if initialized and candidate != previous:
                self.branch_switch_count += 1
            self._nominal_branch[index] = candidate
            self._branch_initialized[index] = True
            self.value_comparison_count += 1
            if nonfinite[index]:
                self.nonfinite_value_count += 1
                self.nonfinite_caar_value_count += int(
                    not np.isfinite(caar_values[index])
                )
                self.nonfinite_ao_value_count += int(
                    not np.isfinite(ao_values[index])
                )

    def _coerce_caar_actions(self, actions, count: int) -> tuple[int, ...]:
        values = np.asarray(actions, dtype=object).reshape(-1)
        if values.shape != (count,):
            raise RuntimeError(
                f"CAAR returned {len(values)} actions for {count} agents."
            )
        result = []
        for action in values:
            if action is None:
                raise RuntimeError("CAAR returned a None action.")
            try:
                integer = int(action)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(f"CAAR returned an invalid action: {action!r}.") from exc
            if integer != action or not 0 <= integer < self.action_count:
                raise RuntimeError(f"CAAR returned an invalid action: {action!r}.")
            result.append(integer)
        return tuple(result)

    def _classify_plan_batch(
        self,
        batch: RawPlanBatch,
        count: int,
    ) -> tuple[tuple[int | None, ...], tuple[bool, ...], tuple[bool, ...]]:
        if not (
            len(batch.actions)
            == len(batch.planned_mask)
            == len(batch.reverse_mask)
            == count
        ):
            raise RuntimeError("Raw Plan returned the wrong action count.")

        converted: list[int | None] = []
        none_mask: list[bool] = []
        invalid_mask: list[bool] = []
        for action, planned in zip(batch.actions, batch.planned_mask):
            is_none = action is None
            integer = None
            valid_integer = False
            if not is_none:
                try:
                    integer = int(action)
                    valid_integer = integer == action
                except (TypeError, ValueError, OverflowError):
                    integer = None
            valid = bool(
                planned
                and valid_integer
                and integer is not None
                and 0 <= integer < self.action_count
            )
            converted.append(integer if valid else None)
            none_mask.append(is_none)
            invalid_mask.append(bool(not is_none and not valid))
        return tuple(converted), tuple(none_mask), tuple(invalid_mask)

    def _cancel_pending_plan(self, count: int) -> None:
        if getattr(self.plan_candidates, "pending", None) is not None:
            self.plan_candidates.commit([False] * count)

    def act(self, observations, rewards=None, dones=None, infos=None):
        raw_observations = tuple(observations)
        count = len(raw_observations)
        self._ensure_agent_state(count)

        caar_actions = self._coerce_caar_actions(
            self.caar.act(raw_observations, rewards, dones, infos),
            count,
        )
        plan_batch = self.plan_candidates.propose(raw_observations)
        try:
            plan_actions, none_mask, invalid_mask = self._classify_plan_batch(
                plan_batch,
                count,
            )
            self._select_branches(raw_observations)

            final_actions: list[int] = []
            executed_ao_mask: list[bool] = []
            commit_mask: list[bool] = []
            reverse_mask = tuple(bool(value) for value in plan_batch.reverse_mask)
            cooldown_active_mask = tuple(
                remaining > 0
                for remaining in self._reverse_caar_cooldown_remaining
            )
            reverse_cooldown_trigger_mask: list[bool] = []

            for index in range(count):
                nominal_ao = self._nominal_branch[index] == AO_BRANCH
                reverse_cooldown_trigger_mask.append(
                    bool(
                        self.cfg.reverse_caar_override_enabled
                        and self._effective_reverse_caar_cooldown_steps > 0
                        and nominal_ao
                        and reverse_mask[index]
                    )
                )
                usable_plan = (
                    plan_actions[index] is not None
                    and (
                        not reverse_mask[index]
                        or not self.cfg.reverse_caar_override_enabled
                    )
                )
                execute_ao = bool(nominal_ao and usable_plan)
                final_action = (
                    int(plan_actions[index])
                    if execute_ao
                    else caar_actions[index]
                )
                final_actions.append(final_action)
                executed_ao_mask.append(execute_ao)

                # Keep the raw planner synchronized with the physical action.
                # A valid agreement under nominal CAAR may advance it.  A
                # reverse proposal is executable and committable only in the
                # predictor-only variant.
                commit_mask.append(
                    bool(
                        usable_plan
                        and int(plan_actions[index]) == final_action
                    )
                )

            for index, active in enumerate(cooldown_active_mask):
                if active:
                    self._reverse_caar_cooldown_remaining[index] -= 1
                if reverse_cooldown_trigger_mask[index]:
                    # The reverse-fallback step itself is step one of the
                    # configured lock, leaving only N-1 future CAAR steps.
                    self._reverse_caar_cooldown_remaining[index] = max(
                        self._effective_reverse_caar_cooldown_steps - 1,
                        0,
                    )
            self.max_reverse_caar_cooldown_remaining = max(
                self.max_reverse_caar_cooldown_remaining,
                max(self._reverse_caar_cooldown_remaining, default=0),
            )

            self.plan_candidates.commit(
                commit_mask,
                allow_reverse=not self.cfg.reverse_caar_override_enabled,
            )
        except Exception:
            self._cancel_pending_plan(count)
            raise

        nominal_ao_mask = [branch == AO_BRANCH for branch in self._nominal_branch]
        concurrent_nominal = sum(nominal_ao_mask)
        concurrent_executed = sum(executed_ao_mask)

        self.environment_step_count += 1
        self.total_action_count += count
        self.nominal_ao_count += concurrent_nominal
        self.nominal_caar_count += count - concurrent_nominal
        self.executed_ao_count += concurrent_executed
        self.executed_caar_count += count - concurrent_executed
        self.reverse_count += sum(reverse_mask)
        self.none_count += sum(none_mask)
        self.invalid_plan_count += sum(invalid_mask)
        self.reverse_override_count += sum(
            self.cfg.reverse_caar_override_enabled and nominal and reverse
            for nominal, reverse in zip(nominal_ao_mask, reverse_mask)
        )
        self.none_override_count += sum(
            nominal and is_none
            for nominal, is_none in zip(nominal_ao_mask, none_mask)
        )
        self.invalid_override_count += sum(
            nominal and invalid
            for nominal, invalid in zip(nominal_ao_mask, invalid_mask)
        )
        self.reverse_ao_executed_count += sum(
            executed and reverse
            for executed, reverse in zip(executed_ao_mask, reverse_mask)
        )
        self.forced_caar_count += sum(
            nominal and not executed
            for nominal, executed in zip(nominal_ao_mask, executed_ao_mask)
        )
        cooldown_triggers = sum(reverse_cooldown_trigger_mask)
        cooldown_followups = sum(cooldown_active_mask)
        self.reverse_caar_cooldown_trigger_count += cooldown_triggers
        self.reverse_caar_cooldown_followup_action_count += (
            cooldown_followups
        )
        self.reverse_caar_cooldown_action_count += (
            cooldown_triggers + cooldown_followups
        )
        agreement_mask = [
            plan is not None
            and (
                not reverse
                or not self.cfg.reverse_caar_override_enabled
            )
            and plan == caar
            for plan, reverse, caar in zip(
                plan_actions,
                reverse_mask,
                caar_actions,
            )
        ]
        self.plan_caar_agreement_count += sum(agreement_mask)
        self.nominal_ao_agreement_count += sum(
            nominal and agreement
            for nominal, agreement in zip(nominal_ao_mask, agreement_mask)
        )
        self.plan_commit_count += sum(commit_mask)
        self.caar_agreement_commit_count += sum(
            (not nominal) and committed
            for nominal, committed in zip(nominal_ao_mask, commit_mask)
        )
        self.final_none_action_count += sum(
            action is None for action in final_actions
        )
        self.max_concurrent_nominal_ao = max(
            self.max_concurrent_nominal_ao,
            concurrent_nominal,
        )
        self.max_concurrent_ao_executed = max(
            self.max_concurrent_ao_executed,
            concurrent_executed,
        )

        return final_actions

    def after_step(self, dones: Sequence[bool]):
        done_flags = tuple(bool(value) for value in dones)
        self.caar.after_step(done_flags)
        if self._nominal_branch is not None:
            if len(done_flags) != len(self._nominal_branch):
                raise ValueError("Done mask and CAAR-LS agent count differ.")
            for index, done in enumerate(done_flags):
                if done:
                    self._nominal_branch[index] = CAAR_BRANCH
                    self._branch_initialized[index] = False
                    self._reverse_caar_cooldown_remaining[index] = 0
        if done_flags and all(done_flags):
            self.plan_candidates.reset()

    @staticmethod
    def _ratio(numerator: int, denominator: int):
        return numerator / denominator if denominator else None

    def get_switch_stats(self):
        cooldown_steps = self._effective_reverse_caar_cooldown_steps
        reverse_override_enabled = bool(
            self.cfg.reverse_caar_override_enabled
        )
        return {
            "hybrid_mode": (
                PREDICTOR_ONLY_HYBRID_MODE
                if not reverse_override_enabled
                else (
                    REVERSE_COOLDOWN_HYBRID_MODE
                    if cooldown_steps > 0
                    else HYBRID_MODE
                )
            ),
            "switch_pair": ["CAAR", "AO-RePlan"],
            "comparison_cadence": "every_step_per_agent",
            "switch_constraint": (
                "reverse_caar_cooldown" if cooldown_steps > 0 else "none"
            ),
            "value_margin": self.cfg.value_margin,
            "reverse_caar_override_enabled": reverse_override_enabled,
            "reverse_caar_cooldown_steps": cooldown_steps,
            "base_reverse_caar_cooldown_steps": int(
                self.cfg.reverse_caar_cooldown_steps
            ),
            "reverse_caar_cooldown_includes_trigger_step": True,
            "road_topology_adaptive_cooldown_enabled": bool(
                self.cfg.road_topology_adaptive_cooldown_enabled
            ),
            "road_reverse_caar_cooldown_steps": int(
                self.cfg.road_reverse_caar_cooldown_steps
            ),
            "road_open4_threshold": float(
                self.cfg.road_open4_threshold
            ),
            "road_dense_obstacle_threshold": float(
                self.cfg.road_dense_obstacle_threshold
            ),
            "road_open4": self._road_open4_ratio,
            "road_dense_obstacle": self._road_dense_obstacle_ratio,
            "road_open4_count": self._road_open4_count,
            "road_dense_obstacle_count": self._road_dense_obstacle_count,
            "road_free_cells": self._road_free_cell_count,
            "road_obstacle_cells": self._road_obstacle_cell_count,
            "road_map_shape": (
                list(self._road_map_shape)
                if self._road_map_shape is not None
                else None
            ),
            "road_like": self._road_topology_detected,
            "road_agent_density": self._road_agent_density,
            "road_caar_only_density_threshold": (
                self.cfg.road_caar_only_density_threshold
            ),
            "density_gate_active": self._road_density_gate_active,
            "road_density_gate_forced_nominal_count": (
                self.road_density_gate_forced_nominal_count
            ),
            "road_cooldown_source": (
                "road_topology"
                if (
                    self.cfg.road_topology_adaptive_cooldown_enabled
                    and self._road_topology_detected
                )
                else "base"
            ),
            "road_topology_provenance": {
                "schema_version": ROAD_TOPOLOGY_PROVENANCE_VERSION,
                "source": "grid_config.map",
                "uses_map_name": False,
                "neighbourhood": "four_cardinal_in_bounds",
                "open4_denominator": "free_cells",
                "dense_obstacle_denominator": "obstacle_cells",
                "road_decision": (
                    "open4>=road_open4_threshold and "
                    "dense_obstacle>=road_dense_obstacle_threshold"
                ),
                "agent_density_definition": "num_agents/free_cells",
            },
            "environment_step_count": self.environment_step_count,
            "total_action_count": self.total_action_count,
            "total_actions": self.total_action_count,
            "value_comparison_count": self.value_comparison_count,
            "branch_switch_count": self.branch_switch_count,
            "nominal_caar_count": self.nominal_caar_count,
            "nominal_ao_count": self.nominal_ao_count,
            "nominal_ao_rate": self._ratio(
                self.nominal_ao_count,
                self.total_action_count,
            ),
            "executed_caar_count": self.executed_caar_count,
            "executed_ao_count": self.executed_ao_count,
            "executed_ao_rate": self._ratio(
                self.executed_ao_count,
                self.total_action_count,
            ),
            "reverse_count": self.reverse_count,
            "reverse_override_count": self.reverse_override_count,
            "reverse_ao_executed_count": self.reverse_ao_executed_count,
            "none_count": self.none_count,
            "none_override_count": self.none_override_count,
            "invalid_plan_count": self.invalid_plan_count,
            "invalid_override_count": self.invalid_override_count,
            "nonfinite_value_count": self.nonfinite_value_count,
            "nonfinite_caar_value_count": self.nonfinite_caar_value_count,
            "nonfinite_ao_value_count": self.nonfinite_ao_value_count,
            "plan_commit_count": self.plan_commit_count,
            "caar_agreement_commit_count": self.caar_agreement_commit_count,
            "plan_caar_agreement_count": self.plan_caar_agreement_count,
            "nominal_ao_agreement_count": self.nominal_ao_agreement_count,
            "forced_caar_count": self.forced_caar_count,
            "reverse_caar_cooldown_trigger_count": (
                self.reverse_caar_cooldown_trigger_count
            ),
            "reverse_caar_cooldown_action_count": (
                self.reverse_caar_cooldown_action_count
            ),
            "reverse_caar_cooldown_followup_action_count": (
                self.reverse_caar_cooldown_followup_action_count
            ),
            "max_reverse_caar_cooldown_remaining": (
                self.max_reverse_caar_cooldown_remaining
            ),
            "probe_call_count": 0,
            "final_none_action_count": self.final_none_action_count,
            "max_concurrent_nominal_ao": self.max_concurrent_nominal_ao,
            "max_concurrent_ao_executed": self.max_concurrent_ao_executed,
            "max_concurrent_ao": self.max_concurrent_ao_executed,
        }

    def get_additional_info(self):
        return self.get_switch_stats()

    def get_action_correction_stats(self):
        getter = getattr(self.caar, "get_action_correction_stats", None)
        return getter() if callable(getter) else {}


# Descriptive alias for callers that prefer the full algorithm name.
CAARLSwitcher = CAARLS


__all__ = [
    "AO_BRANCH",
    "CAAR_BRANCH",
    "CAARLS",
    "CAARLSConfig",
    "CAARLSwitcher",
    "HYBRID_MODE",
    "REVERSE_COOLDOWN_HYBRID_MODE",
    "select_ao_by_absolute_return",
]
