"""Learnable per-step switcher between CAAR and raw planner proposals.

The switcher follows the value-evaluation design from *When to Switch*: two
independent estimators predict absolute Monte-Carlo returns on every step.  The
larger prediction selects the nominal branch immediately.  A missing or invalid
raw Plan proposal is replaced for that step by CAAR.  Reverse proposals can
either use the default CAAR safety override or execute exactly as selected by
the predictor.  The branch values are compared again on the next step.  The
switcher never constructs or calls AO-RePlan's Probe.
"""

from __future__ import annotations

from typing import Callable, Literal, Sequence

import numpy as np
from pydantic import Extra, Field, validator

from agents.caar import CAAR, CAARConfig
from agents.utils_agents import AlgoBase
from planning.raw_aoreplan_candidates import RawAORePlanCandidates, RawPlanBatch
from pomapf_env.wrappers import MatrixObservationWrapper


CAAR_BRANCH = 0
AO_BRANCH = 1
HYBRID_MODE = "per_step_absolute_return_srlsm_reverse_to_caar_v1"
PREDICTOR_ONLY_HYBRID_MODE = (
    "per_step_absolute_return_srlsm_predictor_only_v1"
)
DEFAULT_ACTION_COUNT = 5


def _freeze_policy_parameters(policy) -> None:
    """Freeze weights without changing CAAR's deployed normalizer mode."""

    actor = getattr(policy, "ppo", None)
    parameters = getattr(actor, "parameters", None)
    if callable(parameters):
        for parameter in parameters():
            parameter.requires_grad_(False)


class SRSLMConfig(AlgoBase, extra=Extra.forbid):
    """Deployment configuration for the absolute-return learnable switcher."""

    name: Literal["SRSLM"] = "SRSLM"
    hybrid_mode: Literal[
        "per_step_absolute_return_srlsm_reverse_to_caar_v1",
        "per_step_absolute_return_srlsm_predictor_only_v1",
    ] = HYBRID_MODE
    caar: CAARConfig = CAARConfig(
        path_to_weights="weights/CAAR/radius_ablation/R5",
        checkpoint_kind="latest",
    )
    caar_estimator_checkpoint_path: str = "weights/SRSLM-v1/caar_estimator.pth"
    ao_estimator_checkpoint_path: str = "weights/SRSLM-v1/ao_estimator.pth"
    estimator_device: str = "auto"
    value_margin: float = 0.0
    reverse_caar_override_enabled: bool = True
    plan_use_best_move: bool = True
    max_planning_steps: int = Field(10_000, gt=0)

    @validator("value_margin")
    def finite_margin(cls, value):
        if not np.isfinite(value):
            raise ValueError("value_margin must be finite.")
        return float(value)


def _default_estimator_factory(**kwargs):
    """Lazily import the estimator so injected unit tests need no checkpoint."""

    from policy_estimation.model import PolicyReturnEstimator

    return PolicyReturnEstimator(**kwargs)


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
    """Allow reverse commits only for predictor-only SRSLM deployment.

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


class SRSLM:
    """Per-step absolute-return switcher over CAAR and raw AO-RePlan."""

    def __init__(
        self,
        cfg: SRSLMConfig,
        *,
        caar_factory: Callable[[CAARConfig], object] = CAAR,
        planner_factory: Callable[..., object] = _DeploymentRawAORePlanCandidates,
        estimator_factory: Callable[..., object] | None = None,
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
            raise ValueError("SRSLM requires two independent value estimators.")
        self._freeze_estimator(self.caar_estimator)
        self._freeze_estimator(self.ao_estimator)

        self.device = getattr(self.caar, "device", cfg.device)
        self.env = None
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

    def set_grid_config(self, grid_config):
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
        self.final_none_action_count = 0
        self.max_concurrent_nominal_ao = 0
        self.max_concurrent_ao_executed = 0

    def _ensure_agent_state(self, count: int) -> None:
        if self._nominal_branch is None:
            self._nominal_branch = [CAAR_BRANCH] * count
            self._branch_initialized = [False] * count
            return
        if len(self._nominal_branch) != count:
            raise ValueError(
                "SRSLM agent count changed without an environment reset."
            )

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
                "Trace-aware SRSLM requires "
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
            candidate = int(choices[index])
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
            for index in range(count):
                nominal_ao = self._nominal_branch[index] == AO_BRANCH
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
                raise ValueError("Done mask and SRSLM agent count differ.")
            for index, done in enumerate(done_flags):
                if done:
                    self._nominal_branch[index] = CAAR_BRANCH
                    self._branch_initialized[index] = False
        if done_flags and all(done_flags):
            self.plan_candidates.reset()

    @staticmethod
    def _ratio(numerator: int, denominator: int):
        return numerator / denominator if denominator else None

    def get_switch_stats(self):
        reverse_override_enabled = bool(
            self.cfg.reverse_caar_override_enabled
        )
        return {
            "hybrid_mode": (
                PREDICTOR_ONLY_HYBRID_MODE
                if not reverse_override_enabled
                else HYBRID_MODE
            ),
            "switch_pair": ["CAAR", "AO-RePlan"],
            "comparison_cadence": "every_step_per_agent",
            "switch_constraint": (
                "reverse_to_caar_current_step"
                if reverse_override_enabled
                else "none"
            ),
            "value_margin": self.cfg.value_margin,
            "reverse_caar_override_enabled": reverse_override_enabled,
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
SRSLMSwitcher = SRSLM


__all__ = [
    "AO_BRANCH",
    "CAAR_BRANCH",
    "SRSLM",
    "SRSLMConfig",
    "SRSLMSwitcher",
    "HYBRID_MODE",
    "select_ao_by_absolute_return",
]
