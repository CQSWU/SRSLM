"""Shared state construction and action routing for the SRSLM Switcher.

The switcher never predicts primitive MAPF actions.  CAAR and AORePlan first
produce one action each.  AORePlan waits bypass the learned selector and use
CAAR directly.  All other states are sent to the two-branch Switcher.  This
module is independent from Sample Factory so training and deployment share the
same routing and planner-feedback semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


CAAR_BRANCH = 0
AO_BRANCH = 1
NUM_BRANCHES = 2
NUM_PRIMITIVE_ACTIONS = 5
SWITCHER_DECISION_SCOPE = "aoreplan_nonwait_only"
ALL_STATE_SWITCHER_DECISION_SCOPE = "all_states"
SWITCHER_FEATURE_SCHEMA = "srslm_switcher_state_v3"
SWITCHER_CROP_SIZE = 11
SWITCHER_SPATIAL_SHAPE = (3, SWITCHER_CROP_SIZE, SWITCHER_CROP_SIZE)
SWITCHER_COORD_DIM = 2
SWITCHER_VECTOR_DIM = 2 * SWITCHER_COORD_DIM + 2 * NUM_PRIMITIVE_ACTIONS
# The total number of scalar inputs is useful for architecture reporting even
# though the encoder keeps the spatial tensor and vector fields separate.
FEATURE_DIM = int(np.prod(SWITCHER_SPATIAL_SHAPE)) + SWITCHER_VECTOR_DIM


def _one_hot(actions: np.ndarray) -> np.ndarray:
    result = np.zeros((len(actions), NUM_PRIMITIVE_ACTIONS), dtype=np.float32)
    result[np.arange(len(actions)), actions] = 1.0
    return result


def _target_layer(xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    radius = SWITCHER_CROP_SIZE // 2
    dx = int(round(float(xy[0]) - float(target_xy[0])))
    dy = int(round(float(xy[1]) - float(target_xy[1])))
    dx = min(dx, radius) if dx >= 0 else max(dx, -radius)
    dy = min(dy, radius) if dy >= 0 else max(dy, -radius)
    result = np.zeros((SWITCHER_CROP_SIZE, SWITCHER_CROP_SIZE), dtype=np.float32)
    result[radius - dx, radius - dy] = 1.0
    return result


def build_switcher_state(
    observations: Sequence[Mapping],
    caar_actions: Sequence[int],
    aoreplan_actions: Sequence[int],
) -> dict[str, np.ndarray]:
    """Build the minimal spatial state consumed by the Switcher."""

    count = len(observations)
    caar = np.asarray(caar_actions, dtype=np.int64).reshape(-1)
    aoreplan = np.asarray(aoreplan_actions, dtype=np.int64).reshape(-1)
    if caar.shape != (count,) or aoreplan.shape != (count,):
        raise RuntimeError("Switcher action arrays have inconsistent lengths.")
    if np.any((caar < 0) | (caar >= NUM_PRIMITIVE_ACTIONS)):
        raise RuntimeError("CAAR produced an invalid primitive action.")
    if np.any((aoreplan < 0) | (aoreplan >= NUM_PRIMITIVE_ACTIONS)):
        raise RuntimeError("AORePlan produced an invalid primitive action.")

    spatial_rows = []
    xy_rows = []
    target_rows = []
    expected_crop = (SWITCHER_CROP_SIZE, SWITCHER_CROP_SIZE)
    for observation in observations:
        obstacles = np.asarray(observation["obstacles"], dtype=np.float32)
        agents = np.asarray(observation["agents"], dtype=np.float32)
        xy = np.asarray(observation["xy"], dtype=np.float32).reshape(2)
        target = np.asarray(observation["target_xy"], dtype=np.float32).reshape(2)
        if obstacles.shape != expected_crop or agents.shape != expected_crop:
            raise RuntimeError(
                "Switcher requires aligned 11x11 obstacle and agent crops."
            )
        spatial_rows.append(
            np.stack(
                [
                    np.clip(obstacles, 0.0, 1.0),
                    np.clip(agents, 0.0, 1.0),
                    _target_layer(xy, target),
                ],
                axis=0,
            )
        )
        xy_rows.append(xy)
        target_rows.append(target)

    state = {
        "obs": np.asarray(spatial_rows, dtype=np.float32),
        "xy": np.asarray(xy_rows, dtype=np.float32),
        "target_xy": np.asarray(target_rows, dtype=np.float32),
        "caar_action": _one_hot(caar),
        "aoreplan_action": _one_hot(aoreplan),
    }
    expected_shapes = {
        "obs": (count, *SWITCHER_SPATIAL_SHAPE),
        "xy": (count, SWITCHER_COORD_DIM),
        "target_xy": (count, SWITCHER_COORD_DIM),
        "caar_action": (count, NUM_PRIMITIVE_ACTIONS),
        "aoreplan_action": (count, NUM_PRIMITIVE_ACTIONS),
    }
    for key, expected in expected_shapes.items():
        if state[key].shape != expected:
            raise AssertionError(
                f"Switcher field {key!r} has shape {state[key].shape}, "
                f"expected {expected}."
            )
        if not np.all(np.isfinite(state[key])):
            raise RuntimeError(f"Switcher field {key!r} is non-finite.")
    return state


@dataclass(frozen=True)
class PreparedSwitcherStep:
    observations: tuple
    caar_actions: tuple[int, ...]
    aoreplan_step: object
    aoreplan_actions: tuple[int, ...]
    switch_allowed_mask: tuple[bool, ...]
    switcher_state: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class ResolvedSwitcherStep:
    actions: tuple[int, ...]
    selected_branches: tuple[int, ...]
    executed_ao_mask: tuple[bool, ...]
    wait_bypass_mask: tuple[bool, ...]
    commit_mask: tuple[bool, ...]


class SwitcherController:
    """Own both branches and route only AORePlan moves through Switcher."""

    selector_kind = "ppo_two_branch_categorical"
    decision_scope = SWITCHER_DECISION_SCOPE
    wait_detection_enabled = True
    learned_switcher_called = True
    choice_error = (
        "Switcher choices must match the number of non-wait AORePlan actions."
    )

    def __init__(self, caar, aoreplan):
        self.caar = caar
        self.aoreplan = aoreplan
        self.env = None
        self._pending: PreparedSwitcherStep | None = None
        self.after_reset()

    def set_grid_config(self, grid_config) -> None:
        if getattr(grid_config, "collision_system", None) != "block_both":
            raise ValueError("SRSLM requires collision_system='block_both'.")
        self.caar.set_grid_config(grid_config)

    def set_env(self, env) -> None:
        self.env = env
        self.caar.set_env(env)

    def after_reset(self) -> None:
        self.caar.after_reset()
        self.aoreplan.reset()
        self._pending = None
        self.environment_step_count = 0
        self.total_action_count = 0
        self.switcher_choice_count = 0
        self.selected_ao_count = 0
        self.executed_ao_count = 0
        self.wait_bypass_count = 0
        self.branch_switch_count = 0
        self.branch_action_agreement_count = 0
        self.reverse_count = 0
        self.static_astar_query_count = 0
        self.aoreplan_commit_count = 0
        self._last_executed_ao: list[bool | None] | None = None

    @staticmethod
    def _coerce_caar_actions(actions, count: int) -> tuple[int, ...]:
        values = np.asarray(actions, dtype=object).reshape(-1)
        if values.shape != (count,):
            raise RuntimeError("CAAR returned the wrong number of actions.")
        converted = []
        for action in values:
            try:
                integer = int(action)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(f"CAAR returned invalid action {action!r}.") from exc
            if integer != action or not 0 <= integer < NUM_PRIMITIVE_ACTIONS:
                raise RuntimeError(f"CAAR returned invalid action {action!r}.")
            converted.append(integer)
        return tuple(converted)

    @staticmethod
    def _validate_aoreplan_step(step, count: int):
        fields = (step.actions, step.planned_mask, step.reverse_mask)
        if any(len(values) != count for values in fields):
            raise RuntimeError("AORePlan returned the wrong number of actions.")
        actions = []
        for action, planned in zip(step.actions, step.planned_mask):
            try:
                integer = int(action)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError("AORePlan must return a primitive action.") from exc
            if not planned or integer != action or not 0 <= integer < NUM_PRIMITIVE_ACTIONS:
                raise RuntimeError("AORePlan must return a valid primitive action.")
            actions.append(integer)
        return tuple(actions)

    def prepare_actions(self, observations, rewards=None, dones=None, infos=None):
        if self._pending is not None:
            raise RuntimeError("The previous switch decision was not applied.")
        raw_observations = tuple(observations)
        count = len(raw_observations)
        if count < 1:
            raise RuntimeError("Switcher received an empty agent batch.")
        caar_actions = self._coerce_caar_actions(
            self.caar.act(raw_observations, rewards, dones, infos),
            count,
        )
        step = self.aoreplan.propose(raw_observations)
        try:
            aoreplan_actions = self._validate_aoreplan_step(step, count)
            switcher_state = build_switcher_state(
                raw_observations,
                caar_actions,
                aoreplan_actions,
            )
        except Exception:
            self.aoreplan.commit([False] * count)
            raise
        switch_allowed = self._switch_allowed_mask(aoreplan_actions)
        self._pending = PreparedSwitcherStep(
            observations=raw_observations,
            caar_actions=caar_actions,
            aoreplan_step=step,
            aoreplan_actions=aoreplan_actions,
            switch_allowed_mask=switch_allowed,
            switcher_state=switcher_state,
        )
        return self._pending

    @staticmethod
    def _switch_allowed_mask(
        aoreplan_actions: Sequence[int],
    ) -> tuple[bool, ...]:
        return tuple(action != 0 for action in aoreplan_actions)

    @staticmethod
    def _mask(batch, name: str, count: int) -> tuple[bool, ...]:
        values = getattr(batch, name, None)
        if values is None:
            return tuple(False for _ in range(count))
        if len(values) != count:
            raise RuntimeError(f"AORePlan field {name!r} has the wrong length.")
        return tuple(bool(value) for value in values)

    def _apply_selected_branches(
        self,
        selected: Sequence[int],
        *,
        switcher_choice_count: int,
        selected_ao_count: int,
        wait_bypass_mask: Sequence[bool],
    ) -> ResolvedSwitcherStep:
        pending = self._pending
        if pending is None:
            raise RuntimeError("prepare_actions() must be called before resolve_actions().")
        count = len(pending.caar_actions)
        selected = np.asarray(selected, dtype=np.int64).reshape(-1)
        wait_bypass = np.asarray(wait_bypass_mask, dtype=bool).reshape(-1)
        if selected.shape != (count,) or np.any(
            (selected < 0) | (selected >= NUM_BRANCHES)
        ):
            raise ValueError("Resolved branch choices must cover every agent.")
        if wait_bypass.shape != (count,):
            raise ValueError("Wait-bypass diagnostics must cover every agent.")
        if not 0 <= switcher_choice_count <= count:
            raise ValueError("Switcher choice count is outside the agent batch.")
        if not 0 <= selected_ao_count <= switcher_choice_count:
            raise ValueError("Switcher AO count exceeds Switcher choices.")
        executed_ao = selected == AO_BRANCH
        final_actions = [
            pending.aoreplan_actions[index]
            if executed_ao[index]
            else pending.caar_actions[index]
            for index in range(count)
        ]

        commit = [
            bool(
                pending.aoreplan_actions[index] == final_actions[index]
            )
            for index in range(count)
        ]
        try:
            self.aoreplan.commit(commit)
        except Exception:
            self._pending = None
            raise

        if self._last_executed_ao is None:
            self._last_executed_ao = [None] * count
        elif len(self._last_executed_ao) != count:
            raise RuntimeError("Agent count changed without an environment reset.")
        self.branch_switch_count += sum(
            previous is not None and bool(previous) != bool(current)
            for previous, current in zip(self._last_executed_ao, executed_ao)
        )
        self._last_executed_ao = [bool(value) for value in executed_ao]

        self.environment_step_count += 1
        self.total_action_count += count
        self.switcher_choice_count += int(switcher_choice_count)
        self.selected_ao_count += int(selected_ao_count)
        self.executed_ao_count += int(executed_ao.sum())
        self.wait_bypass_count += int(wait_bypass.sum())
        self.branch_action_agreement_count += sum(
            left == right
            for left, right in zip(
                pending.caar_actions,
                pending.aoreplan_actions,
            )
        )
        self.reverse_count += sum(
            bool(value) for value in pending.aoreplan_step.reverse_mask
        )
        self.static_astar_query_count += sum(
            self._mask(
                pending.aoreplan_step,
                "static_astar_invoked_mask",
                count,
            )
        )
        self.aoreplan_commit_count += sum(commit)
        self._pending = None
        return ResolvedSwitcherStep(
            actions=tuple(int(value) for value in final_actions),
            selected_branches=tuple(int(value) for value in selected),
            executed_ao_mask=tuple(bool(value) for value in executed_ao),
            wait_bypass_mask=tuple(bool(value) for value in wait_bypass),
            commit_mask=tuple(bool(value) for value in commit),
        )

    def resolve_actions(self, branches: Sequence[int]) -> ResolvedSwitcherStep:
        pending = self._pending
        if pending is None:
            raise RuntimeError("prepare_actions() must be called before resolve_actions().")
        count = len(pending.caar_actions)
        switch_allowed = np.asarray(pending.switch_allowed_mask, dtype=bool)
        eligible_count = int(switch_allowed.sum())
        requested = np.asarray(branches, dtype=np.int64).reshape(-1)
        if requested.shape != (eligible_count,) or np.any(
            (requested < 0) | (requested >= NUM_BRANCHES)
        ):
            raise ValueError(self.choice_error)

        selected = np.full(count, CAAR_BRANCH, dtype=np.int64)
        selected[switch_allowed] = requested
        return self._apply_selected_branches(
            selected,
            switcher_choice_count=eligible_count,
            selected_ao_count=int((requested == AO_BRANCH).sum()),
            wait_bypass_mask=np.logical_not(switch_allowed),
        )

    def after_step(self, dones: Sequence[bool]) -> None:
        flags = tuple(bool(value) for value in dones)
        self.caar.after_step(flags)
        if self._last_executed_ao is not None:
            if len(flags) != len(self._last_executed_ao):
                raise RuntimeError("Done mask and agent count differ.")
            for index, done in enumerate(flags):
                if done:
                    self._last_executed_ao[index] = None

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    def get_stats(self) -> dict:
        return {
            "switcher_feature_schema": SWITCHER_FEATURE_SCHEMA,
            "selector_kind": self.selector_kind,
            "switcher_decision_scope": self.decision_scope,
            "wait_detection_enabled": self.wait_detection_enabled,
            "learned_switcher_called": self.learned_switcher_called,
            "joint_conflict_prediction_enabled": False,
            "environment_step_count": self.environment_step_count,
            "total_action_count": self.total_action_count,
            "switcher_choice_count": self.switcher_choice_count,
            "switcher_choice_rate": self._ratio(
                self.switcher_choice_count, self.total_action_count
            ),
            "selected_ao_count": self.selected_ao_count,
            "selected_ao_rate": self._ratio(
                self.selected_ao_count, self.switcher_choice_count
            ),
            "executed_ao_count": self.executed_ao_count,
            "executed_ao_rate": self._ratio(
                self.executed_ao_count, self.total_action_count
            ),
            "executed_caar_count": (
                self.total_action_count - self.executed_ao_count
            ),
            "aoreplan_wait_bypass_count": self.wait_bypass_count,
            "aoreplan_wait_bypass_rate": self._ratio(
                self.wait_bypass_count, self.total_action_count
            ),
            "branch_switch_count": self.branch_switch_count,
            "branch_action_agreement_count": self.branch_action_agreement_count,
            "branch_action_agreement_rate": self._ratio(
                self.branch_action_agreement_count, self.total_action_count
            ),
            "reverse_count": self.reverse_count,
            "static_astar_query_count": self.static_astar_query_count,
            "aoreplan_commit_count": self.aoreplan_commit_count,
        }


class AllStateSwitcherController(SwitcherController):
    """Route every state, including AORePlan waits, through Switcher."""

    decision_scope = ALL_STATE_SWITCHER_DECISION_SCOPE
    wait_detection_enabled = False
    choice_error = "All-state Switcher choices must match the agent batch."

    @staticmethod
    def _switch_allowed_mask(
        aoreplan_actions: Sequence[int],
    ) -> tuple[bool, ...]:
        return tuple(True for _ in aoreplan_actions)


class WaitDetectOnlyController(SwitcherController):
    """Use CAAR on AORePlan waits and AORePlan on every non-wait state."""

    selector_kind = "deterministic_wait_detect_only"
    decision_scope = "none"
    wait_detection_enabled = True
    learned_switcher_called = False

    def resolve_actions(self) -> ResolvedSwitcherStep:
        pending = self._pending
        if pending is None:
            raise RuntimeError("prepare_actions() must be called before resolve_actions().")
        switch_allowed = np.asarray(pending.switch_allowed_mask, dtype=bool)
        selected = np.where(switch_allowed, AO_BRANCH, CAAR_BRANCH)
        return self._apply_selected_branches(
            selected,
            switcher_choice_count=0,
            selected_ao_count=0,
            wait_bypass_mask=np.logical_not(switch_allowed),
        )


__all__ = [
    "AO_BRANCH",
    "ALL_STATE_SWITCHER_DECISION_SCOPE",
    "AllStateSwitcherController",
    "CAAR_BRANCH",
    "FEATURE_DIM",
    "SWITCHER_DECISION_SCOPE",
    "NUM_BRANCHES",
    "NUM_PRIMITIVE_ACTIONS",
    "SWITCHER_COORD_DIM",
    "SWITCHER_CROP_SIZE",
    "SWITCHER_FEATURE_SCHEMA",
    "SWITCHER_SPATIAL_SHAPE",
    "SWITCHER_VECTOR_DIM",
    "PreparedSwitcherStep",
    "ResolvedSwitcherStep",
    "SwitcherController",
    "WaitDetectOnlyController",
    "build_switcher_state",
]
