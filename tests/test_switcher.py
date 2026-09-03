import numpy as np
import pytest

from agents.switcher_core import (
    AO_BRANCH,
    CAAR_BRANCH,
    SWITCHER_FEATURE_SCHEMA,
    AllStateSwitcherController,
    SwitcherController,
    WaitDetectOnlyController,
)
from planning.aoreplan_branch import AORePlanStep


class FakeCAAR:
    def __init__(self, actions):
        self.actions = actions

    def act(self, *_args):
        return list(self.actions)

    def after_reset(self):
        pass

    def after_step(self, _dones):
        pass

    def set_grid_config(self, _cfg):
        pass

    def set_env(self, _env):
        pass


class FakeAORePlan:
    def __init__(self, actions):
        self.actions = tuple(actions)
        self.commits = []

    def reset(self):
        pass

    def propose(self, observations):
        count = len(observations)
        return AORePlanStep(
            actions=self.actions,
            planned_mask=(True,) * count,
            reverse_mask=(False,) * count,
            static_astar_invoked_mask=(False,) * count,
        )

    def commit(self, mask):
        self.commits.append(tuple(mask))


def observations(count):
    return [{
        "obstacles": np.zeros((11, 11), dtype=np.float32),
        "agents": np.zeros((11, 11), dtype=np.float32),
        "xy": (i, 0),
        "target_xy": (i, 5),
    } for i in range(count)]


def test_wait_actions_bypass_switcher_and_directly_use_caar():
    planner = FakeAORePlan([0, 4, 0])
    controller = SwitcherController(FakeCAAR([1, 2, 3]), planner)
    prepared = controller.prepare_actions(observations(3))
    assert prepared.switch_allowed_mask == (False, True, False)
    result = controller.resolve_actions([AO_BRANCH])
    assert result.actions == (1, 4, 3)
    assert result.selected_branches == (CAAR_BRANCH, AO_BRANCH, CAAR_BRANCH)
    assert result.wait_bypass_mask == (True, False, True)
    stats = controller.get_stats()
    assert stats["switcher_feature_schema"] == SWITCHER_FEATURE_SCHEMA
    assert stats["switcher_choice_count"] == 1
    assert stats["aoreplan_wait_bypass_count"] == 2
    assert stats["executed_ao_count"] == 1


def test_switcher_choice_count_must_equal_nonwait_count():
    controller = SwitcherController(FakeCAAR([1, 2]), FakeAORePlan([4, 3]))
    controller.prepare_actions(observations(2))
    with pytest.raises(ValueError, match="non-wait"):
        controller.resolve_actions([AO_BRANCH])


def test_wait_bypass_never_commits_aoreplan_wait():
    planner = FakeAORePlan([0])
    controller = SwitcherController(FakeCAAR([1]), planner)
    controller.prepare_actions(observations(1))
    result = controller.resolve_actions([])
    assert result.actions == (1,)
    assert planner.commits == [(False,)]


def test_all_state_controller_sends_waits_to_the_switcher():
    planner = FakeAORePlan([0, 4])
    controller = AllStateSwitcherController(FakeCAAR([1, 2]), planner)
    prepared = controller.prepare_actions(observations(2))
    assert prepared.switch_allowed_mask == (True, True)

    result = controller.resolve_actions([AO_BRANCH, CAAR_BRANCH])

    assert result.actions == (0, 2)
    assert result.wait_bypass_mask == (False, False)
    assert planner.commits == [(True, False)]
    stats = controller.get_stats()
    assert stats["switcher_decision_scope"] == "all_states"
    assert stats["wait_detection_enabled"] is False
    assert stats["learned_switcher_called"] is True
    assert stats["switcher_choice_count"] == 2
    assert stats["aoreplan_wait_bypass_count"] == 0


def test_wait_detect_only_uses_no_learned_switcher_choices():
    planner = FakeAORePlan([0, 4, 0])
    controller = WaitDetectOnlyController(FakeCAAR([1, 2, 0]), planner)
    prepared = controller.prepare_actions(observations(3))
    assert prepared.switch_allowed_mask == (False, True, False)

    result = controller.resolve_actions()

    assert result.actions == (1, 4, 0)
    assert result.selected_branches == (CAAR_BRANCH, AO_BRANCH, CAAR_BRANCH)
    assert result.wait_bypass_mask == (True, False, True)
    assert planner.commits == [(False, True, True)]
    stats = controller.get_stats()
    assert stats["selector_kind"] == "deterministic_wait_detect_only"
    assert stats["switcher_decision_scope"] == "none"
    assert stats["learned_switcher_called"] is False
    assert stats["switcher_choice_count"] == 0
    assert stats["executed_ao_count"] == 1
    assert stats["executed_caar_count"] == 2
    assert stats["aoreplan_wait_bypass_count"] == 2
