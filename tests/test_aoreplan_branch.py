import numpy as np
import pytest

from planning.aoreplan_branch import AO_PLANNER_POLICY, AORePlanBranch


class SequenceBase:
    def __init__(self, sequences):
        self.sequences = iter(sequences)
        self.rnd = np.random.default_rng(0)
        self.commits = []

    def act(self, observations, skip_agents=None):
        actions = list(next(self.sequences))
        return [None if skip_agents[i] else action for i, action in enumerate(actions)]

    def commit_proposals(self, executed_mask):
        self.commits.append(tuple(bool(value) for value in executed_mask))


class FixedStaticAStar:
    def __init__(self, action):
        self.action = action

    def get_action(self, observation):
        del observation
        return self.action


def obs(position=(5, 5)):
    return {
        "obstacles": np.zeros((11, 11), dtype=np.int8),
        "agents": np.zeros((11, 11), dtype=np.int8),
        "xy": position,
        "target_xy": (8, 8),
    }


def make_branch(sequences, static_action=3):
    base = SequenceBase(sequences)
    branch = AORePlanBranch(base_factory=lambda **_kwargs: base)
    branch._wrapper.static_astar = FixedStaticAStar(static_action)
    return branch, base


def test_branch_has_one_fixed_policy_and_minimal_step_fields():
    branch, _ = make_branch([[1]])
    assert AO_PLANNER_POLICY == "reverse_static_astar_with_original_no_path_fallback"
    assert branch.planner_policy == AO_PLANNER_POLICY
    step = branch.propose([obs()])
    assert set(step.__dataclass_fields__) == {
        "actions", "planned_mask", "reverse_mask", "static_astar_invoked_mask"
    }
    branch.commit([False])


def test_static_replacement_is_not_committed_to_dynamic_planner():
    branch, base = make_branch([[1], [2]], static_action=3)
    first = branch.propose([obs((5, 5))])
    assert first.actions == (1,)
    branch.commit([True])
    second = branch.propose([obs((4, 5))])
    assert second.actions == (3,)
    assert second.reverse_mask == (True,)
    assert second.static_astar_invoked_mask == (True,)
    branch.commit([True])
    assert base.commits == [(True,), (False,)]


def test_pending_step_must_be_committed_once():
    branch, _ = make_branch([[1]])
    branch.propose([obs()])
    with pytest.raises(RuntimeError):
        branch.propose([obs()])
    branch.commit([False])
    with pytest.raises(RuntimeError):
        branch.commit([False])

