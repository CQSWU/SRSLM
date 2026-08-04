import unittest

import numpy as np

from planning.raw_aoreplan_candidates import RawAORePlanCandidates


class _SequenceBase:
    def __init__(self, *, sequences, **_kwargs):
        self._sequences = iter(sequences)
        self.commits = []

    def act(self, observations, skip_agents=None):
        actions = list(next(self._sequences))
        return [
            None if bool(skip_agents[index]) else action
            for index, action in enumerate(actions)
        ]

    def commit_proposals(self, executed_mask):
        self.commits.append(list(executed_mask))


def _factory(sequences, sink):
    def create(**kwargs):
        base = _SequenceBase(sequences=sequences, **kwargs)
        sink.append(base)
        return base

    return create


def _observation(position=(5, 5), target=(8, 8)):
    return {
        "obstacles": np.zeros((11, 11), dtype=np.int8),
        "agents": np.zeros((11, 11), dtype=np.int8),
        "xy": position,
        "target_xy": target,
    }


class RawAORePlanCandidatesTests(unittest.TestCase):
    def make_candidates(self, sequences):
        sink = []
        candidates = RawAORePlanCandidates(
            base_factory=_factory(sequences, sink),
        )
        return candidates, sink[0]

    def test_raw_reverse_is_exposed_and_cannot_be_committed(self):
        candidates, base = self.make_candidates([[1], [2]])

        first = candidates.propose([_observation(position=(5, 5))])
        self.assertEqual(first.actions, (1,))
        self.assertEqual(first.reverse_mask, (False,))
        candidates.commit([True])

        second = candidates.propose([_observation(position=(4, 5))])
        self.assertEqual(second.actions, (2,))
        self.assertEqual(second.reverse_mask, (True,))
        with self.assertRaisesRegex(ValueError, "fall back to CAAR"):
            candidates.commit([True])
        candidates.commit([False])

        self.assertEqual(base.commits, [[True], [False]])

    def test_blocked_command_does_not_create_false_reverse(self):
        candidates, _base = self.make_candidates([[1], [2]])
        candidates.propose([_observation(position=(5, 5))])
        candidates.commit([True])

        second = candidates.propose([_observation(position=(5, 5))])
        self.assertEqual(second.reverse_mask, (False,))
        candidates.commit([False])

    def test_target_change_clears_reverse_history(self):
        candidates, _base = self.make_candidates([[1], [2]])
        candidates.propose(
            [_observation(position=(5, 5), target=(8, 8))]
        )
        candidates.commit([True])

        second = candidates.propose(
            [_observation(position=(4, 5), target=(1, 1))]
        )
        self.assertEqual(second.reverse_mask, (False,))
        candidates.commit([False])

    def test_missing_and_wait_actions_are_never_reverse(self):
        candidates, base = self.make_candidates([[None, 0]])
        batch = candidates.propose(
            [_observation(), _observation(position=(6, 6))]
        )

        self.assertEqual(batch.planned_mask, (False, True))
        self.assertEqual(batch.reverse_mask, (False, False))
        with self.assertRaisesRegex(ValueError, "missing"):
            candidates.commit([True, False])
        candidates.commit([False, True])
        self.assertEqual(base.commits, [[False, True]])

    def test_multiple_nonconflicting_plan_actions_can_commit_together(self):
        candidates, base = self.make_candidates([[1, 4]])
        batch = candidates.propose(
            [_observation(), _observation(position=(20, 20))]
        )

        self.assertEqual(batch.reverse_mask, (False, False))
        candidates.commit([True, True])
        self.assertEqual(base.commits, [[True, True]])

    def test_proposal_must_be_resolved_before_next_step(self):
        candidates, _base = self.make_candidates([[1], [2]])
        candidates.propose([_observation()])

        with self.assertRaisesRegex(RuntimeError, "must be committed"):
            candidates.propose([_observation(position=(4, 5))])

    def test_skip_mask_preserves_position_history_without_a_proposal(self):
        candidates, base = self.make_candidates([[1], [2], [1]])
        candidates.propose([_observation(position=(5, 5))])
        candidates.commit([True])
        skipped = candidates.propose(
            [_observation(position=(4, 5))],
            skip_agents=[True],
        )
        self.assertEqual(skipped.actions, (None,))
        candidates.commit([False])

        final = candidates.propose([_observation(position=(4, 5))])
        self.assertEqual(final.reverse_mask, (False,))
        candidates.commit([True])
        self.assertEqual(base.commits, [[True], [False], [True]])


if __name__ == "__main__":
    unittest.main()
