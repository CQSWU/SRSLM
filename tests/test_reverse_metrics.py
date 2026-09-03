import unittest

from agents.reverse_metrics import ExecutedPositionReverseCounter


class ExecutedPositionReverseCounterTests(unittest.TestCase):
    MOVES = ((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0))

    @staticmethod
    def _observation(position, target=(9, 9)):
        return {"xy": list(position), "target_xy": list(target)}

    def test_wait_and_failed_move_are_recorded_as_previous_timestep(self):
        counter = ExecutedPositionReverseCounter(self.MOVES)
        counter.record([1], [self._observation((0, 0))])
        counter.record([0], [self._observation((0, 1))])
        counter.record([1], [self._observation((0, 1))])
        counter.record([2], [self._observation((0, 1))])

        self.assertEqual(counter.movement_count, 3)
        self.assertEqual(counter.reverse_count, 0)
        self.assertEqual(counter.rate, 0.0)
        self.assertEqual(
            counter.METRIC_VERSION,
            "previous_timestep_position_target_segment_v3",
        )

    def test_target_change_resets_position_history(self):
        counter = ExecutedPositionReverseCounter(self.MOVES)
        counter.record([1], [self._observation((0, 0), (0, 1))])
        counter.record([2], [self._observation((0, 1), (5, 5))])

        self.assertEqual(counter.movement_count, 2)
        self.assertEqual(counter.reverse_count, 0)

    def test_direction_change_without_return_is_not_reverse(self):
        counter = ExecutedPositionReverseCounter(self.MOVES)
        counter.record([1], [self._observation((0, 0))])
        counter.record([3], [self._observation((0, 1))])

        self.assertEqual(counter.reverse_count, 0)


if __name__ == "__main__":
    unittest.main()
