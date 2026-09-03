"""Reverse-action statistics based on executed position history."""

from __future__ import annotations

from collections.abc import Sequence


class ExecutedPositionReverseCounter:
    """Count proposals that return to the previous timestep's position.

    The observed position is recorded every step, including waits and blocked
    moves, and is reset when the target assignment changes.
    """

    METRIC_VERSION = "previous_timestep_position_target_segment_v3"

    def __init__(self, moves: Sequence[Sequence[int]]):
        self.moves = tuple(
            (int(move[0]), int(move[1])) for move in moves
        )
        self.reset()

    @staticmethod
    def _point(observation, key: str) -> tuple[int, int]:
        value = observation[key]
        return int(value[0]), int(value[1])

    def reset(self) -> None:
        self._previous_position: list[tuple[int, int] | None] | None = None
        self._last_target: list[tuple[int, int] | None] | None = None
        self.reverse_count = 0
        self.movement_count = 0

    def _ensure_state(self, count: int) -> None:
        if self._previous_position is None:
            self._previous_position = [None] * count
            self._last_target = [None] * count
            return
        if len(self._previous_position) != count:
            raise ValueError(
                "The number of agents changed without resetting reverse statistics."
            )

    def record(self, actions, observations) -> None:
        if len(actions) != len(observations):
            raise ValueError("Actions and observations must have equal lengths.")
        self._ensure_state(len(actions))

        for index, (action, observation) in enumerate(zip(actions, observations)):
            target = self._point(observation, "target_xy")
            if self._last_target[index] != target:
                self._previous_position[index] = None
                self._last_target[index] = target

            position = self._point(observation, "xy")
            previous = self._previous_position[index]
            self._previous_position[index] = position

            if action in (None, 0):
                continue
            try:
                dx, dy = self.moves[int(action)]
            except (IndexError, TypeError, ValueError):
                continue

            self.movement_count += 1
            candidate = position[0] + dx, position[1] + dy
            if previous is not None and candidate == previous:
                self.reverse_count += 1

    @property
    def rate(self) -> float:
        if self.movement_count == 0:
            return 0.0
        return self.reverse_count / self.movement_count
