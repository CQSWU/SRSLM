"""Pure-Python fallback for the local AORePlan planner binding.

Linux experiments continue to import :mod:`planning.planner`, the compiled
pybind11 implementation.  ``planning.ao_replan_algo`` imports this module only
on Windows, where the Linux extension cannot be loaded.  The class below
mirrors the small public interface and state transitions of ``planner.cpp``;
it is intended for tests and local demonstrations, not high-throughput runs.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable, Sequence


INF = 1_000_000_000
_NEIGHBOR_DELTAS = ((0, 1), (1, 0), (-1, 0), (0, -1))


def _pair(value: Sequence[int]) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"Expected a coordinate pair, got {value!r}.")
    return int(value[0]), int(value[1])


class planner:
    """Interface-compatible Python implementation of ``planner.cpp``."""

    def __init__(self, steps: int = 10_000):
        self.obstacles: set[tuple[int, int]] = set()
        self.other_agents: set[tuple[int, int]] = set()
        self.bad_actions: set[tuple[int, int]] = set()
        self.start = (INF, INF)
        self.desired_position = (INF, INF)
        self.goal = (INF, INF)
        self.max_steps = int(steps)

        self._open: list[tuple[int, int, int, int]] = []
        self._closed: dict[tuple[int, int], tuple[int, int]] = {}
        self._best_node = (INF, INF, 0, 0)

    def _has_desired_position(self) -> bool:
        return self.desired_position[0] < INF

    def _consume_execution_feedback(self, position) -> None:
        position = _pair(position)
        if self.desired_position != position:
            self.bad_actions.add(self.desired_position)
            if self.start == position:
                self.other_agents.update(self.bad_actions)
        else:
            self.bad_actions.clear()
        self.desired_position = (INF, INF)

    def _heuristic(self, node: tuple[int, int]) -> int:
        return abs(node[0] - self.goal[0]) + abs(node[1] - self.goal[1])

    def _neighbors(self, node: tuple[int, int]):
        for di, dj in _NEIGHBOR_DELTAS:
            neighbor = node[0] + di, node[1] + dj
            if neighbor not in self.obstacles:
                yield neighbor

    def _reset_search(self) -> None:
        self._closed.clear()
        self._open.clear()
        start_h = self._heuristic(self.start)
        # C++ Node ordering is f, then g, then i, then j.
        heapq.heappush(
            self._open,
            (start_h, 0, self.start[0], self.start[1]),
        )
        self._closed[self.start] = self.start
        self._best_node = (
            self.start[0],
            self.start[1],
            0,
            start_h,
        )

    def _compute_shortest_path(self) -> None:
        current = (INF, INF)
        steps = 0
        while (
            self._open
            and steps < self.max_steps
            and current != self.goal
        ):
            _f, g, i, j = heapq.heappop(self._open)
            current = i, j
            current_h = self._heuristic(current)
            if current_h < self._best_node[3]:
                self._best_node = (i, j, g, current_h)
            steps += 1

            for neighbor in self._neighbors(current):
                if (
                    neighbor in self._closed
                    or neighbor in self.other_agents
                ):
                    continue
                next_g = g + 1
                next_h = self._heuristic(neighbor)
                heapq.heappush(
                    self._open,
                    (
                        next_g + next_h,
                        next_g,
                        neighbor[0],
                        neighbor[1],
                    ),
                )
                # planner.cpp records the parent when a node is discovered.
                self._closed[neighbor] = current

    def update_obstacles(
        self,
        obstacles: Iterable[Sequence[int]],
        other_agents: Iterable[Sequence[int]],
        cur_pos: Sequence[int],
    ) -> None:
        cur_pos = _pair(cur_pos)
        # Static obstacles accumulate as the partially observed map is
        # revealed; dynamic agent positions are refreshed every step.
        for obstacle in obstacles:
            oi, oj = _pair(obstacle)
            self.obstacles.add((cur_pos[0] + oi, cur_pos[1] + oj))

        self.other_agents.clear()
        for agent in other_agents:
            ai, aj = _pair(agent)
            self.other_agents.add((cur_pos[0] + ai, cur_pos[1] + aj))

    def observe_position(self, position: Sequence[int]) -> None:
        if self._has_desired_position():
            self._consume_execution_feedback(position)

    def plan_path(
        self,
        start: Sequence[int],
        goal: Sequence[int],
    ) -> None:
        start = _pair(start)
        goal = _pair(goal)
        if self.start == start:
            self.other_agents.update(self.bad_actions)
        else:
            self.bad_actions.clear()
        self.start = start
        self.goal = goal
        self._reset_search()
        self._compute_shortest_path()

    def cancel_desired(self) -> None:
        self.desired_position = (INF, INF)

    def update_path(
        self,
        start: Sequence[int],
        goal: Sequence[int],
    ) -> None:
        if self._has_desired_position():
            self._consume_execution_feedback(start)
        else:
            self.bad_actions.clear()
        self.plan_path(start, goal)

    def update_static_path(
        self,
        start: Sequence[int],
        goal: Sequence[int],
    ) -> None:
        self.other_agents.clear()
        self.bad_actions.clear()
        self.start = _pair(start)
        self.goal = _pair(goal)
        self._reset_search()
        self._compute_shortest_path()

    def _selected_endpoint(self, use_best_node: bool) -> tuple[int, int]:
        if self.goal in self._closed:
            return self.goal
        if use_best_node:
            return self._best_node[0], self._best_node[1]
        return INF, INF

    def get_path(self, use_best_node: bool = True) -> list[tuple[int, int]]:
        endpoint = self._selected_endpoint(bool(use_best_node))
        path: list[tuple[int, int]] = []
        if endpoint[0] < INF and endpoint != self.start:
            next_node = endpoint
            while self._closed[next_node] != self.start:
                path.append(next_node)
                next_node = self._closed[next_node]
            path.append(next_node)
            path.append(self.start)
            path.reverse()
        # This intentionally matches planner.cpp: get_path stores the chosen
        # endpoint, whereas get_next_node stores the first primitive step.
        self.desired_position = endpoint
        return path

    def get_next_node(
        self,
        use_best_node: bool = True,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        next_node = self._selected_endpoint(bool(use_best_node))
        if next_node[0] < INF and next_node != self.start:
            while self._closed[next_node] != self.start:
                next_node = self._closed[next_node]
        if next_node == self.start:
            next_node = (INF, INF)
        self.desired_position = next_node
        return self.start, next_node

