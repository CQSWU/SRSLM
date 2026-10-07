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
        self._best_node = (INF, INF, 0)

    def _has_desired_position(self) -> bool:
        return self.desired_position[0] < INF

    def _consume_execution_feedback(
        self,
        position,
        cache_failed_action: bool = True,
    ) -> None:
        position = _pair(position)
        if self.desired_position != position:
            if cache_failed_action:
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

        heapq.heappush(
            self._open,
            (start_h, 0, self.start[0], self.start[1]),
        )
        self._closed[self.start] = self.start
        self._best_node = (
            self.start[0],
            self.start[1],
            start_h,
        )

    def _compute_shortest_path(self) -> None:
        current = (INF, INF)
        steps = 0
        while self._open and steps < self.max_steps and current != self.goal:
            _f, g, i, j = heapq.heappop(self._open)
            current = i, j
            current_h = self._heuristic(current)
            if current_h < self._best_node[2]:
                self._best_node = (i, j, current_h)
            steps += 1

            for neighbor in self._neighbors(current):
                if neighbor in self._closed or neighbor in self.other_agents:
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

                self._closed[neighbor] = current

    def update_obstacles(
        self,
        obstacles: Iterable[Sequence[int]],
        other_agents: Iterable[Sequence[int]],
        cur_pos: Sequence[int],
    ) -> None:
        cur_pos = _pair(cur_pos)

        for obstacle in obstacles:
            oi, oj = _pair(obstacle)
            self.obstacles.add((cur_pos[0] + oi, cur_pos[1] + oj))

        self.other_agents.clear()
        for agent in other_agents:
            ai, aj = _pair(agent)
            self.other_agents.add((cur_pos[0] + ai, cur_pos[1] + aj))

    def proposal_failed(self, position: Sequence[int]) -> bool:
        position = _pair(position)
        return self._has_desired_position() and self.desired_position != position

    def observe_position(
        self,
        position: Sequence[int],
        cache_failed_action: bool = True,
    ) -> None:
        if self._has_desired_position():
            self._consume_execution_feedback(position, cache_failed_action)

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

    def release_failed_actions(self) -> None:
        self.bad_actions.clear()

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

    def get_next_node(
        self,
        use_best_node: bool = True,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        if self.goal in self._closed:
            next_node = self.goal
        elif use_best_node:
            next_node = self._best_node[0], self._best_node[1]
        else:
            next_node = (INF, INF)
        if next_node[0] < INF and next_node != self.start:
            while self._closed[next_node] != self.start:
                next_node = self._closed[next_node]
        if next_node == self.start:
            next_node = (INF, INF)
        self.desired_position = next_node
        return self.start, next_node
