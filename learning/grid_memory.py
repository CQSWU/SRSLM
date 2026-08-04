import numpy as np


class GridMemory:
    def __init__(self, start_radius=32):
        size = start_radius * 2 + 1
        self.memory = np.zeros((size, size), dtype=np.float32)

    @staticmethod
    def _insert(x, y, source, target):
        radius = source.shape[0] // 2
        try:
            target[
                x - radius:x + radius + 1,
                y - radius:y + radius + 1,
            ] = source
            return True
        except ValueError:
            return False

    def _grow(self):
        old = self.memory
        old_size = old.shape[0]
        self.memory = np.zeros(
            (old_size * 2 + 1, old_size * 2 + 1),
            dtype=np.float32,
        )
        if not self._insert(old_size, old_size, old, self.memory):
            raise RuntimeError("Failed to grow grid memory.")

    def update(self, x, y, obstacles):
        obstacles = np.asarray(obstacles)
        if (
            obstacles.ndim != 2
            or obstacles.shape[0] != obstacles.shape[1]
            or obstacles.shape[0] == 0
            or obstacles.shape[0] % 2 == 0
        ):
            raise ValueError(
                "Grid-memory obstacles must be a non-empty, odd-sized "
                f"square 2-D array; received shape {obstacles.shape}."
            )
        while True:
            radius = self.memory.shape[0] // 2
            if self._insert(
                radius + int(x),
                radius + int(y),
                obstacles,
                self.memory,
            ):
                return
            self._grow()

    def observation(self, x, y, obs_radius):
        while True:
            radius = self.memory.shape[0] // 2
            tx = int(x) + radius
            ty = int(y) + radius
            size = self.memory.shape[0]
            if (
                0 <= tx - obs_radius
                and tx + obs_radius + 1 <= size
                and 0 <= ty - obs_radius
                and ty + obs_radius + 1 <= size
            ):
                return self.memory[
                    tx - obs_radius:tx + obs_radius + 1,
                    ty - obs_radius:ty + obs_radius + 1,
                ]
            self._grow()


class MultipleGridMemory:
    def __init__(self):
        self.memories = None

    def clear(self):
        self.memories = None

    def update(self, observations):
        if self.memories is None or len(self.memories) != len(observations):
            self.memories = [GridMemory() for _ in observations]
        for memory, observation in zip(self.memories, observations):
            memory.update(
                *observation["xy"],
                observation["obstacles"],
            )

    def modify_observation(self, observations, obs_radius):
        for memory, observation in zip(self.memories, observations):
            observation["obstacles"] = memory.observation(
                *observation["xy"],
                obs_radius,
            )

        source_radius = observations[0]["agents"].shape[0] // 2
        for observation in observations:
            if source_radius <= obs_radius:
                agents = np.zeros(
                    (obs_radius * 2 + 1, obs_radius * 2 + 1),
                    dtype=observation["agents"].dtype,
                )
                offset = obs_radius - source_radius
                agents[
                    offset:offset + source_radius * 2 + 1,
                    offset:offset + source_radius * 2 + 1,
                ] = observation["agents"]
                observation["agents"] = agents
            else:
                offset = source_radius - obs_radius
                observation["agents"] = observation["agents"][
                    offset:offset + obs_radius * 2 + 1,
                    offset:offset + obs_radius * 2 + 1,
                ]
