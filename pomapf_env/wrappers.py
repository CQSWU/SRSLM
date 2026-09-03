import re

from copy import deepcopy

from pathlib import Path


import numpy as np
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from numpy import float32
from pogema import GridConfig

from pomapf_env.custom_maps import MAPS_REGISTRY
from pomapf_env.stigmergic import AcoState
from learning.grid_memory import MultipleGridMemory


class RewardShaping(gym.Wrapper):

    def __init__(self, env):

        super().__init__(env)
        self._previous_xy = None

    def step(self, action):

        observations, rewards, terminated, truncated, infos = self.env.step(action)
        for agent_idx in range(self.env.unwrapped.grid_config.num_agents):
            reward = rewards[agent_idx]
            reward -= 0.0001
            if action[agent_idx] != 0:
                if tuple(self._previous_xy[agent_idx]) == tuple(observations[agent_idx]['xy']):
                    reward -= 0.0002
            rewards[agent_idx] = reward
            self._previous_xy[agent_idx] = observations[agent_idx]['xy']


        return observations, rewards, terminated, truncated, infos

    def reset(self, **kwargs):

        observation, info = self.env.reset(**kwargs)
        self._previous_xy = [
            np.asarray(agent_observation["xy"]).copy()
            for agent_observation in observation
        ]

        return observation, info


class EnvAttributesWrapper(gym.Wrapper):

    """Expose Pogema metadata through Gymnasium wrapper stacks."""


    @property

    def grid_config(self):

        return self.env.unwrapped.grid_config


    @property

    def num_agents(self):

        return self.grid_config.num_agents

    def get_num_agents(self):

        return self.num_agents


    @property

    def is_multiagent(self):

        return True


    @property

    def grid(self):

        return self._find_attr_in_chain('grid')

    def set_elapsed_steps(self, steps):

        return self._find_attr_in_chain('set_elapsed_steps')(steps)

    def _find_attr_in_chain(self, name):

        current = self.env
        while hasattr(current, 'env'):
            if hasattr(current, name):
                return getattr(current, name)
            current = current.env

        return getattr(current, name)

    def __getattr__(self, name):

        if name in ('env', '_env', '_previous_xy', '_configs', '_rnd'):

            raise AttributeError(name)

        return self._find_attr_in_chain(name)


class MultiMapWrapper(gym.Wrapper):

    @property

    def grid_config(self):

        return self.env.unwrapped.grid_config

    def __init__(self, env):

        super().__init__(env)
        self._configs = []
        self._rnd = np.random.default_rng(self.grid_config.seed)
        pattern = self.grid_config.map_name


        if pattern:

            map_path = Path(pattern)

            if map_path.exists() and map_path.suffix.lower() in ('.yaml', '.yml'):

                import yaml

                file_maps = yaml.safe_load(
                    map_path.read_text(encoding="utf-8")
                )

                if not isinstance(file_maps, dict) or not file_maps:

                    raise ValueError(
                        f"Map file must contain a non-empty YAML mapping: "
                        f"{map_path}"
                    )

                candidates = file_maps.items()

            elif pattern in MAPS_REGISTRY:

                candidates = ((pattern, MAPS_REGISTRY[pattern]),)

            else:

                try:

                    matcher = re.compile(pattern)

                except re.error as exc:

                    raise ValueError(
                        f"Invalid map-name regular expression {pattern!r}: "
                        f"{exc}"
                    ) from exc

                candidates = (
                    (map_name, map_value)
                    for map_name, map_value in MAPS_REGISTRY.items()
                    if matcher.match(map_name)
                )

            for map_name, map_value in candidates:

                cfg = deepcopy(self.grid_config)
                cfg.map = map_value
                cfg.map_name = map_name
                cfg = GridConfig(**cfg.dict())
                self._configs.append(cfg)

            if not self._configs:

                raise KeyError(f"No map matching: {pattern}")

    def step(self, action):

        observations, rewards, terminated, truncated, info = self.env.step(action)
        cfg = self.grid_config
        if cfg.map_name:
            for agent_idx in range(cfg.num_agents):
                if 'episode_extra_stats' in info[agent_idx]:
                    for key, value in list(info[agent_idx]['episode_extra_stats'].items()):
                        if key == 'Done':
                            continue
                        info[agent_idx]['episode_extra_stats'][f'{key}-{cfg.map_name.split("-")[0]}'] = value
        return observations, rewards, terminated, truncated, info


    def reset(self, **kwargs):

        if self._configs is not None and len(self._configs) >= 1:
            cfg = deepcopy(self._configs[self._rnd.integers(0, len(self._configs))])
            self.env.unwrapped.grid_config = cfg

        return self.env.reset(**kwargs)


class MatrixObservationWrapper(ObservationWrapper):


    def __init__(self, env):

        super().__init__(env)
        full_size = self.env.observation_space['obstacles'].shape[0]

        self.observation_space = gym.spaces.Dict(
            obs=gym.spaces.Box(0.0, 1.0, shape=(3, full_size, full_size)),
            xy=Box(low=-1024, high=1024, shape=(2,), dtype=np.float32),
            target_xy=Box(
                low=-1024,
                high=1024,
                shape=(2,),
                dtype=np.float32,
            ),
        )

        self.num_agents = self.env.num_agents
        self.is_multiagent = self.env.is_multiagent


    @staticmethod

    def get_square_target(x, y, tx, ty, obs_radius):

        full_size = obs_radius * 2 + 1
        result = np.zeros((full_size, full_size), dtype=np.float32)
        dx = int(round(float(x) - float(tx)))
        dy = int(round(float(y) - float(ty)))

        dx = min(dx, obs_radius) if dx >= 0 else max(dx, -obs_radius)
        dy = min(dy, obs_radius) if dy >= 0 else max(dy, -obs_radius)
        result[obs_radius - dx, obs_radius - dy] = 1
        return result


    @staticmethod

    def to_matrix(observations):

        result = []
        obs_radius = observations[0]['obstacles'].shape[0] // 2

        for obs in observations:
            result.append(
                {"obs": np.concatenate([obs['obstacles'][None], obs['agents'][None],
                                        MatrixObservationWrapper.get_square_target(*obs['xy'], *obs['target_xy'],
                                                                                   obs_radius)[None]]).astype(float32),
                 "xy": np.array(obs['xy'], dtype=float32),
                 "target_xy": np.array(obs['target_xy'], dtype=float32),
                 })
        return result


    def observation(self, observation):

        result = self.to_matrix(observation)
        return result


class GridMemoryObservationWrapper(gym.Wrapper):
    """Expand raw POGEMA observations with EPOM's per-agent grid memory.

    The official EPOM policy observes an 11x11 live crop but feeds a 15x15
    remembered obstacle crop to its encoder.  This stateful wrapper mirrors the
    inference-time ``MultipleGridMemory`` path during Sample Factory rollouts.
    """

    def __init__(self, env, memory_radius=7):
        super().__init__(env)
        self.memory_radius = int(memory_radius)
        if self.memory_radius < 1:
            raise ValueError("Grid-memory radius must be at least 1.")

        self.grid_memory = MultipleGridMemory()
        size = self.memory_radius * 2 + 1
        spaces = dict(self.env.observation_space.spaces)
        obstacle_space = spaces["obstacles"]
        agent_space = spaces["agents"]
        spaces["obstacles"] = Box(
            low=0,
            high=1,
            shape=(size, size),
            dtype=obstacle_space.dtype,
        )
        spaces["agents"] = Box(
            low=0,
            high=1,
            shape=(size, size),
            dtype=agent_space.dtype,
        )
        self.observation_space = gym.spaces.Dict(spaces)
        self.num_agents = self.env.num_agents
        self.is_multiagent = self.env.is_multiagent

    @staticmethod
    def _episode_finished(terminated, truncated):
        done = np.logical_or(
            np.asarray(terminated, dtype=bool),
            np.asarray(truncated, dtype=bool),
        )
        return bool(done.size and np.all(done))

    def _augment(self, observations):
        observations = deepcopy(observations)
        self.grid_memory.update(observations)
        self.grid_memory.modify_observation(
            observations,
            self.memory_radius,
        )
        return observations

    def reset(self, **kwargs):
        observations, info = self.env.reset(**kwargs)
        self.grid_memory.clear()
        return self._augment(observations), info

    def step(self, actions):
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        if self._episode_finished(terminated, truncated):
            self.grid_memory.clear()
        return (
            self._augment(observations),
            rewards,
            terminated,
            truncated,
            infos,
        )


class TauObservationWrapper(gym.Wrapper):
    """Add a signed local mean-centered traffic observation."""

    def __init__(self, env, rho=0.1, tau_radius=None, trace_variant="real",
                 raw_tau=False, variant_seed=None, include_free_mask=False):
        super().__init__(env)
        from pomapf_env.trace_variant import TraceVariant

        self.variant = TraceVariant(trace_variant, seed=variant_seed)
        self.raw_tau = bool(raw_tau)
        self.include_free_mask = bool(include_free_mask)
        obs_space = self.env.observation_space
        channels, height, width = obs_space["obs"].shape
        if channels != 3:
            raise ValueError(f"TauObservationWrapper expects 3 context channels, got {channels}.")
        if height != width or height % 2 != 1:
            raise ValueError(
                "TauObservationWrapper requires a square odd-sized context observation."
            )

        self.aco = AcoState(rho=rho)
        context_radius = height // 2
        self.tau_radius = (
            context_radius
            if tau_radius is None
            else int(tau_radius)
        )
        if self.tau_radius < 1:
            raise ValueError("tau_radius must be at least 1.")
        tau_size = 2 * self.tau_radius + 1
        pressure_bound = 1.0 / self.aco.rho
        spaces = dict(obs_space.spaces)
        spaces["tau"] = Box(
            low=-pressure_bound,
            high=pressure_bound,
            shape=(1, tau_size, tau_size),
            dtype=np.float32,
        )
        if self.include_free_mask:
            spaces["tau_free_mask"] = Box(
                low=0.0,
                high=1.0,
                shape=(1, tau_size, tau_size),
                dtype=np.float32,
            )
        self.observation_space = gym.spaces.Dict(spaces)
        self.num_agents = self.env.num_agents
        self.is_multiagent = self.env.is_multiagent

    def reset(self, **kwargs):
        observations, info = self.env.reset(**kwargs)
        self._configure_trace(clear=True)
        self._observe(observations, reset=True)
        return observations, info

    def step(self, action):
        observations, rewards, terminated, truncated, infos = self.env.step(action)
        self._observe(observations, reset=False)
        return observations, rewards, terminated, truncated, infos

    def _observe(self, observations, reset):
        """Deposit at the true positions, then read at the variant's positions.

        The trace itself is always written from the real occupancy, so all
        variants share an identical global trace; only where each agent reads
        it differs.
        """
        true_positions = self._global_positions()
        if reset:
            self.aco.reset_episode(
                observations,
                positions=true_positions,
                raw_tau=self.raw_tau,
                radius=self.tau_radius,
            )
        else:
            self.aco.observe_for_inference(
                observations,
                positions=true_positions,
                raw_tau=self.raw_tau,
                radius=self.tau_radius,
            )
        read_positions = true_positions
        if self.variant.wants_alternate_positions():
            read_positions = self.variant.alternate_positions(true_positions)
            self.aco.add_tau_observation(
                observations,
                positions=read_positions,
                raw_tau=self.raw_tau,
                radius=self.tau_radius,
            )
        self.variant.apply(observations)
        if self.include_free_mask:
            for observation, (x, y) in zip(observations, read_positions):
                free_mask = self.aco.extract_local_free_mask(
                    int(x),
                    int(y),
                    self.tau_radius,
                )
                observation["tau_free_mask"] = free_mask[
                    np.newaxis, ...
                ].astype(np.float32, copy=False)

    def _configure_trace(self, clear):
        obstacles = np.asarray(self._grid().obstacles, dtype=bool)
        self.aco.configure_from_obstacle_mask(obstacles, clear=clear)

    def _global_positions(self):
        grid = self._grid()
        positions = getattr(grid, "positions_xy", None)
        if positions is None and hasattr(grid, "get_agents_xy"):
            positions = grid.get_agents_xy()
        if positions is None:
            raise RuntimeError("Tau observation requires global agent positions from Pogema.")
        return np.asarray(positions, dtype=np.int64)

    def _grid(self):
        current = self.env
        while current is not None:
            if isinstance(current, EnvAttributesWrapper):
                return current.grid
            current = getattr(current, "env", None)
        raise RuntimeError("Tau observation could not locate the Pogema grid.")


class TraceContextTeamRewardWrapper(gym.Wrapper):
    """Add a cooperative team signal without recursively modifying its mean.

    For a coefficient ``c``, every agent receives its own inner-environment
    reward plus ``c`` times the mean of the unmodified reward vector returned
    by that inner environment for the current step.
    """

    def __init__(self, env, coefficient=1.0):
        super().__init__(env)
        self.coefficient = float(coefficient)
        if not np.isfinite(self.coefficient):
            raise ValueError("Team reward coefficient must be finite.")
        self.num_agents = self.env.num_agents
        self.is_multiagent = self.env.is_multiagent

    def step(self, actions):
        observations, rewards, terminated, truncated, infos = self.env.step(
            actions
        )
        if self.coefficient == 0.0:
            return observations, rewards, terminated, truncated, infos

        original = np.asarray(rewards, dtype=np.float32)
        if original.size == 0:
            return observations, rewards, terminated, truncated, infos
        increment = self.coefficient * float(original.mean())

        if isinstance(rewards, np.ndarray):
            adjusted = rewards + increment
        elif isinstance(rewards, tuple):
            adjusted = tuple(float(value) + increment for value in rewards)
        else:
            adjusted = [float(value) + increment for value in rewards]
        return observations, adjusted, terminated, truncated, infos
