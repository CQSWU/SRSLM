"""Shared runtime methods for the hash-pinned Switcher environments."""

from __future__ import annotations

import gymnasium as gym
import numpy as np

class SwitcherEnv(gym.Env):
    """Runtime base; construct ArpeSwitcherEnv or ArpeNoWaitSwitcherEnv."""

    metadata = {"render_modes": []}

    def __init__(self, *args, **kwargs):
        raise TypeError(
            'SwitcherEnv is a runtime base, not a standalone environment. '
            'Use ArpeSwitcherEnv or ArpeNoWaitSwitcherEnv with a pinned candidate.'
        )

    @property
    def grid_config(self):
        return self.base_env.grid_config

    @property
    def grid(self):
        return self.base_env.grid

    def get_num_agents(self):
        return self.num_agents

    @staticmethod
    def _feature_observations(features):
        count = int(np.asarray(features["obs"]).shape[0])
        return [
            {
                key: np.asarray(value[index], dtype=np.float32).copy()
                for key, value in features.items()
            }
            for index in range(count)
        ]

    def _start_policy_episode(self, observations, infos):
        self.controller.after_reset()
        self.controller.set_grid_config(self.base_env.grid_config)
        self.controller.set_env(self.base_env)
        count = len(observations)
        self._last_rewards = np.zeros(count, dtype=np.float32)
        self._last_dones = np.zeros(count, dtype=np.bool_)
        self._last_infos = infos
        self._prepared = self.controller.prepare_actions(
            observations,
            self._last_rewards,
            self._last_dones,
            infos,
        )
        return self._feature_observations(self._prepared.switcher_state)

    def reset(self, *, seed=None, options=None):
        del options
        if seed is not None:
            # POGEMA uses the seed stored in its grid config.  Sample Factory
            # normally leaves this unset, but accepting it keeps Gymnasium's
            # reset contract intact.
            self.base_env.grid_config.seed = int(seed)
        observations, infos = self.base_env.reset()
        return self._start_policy_episode(observations, infos), infos

    def _append_episode_stats(self, infos) -> None:
        stats = self.controller.get_stats()
        episode_stats = {
            "switch_selected_ao_rate": stats["selected_ao_rate"],
            "switch_executed_ao_rate": stats["executed_ao_rate"],
            "switch_wait_bypass_rate": stats["aoreplan_wait_bypass_rate"],
            "branch_action_agreement_rate": stats[
                "branch_action_agreement_rate"
            ],
        }
        for info in infos:
            target = info.setdefault("episode_extra_stats", {})
            target.update(episode_stats)

    def step(self, selector_actions):
        if self._prepared is None:
            raise RuntimeError("reset() must be called before step().")
        choices = np.asarray(selector_actions, dtype=np.int64).reshape(-1)
        if choices.shape != (self.num_agents,):
            raise ValueError("Switcher returned the wrong number of training actions.")
        switch_allowed = np.asarray(
            self._prepared.switch_allowed_mask,
            dtype=bool,
        )
        decision = self.controller.resolve_actions(choices[switch_allowed])
        (
            observations,
            rewards,
            terminated,
            truncated,
            infos,
        ) = self.base_env.step(list(decision.actions))
        done = np.logical_or(
            np.asarray(terminated, dtype=np.bool_),
            np.asarray(truncated, dtype=np.bool_),
        )
        self.controller.after_step(done)

        rewards_array = np.asarray(rewards, dtype=np.float32)
        team_mean = float(rewards_array.mean())
        training_rewards = rewards_array + (
            self.team_reward_coefficient * team_mean
        )

        episode_finished = bool(done.size and np.all(done))
        if episode_finished:
            self._append_episode_stats(infos)
            # make_pomapf includes AutoResetWrapper, so observations already
            # belong to the next episode here.
            next_features = self._start_policy_episode(observations, infos)
        else:
            self._last_rewards = rewards_array
            self._last_dones = done
            self._last_infos = infos
            self._prepared = self.controller.prepare_actions(
                observations,
                rewards_array,
                done,
                infos,
            )
            next_features = self._feature_observations(
                self._prepared.switcher_state
            )

        return (
            next_features,
            training_rewards.tolist(),
            list(terminated),
            list(truncated),
            infos,
        )

    def close(self):
        self._prepared = None
        return self.base_env.close()


__all__ = ["SwitcherEnv"]
