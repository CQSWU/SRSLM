from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import agents.srslm_arpe_ablation as module


class _Candidate:
    device = "cpu"
    artifact = SimpleNamespace(as_dict=lambda: {"checkpoint_sha256": "a" * 64})

    def set_grid_config(self, grid_config):
        self.grid_config = grid_config

    def set_env(self, env):
        self.env = env

    def after_reset(self):
        pass

    def act(self, observations, rewards=None, dones=None, infos=None):
        del rewards, dones, infos
        return np.asarray([2, 3][: len(observations)], dtype=np.int64)

    def after_step(self, dones):
        self.dones = tuple(dones)

    def get_action_correction_stats(self):
        return {}

    def get_model_provenance(self):
        return {
            "schema": "switcher_candidate_caar_v1",
            "candidate": {"frozen": True},
            "frozen_verification": {
                "verified": True,
                "trainable_parameter_count": 0,
            },
        }


class _Planner:
    def __init__(self, max_steps, seed):
        assert max_steps == 10_000
        assert seed == 7
        self.commits = []

    def reset(self):
        pass

    def set_grid_config(self, grid_config):
        self.grid_config = grid_config

    def propose(self, observations):
        count = len(observations)
        return SimpleNamespace(
            actions=(0, 4)[:count],
            planned_mask=(True,) * count,
            reverse_mask=(False,) * count,
            static_astar_invoked_mask=(False,) * count,
        )

    def commit(self, mask):
        self.commits.append(tuple(mask))


class _SwitcherConfig:
    def copy(self, deep, update):
        assert deep is True
        return SimpleNamespace(**update)


class _AllStateSwitcher:
    def __init__(self, cfg):
        self.cfg = cfg
        self.candidate_artifact = SimpleNamespace(as_dict=lambda: {"checkpoint_sha256": "a" * 64})
        self.choices = 0
        self.ao = 0

    def after_reset(self):
        pass

    def choose(self, state):
        count = len(state["obs"])
        result = np.asarray([1, 0][:count], dtype=np.int64)
        self.choices += count
        self.ao += int((result == 1).sum())
        return result

    def get_stats(self):
        return {
            "switcher_checkpoint_sha256": "a" * 64,
            "switcher_config_sha256": "b" * 64,
            "switcher_stochastic": True,
            "switcher_model_choice_count": self.choices,
            "switcher_model_selected_ao_count": self.ao,
            "switcher_sampled_ao_rate": self.ao / self.choices,
            "switcher_ao_probability_mean": 0.5,
            "switcher_ao_probability_p05": 0.2,
            "switcher_ao_probability_p95": 0.8,
            "switcher_loader_schema": "switcher_caar_checkpoint_loader_v1",
            "switcher_candidate_policy": {},
            "switcher_candidate_artifact": {},
        }


def _config(with_switcher=True):
    values = {
        "seed": 7,
        "device": "cpu",
        "candidate": SimpleNamespace(device="cpu"),
        "max_planning_steps": 10_000,
    }
    if with_switcher:
        values["switcher"] = _SwitcherConfig()
    return SimpleNamespace(**values)


def _observations():
    return [
        {
            "obstacles": np.zeros((11, 11), dtype=np.float32),
            "agents": np.zeros((11, 11), dtype=np.float32),
            "xy": np.asarray([index, index], dtype=np.float32),
            "target_xy": np.asarray([index + 1, index + 2], dtype=np.float32),
        }
        for index in range(2)
    ]


def _grid():
    return SimpleNamespace(collision_system="block_both")


def _candidate(*args, **kwargs):
    del args, kwargs
    return _Candidate()


def test_nowait_routes_every_state_through_same_switcher_weights(monkeypatch):
    monkeypatch.setattr(module, "_frozen_candidate", _candidate)
    policy = module.SRSLMNoWait(
        _config(),
        planner_factory=_Planner,
        switcher_factory=_AllStateSwitcher,
    )
    policy.set_grid_config(_grid())
    policy.after_reset()
    assert policy.act(_observations()) == [0, 3]
    stats = policy.get_switch_stats()
    assert stats["switcher_choice_count"] == stats["total_action_count"] == 2
    assert stats["aoreplan_wait_bypass_count"] == 0


def test_onlywait_has_no_learned_switcher(monkeypatch):
    monkeypatch.setattr(module, "_frozen_candidate", _candidate)
    policy = module.SRSLMOnlyWait(
        _config(with_switcher=False),
        planner_factory=_Planner,
    )
    policy.set_grid_config(_grid())
    policy.after_reset()
    assert policy.act(_observations()) == [2, 4]
    stats = policy.get_switch_stats()
    assert stats["switcher_training"] == "none"
    assert stats["switcher_model_choice_count"] == 0
    assert stats["executed_caar_count"] == stats["aoreplan_wait_bypass_count"] == 1


def test_nowait_rejects_mismatched_candidate_before_planner_or_actions(monkeypatch):
    class WrongPin(_AllStateSwitcher):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.candidate_artifact = SimpleNamespace(as_dict=lambda: {"checkpoint_sha256": "b" * 64})

    def forbidden_planner(**kwargs):
        raise AssertionError("Mismatch must fail before planning")

    monkeypatch.setattr(module, "_frozen_candidate", _candidate)
    with pytest.raises(RuntimeError, match="differs from the candidate pinned"):
        module.SRSLMNoWait(_config(), planner_factory=forbidden_planner,
                           switcher_factory=WrongPin)


def test_nowait_rejects_missing_switcher_candidate_declaration(monkeypatch):
    class MissingPin(_AllStateSwitcher):
        def __init__(self, cfg):
            super().__init__(cfg)
            self.candidate_artifact = None

    monkeypatch.setattr(module, "_frozen_candidate", _candidate)
    with pytest.raises(RuntimeError, match="both candidate artifact declarations"):
        module.SRSLMNoWait(_config(), planner_factory=_Planner,
                           switcher_factory=MissingPin)
