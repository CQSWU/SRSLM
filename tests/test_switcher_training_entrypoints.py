"""The public registry routes Switcher training through its dedicated entry."""
from copy import deepcopy
from functools import partial
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from sample_factory.algo.utils.context import global_env_registry

import train
import train_switcher
from pomapf_env.switcher_env import SwitcherEnv
from pomapf_env.switcher_arpe_env import ArpeSwitcherEnv, ArpeNoWaitSwitcherEnv


@pytest.fixture
def isolated_registry(monkeypatch):
    registry = global_env_registry()
    original = registry.copy()
    monkeypatch.setattr(train, '_CUSTOM_COMPONENTS_REGISTERED', False)
    try:
        yield registry
    finally:
        registry.clear()
        registry.update(original)


@pytest.mark.parametrize(
    'name',
    [spec['environment'] for spec in train_switcher.MODES.values()],
)
def test_generic_switcher_factory_fails_before_configuration_or_loading(name, isolated_registry):
    train.register_custom_components()
    assert isolated_registry[name] is train.create_pogema_env
    with pytest.raises(RuntimeError, match='train_switcher.py'):
        isolated_registry[name](name, cfg=None)


def test_runtime_base_cannot_construct_legacy_arpe():
    with pytest.raises(TypeError, match='runtime base'):
        SwitcherEnv(arpe_weights_path='unused-legacy-path')


class _FrozenCandidate:
    """A lightweight candidate only; the environment and planner remain real."""
    def __init__(self, artifact, *, seed, device):
        self.artifact = artifact
        self.reset_count = 0
        self.steps = 0

    def verify_frozen(self):
        return {'verified': True}

    def get_model_provenance(self):
        return {'candidate': self.artifact.as_dict()}

    def after_reset(self):
        self.reset_count += 1

    def set_grid_config(self, cfg):
        self.grid_config = cfg

    def set_env(self, env):
        self.env = env

    def after_step(self, dones):
        self.steps += 1

    def act(self, observations, *args):
        return [4] * len(observations)


def _configuration(environment_name, collision):
    root = Path(__file__).resolve().parents[1]
    declaration = json.loads((root / 'configs/arpe_final_candidate.json').read_text())
    return SimpleNamespace(full_config={
        'candidate_policy': declaration,
        'environment': {
            'name': environment_name,
            'switcher_caar_weights_path': declaration['weights_path'],
            'switcher_caar_device': 'cpu',
            'switcher_team_reward_coefficient': 1.0,
            'grid_config': {
                'map': '.......\n.......\n.......\n.......\n.......\n.......\n.......',
                'num_agents': 2, 'seed': 0, 'obs_radius': 5,
                'on_target': 'restart', 'collision_system': collision,
                'max_episode_steps': 3,
            },
        },
    })


@pytest.mark.parametrize('mode,env_type', [
    ('final', ArpeSwitcherEnv),
    ('nowait', ArpeNoWaitSwitcherEnv),
])
@pytest.mark.parametrize('collision', ['block_both', 'soft'])
def test_dedicated_registry_constructs_real_env_and_preserves_runtime(
    mode, env_type, collision, isolated_registry, monkeypatch,
):
    # The imported entrypoint constructor still builds its real subclass;
    # only the frozen network is replaced, so this test needs no private weights.
    spec = train_switcher.MODES[mode]
    monkeypatch.setitem(
        spec,
        'environment_class',
        partial(env_type, candidate_factory=_FrozenCandidate),
    )
    train.register_custom_components()
    assert isolated_registry[spec['environment']] is train.create_pogema_env
    train_switcher.register_switcher_components(mode)
    factory = isolated_registry[spec['environment']]
    assert factory is not train.create_pogema_env
    cfg = _configuration(spec['environment'], collision)
    with pytest.raises(ValueError, match='cannot construct'):
        factory('POMAPF-v0', cfg=cfg)
    missing = SimpleNamespace(full_config=deepcopy(cfg.full_config))
    missing.full_config.pop('candidate_policy')
    with pytest.raises(RuntimeError, match='no candidate_policy paths'):
        factory(spec['environment'], cfg=missing)

    env = factory(spec['environment'], cfg=cfg, env_config={'worker_index': 0})
    assert type(env) is env_type
    assert env.candidate_artifact.as_dict() == env.candidate.artifact.as_dict()
    rewards_seen = []
    base_step = env.base_env.step

    def record_base_step(actions):
        result = base_step(actions)
        rewards_seen.append(np.asarray(result[1], dtype=np.float32))
        return result

    monkeypatch.setattr(env.base_env, 'step', record_base_step)
    try:
        with pytest.raises(RuntimeError, match='reset'):
            env.step([0, 1])
        observations, infos = env.reset(seed=0)
        assert len(observations) == len(infos) == 2
        for index in range(8):
            for observation in observations:
                assert env.observation_space.contains(observation)
            allowed = tuple(env._prepared.switch_allowed_mask)
            expected = ((True, True) if env_type is ArpeNoWaitSwitcherEnv else
                        tuple(a != 0 for a in env._prepared.aoreplan_actions))
            assert allowed == expected
            observations, rewards, terminated, truncated, infos = env.step([index % 2, (index + 1) % 2])
            raw = rewards_seen[-1]
            np.testing.assert_allclose(rewards, raw + raw.mean(), rtol=0, atol=0)
            assert len(observations) == len(rewards) == len(terminated) == len(truncated) == len(infos) == 2
            assert np.all(np.isfinite(rewards))
        # The 3-step environment completed and auto-reset more than once.
        assert env.candidate.steps == 8
        assert env.candidate.reset_count >= 4
        assert env.get_candidate_provenance() == env.candidate.get_model_provenance()
    finally:
        env.close()
