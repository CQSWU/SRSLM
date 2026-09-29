"""Public runs accept custom recipes without paper-audit registration."""
from copy import deepcopy
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import yaml

from agents.arpe import ARPE
from agents.epom import _validate_lifelong_finetuned_config
from learning.config import Environment, Experiment
from agents.switcher_core import switcher_observation_space
import run_experiments as runner

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('recipe', ['train_epom.yaml', 'train_arpe_final.yaml'])
def test_training_recipe_can_use_custom_maps_populations_and_hardware(recipe):
    config = yaml.safe_load((ROOT / 'learning' / recipe).read_text())
    config['environment']['grid_config'].update(
        num_agents=16, max_episode_steps=128, on_target='finish',
        collision_system='priority', map_name='maps/my_maps.yaml',
    )
    config['async_ppo'].update(num_workers=32, batch_size=8192,
                               rollout=64, recurrence=64, num_envs_per_worker=4,
                               num_epochs=2, ppo_clip_ratio=0.2)
    config['experiment_settings'].update(learning_rate=0.0003, gamma=0.97)
    if recipe == 'train_arpe_final.yaml':
        config['environment']['training_num_agents_by_worker'] = [16, 32]
        config['experiment_settings'].update(trace_rule_scale=2.5, trace_gate_threshold=0.1)
    saved = deepcopy(config)
    experiment = Experiment(**config)
    assert experiment.async_ppo.num_workers == 32
    assert experiment.environment.grid_config.num_agents == 16
    assert config == saved


def test_epom_loader_does_not_bind_a_compatible_model_to_its_training_protocol():
    config = yaml.safe_load((ROOT / 'learning/train_epom.yaml').read_text())
    config['environment']['grid_config'].update(
        num_agents=16, collision_system='soft', on_target='finish', max_episode_steps=128)
    _validate_lifelong_finetuned_config(config)
    config['environment']['grid_config']['MOVES'] = [[0, 0], [1, 0], [-1, 0], [0, -1], [0, 1]]
    with pytest.raises(ValueError, match='action order'):
        _validate_lifelong_finetuned_config(config)


def test_arpe_freezes_parameters_without_a_file_hash_audit():
    model = torch.nn.Linear(2, 5)
    artifact = SimpleNamespace(inference=None, inspect_files=Mock(side_effect=AssertionError('audit called')))
    adapter = ARPE(SimpleNamespace(ppo=model, device='cpu'), artifact)
    assert adapter.ppo is model
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    artifact.inspect_files.assert_not_called()


def test_runner_does_not_require_private_source_or_result_audits():
    main_source = inspect.getsource(runner.main)
    episode_source = inspect.getsource(runner.run_single_experiment)
    assert 'srslm_integrity_metadata' not in main_source
    assert 'epom_lifelong_result_manifest' not in main_source
    assert 'validate_srslm_stats' not in episode_source
    assert 'validate_final_srslm_ablation_stats' not in episode_source


@pytest.mark.parametrize('context,expected', [
    (None, 16), ({}, 16), ({'worker_index': None}, 16),
    ({'worker_index': 3}, 32), (SimpleNamespace(worker_index=2), 16),
])
def test_worker_population_assignment_keeps_recipe_unchanged(context, expected):
    environment = Environment(training_num_agents_by_worker=[16, 32])
    before = environment.dict()
    worker = environment.for_worker(context)
    assert worker.grid_config.num_agents == expected
    worker.grid_config.seed = 99
    assert environment.dict() == before


@pytest.mark.parametrize('index', [-1, 'invalid'])
def test_worker_population_rejects_invalid_index(index):
    environment = Environment(training_num_agents_by_worker=[16, 32])
    with pytest.raises(ValueError, match='worker index'):
        environment.for_worker({'worker_index': index})


def test_worker_without_population_schedule_keeps_default():
    environment = Environment()
    assert environment.for_worker({'worker_index': 42}) is environment


def test_switcher_observation_space_is_independent_and_matches_saved_shapes():
    import numpy as np
    expected = {'obs': (3, 11, 11), 'xy': (2,), 'target_xy': (2,),
                'caar_action': (5,), 'aoreplan_action': (5,)}
    training = switcher_observation_space()
    inference = switcher_observation_space()
    assert set(training.spaces) == set(expected)
    for key, shape in expected.items():
        assert training[key].shape == shape
        assert training[key].dtype == np.float32
        low, high = (-1024, 1024) if key in {'xy', 'target_xy'} else (0, 1)
        assert np.all(training[key].low == low)
        assert np.all(training[key].high == high)
        assert inference[key] == training[key]
    inference['extra'] = inference['xy']
    assert 'extra' not in training.spaces
