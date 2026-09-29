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
from learning.config import Experiment
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
