"""Retired configuration values must fail rather than select another model."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from learning.config import Environment, Experiment, OBSOLETE_SAVED_SETTINGS, checkpoint_experiment_config
from learning.encoder import make_actor_critic, make_encoder
from train import create_pogema_env

UNUSED_CONTEXT_FIELDS = {
    'trace_context_filters': 32,
    'trace_context_embedding_size': 128,
    'trace_context_hidden_projection': 128,
    'trace_context_fusion_size': 256,
    'trace_context_head_size': 128,
    'trace_context_residual_cap': 2.0,
}

RETIRED_TAU_FIELDS = {
    'caar_tau_num_filters': 8,
    'caar_tau_num_conv_layers': 1,
    'caar_tau_num_res_blocks': 0,
    'caar_tau_hidden_size': 0,
    'caar_learn_residual': True,
    'caar_contextual_pressure': False,
    'caar_pressure_head_mode': 'legacy_multiplier',
    'caar_pressure_output_transform': 'clipped_relu',
    'caar_pressure_cap': 2.0,
    'caar_pressure_init': 0.1,
    'caar_reweight_wait_action': None,
    'trace_correction_mode': 'normalized_linear',
}


def _recipe():
    path = Path(__file__).resolve().parents[1] / 'learning/train_epom_trace_paper_conv_fusion_r5_500m.yaml'
    return yaml.safe_load(path.read_text())


@pytest.mark.parametrize('kind', ['epom_trace', 'nonexistent_trace'])
def test_retired_or_unknown_encoder_is_rejected_by_schema_and_factories(kind):
    raw = _recipe()
    raw['experiment_settings']['encoder_custom'] = kind
    with pytest.raises(ValidationError, match='encoder_custom'):
        Experiment(**raw)
    cfg = SimpleNamespace(encoder_custom=kind)
    with pytest.raises(ValueError, match='Unsupported or retired'):
        make_encoder(cfg, None)
    with pytest.raises(ValueError, match='Unsupported or retired'):
        make_actor_critic(cfg, None, None)


@pytest.mark.parametrize('field,value', (UNUSED_CONTEXT_FIELDS | RETIRED_TAU_FIELDS).items())
def test_new_recipe_rejects_unused_dimensions_and_cap(field, value):
    raw = _recipe()
    raw['experiment_settings'][field] = value
    with pytest.raises(ValidationError, match=field):
        Experiment(**raw)


def test_checkpoint_only_migration_preserves_input_and_current_model_recipe():
    raw = _recipe()
    expected = Experiment(**raw).dict()
    removed = UNUSED_CONTEXT_FIELDS | RETIRED_TAU_FIELDS
    raw['experiment_settings'].update(removed)
    original = deepcopy(raw)
    normalized = checkpoint_experiment_config(raw)
    assert raw == original
    assert set(removed) <= OBSOLETE_SAVED_SETTINGS
    assert set(removed).isdisjoint(normalized['experiment_settings'])
    assert Experiment(**normalized).dict() == expected


@pytest.mark.parametrize('factory', [make_encoder, make_actor_critic])
def test_retired_tau_actor_cannot_fall_back_to_an_uncorrected_policy(factory):
    cfg = SimpleNamespace(encoder_custom='caar')
    obs = gym.spaces.Dict({'tau': gym.spaces.Box(-1, 1, (1, 11, 11), np.float32)})
    with patch('learning.encoder.default_make_actor_critic_func') as default:
        with pytest.raises(ValueError, match=r'caar\+tau actor is retired'):
            if factory is make_encoder:
                factory(cfg, obs)
            else:
                factory(cfg, obs, gym.spaces.Discrete(5))
        default.assert_not_called()


def test_noreweight_without_tau_retains_its_default_actor():
    cfg = SimpleNamespace(encoder_custom='caar')
    obs = gym.spaces.Dict({'obs': gym.spaces.Box(0, 1, (3, 11, 11), np.float32)})
    action_space = gym.spaces.Discrete(5)
    expected = object()
    with patch('learning.encoder.default_make_actor_critic_func', return_value=expected) as default:
        assert make_actor_critic(cfg, obs, action_space) is expected
    default.assert_called_once_with(cfg, obs, action_space)


def test_retired_tau_environment_fails_at_both_entrypoints():
    with pytest.raises(ValidationError, match='name'):
        Environment(name='POMAPF-ST-v0')
    with pytest.raises(RuntimeError, match='POMAPF-ST-v0 is retired'):
        create_pogema_env('POMAPF-ST-v0')


def test_noentropy_training_recipe_changes_only_gate_and_output_identity():
    raw = _recipe()
    root = Path(__file__).resolve().parents[1]
    ungated = yaml.safe_load((root / 'learning/train_arpe_noentropy_r5_500m.yaml').read_text())
    assert ungated['name'] == 'ARPE-NoEntropy-FromScratch-R5-500M'
    assert ungated['global_settings']['train_dir'] == 'weights/ARPE-noentropy-retrain'
    assert ungated['experiment_settings']['trace_context_learned_gate'] == 'all'
    Experiment(**ungated)
    ungated['name'] = raw['name']
    ungated['global_settings']['train_dir'] = raw['global_settings']['train_dir']
    ungated['experiment_settings']['trace_context_learned_gate'] = raw['experiment_settings']['trace_context_learned_gate']
    assert ungated == raw
