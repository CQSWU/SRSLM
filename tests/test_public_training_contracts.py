"""Retired configuration values must fail rather than select another model."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

from learning.config import Experiment, OBSOLETE_SAVED_SETTINGS, checkpoint_experiment_config
from learning.encoder import make_actor_critic, make_encoder

UNUSED_CONTEXT_FIELDS = {
    'trace_context_filters': 32,
    'trace_context_embedding_size': 128,
    'trace_context_hidden_projection': 128,
    'trace_context_fusion_size': 256,
    'trace_context_head_size': 128,
    'trace_context_residual_cap': 2.0,
}


def _recipe():
    path = Path(__file__).resolve().parents[1] / 'learning/train_arpe.yaml'
    return yaml.safe_load(path.read_text())


def test_zero_trace_control_is_capacity_and_budget_matched():
    root = Path(__file__).resolve().parents[1]
    real = _recipe()
    zero = yaml.safe_load(
        (root / 'learning/train_arpe_zero_trace.yaml').read_text()
    )
    assert zero['environment']['trace_variant'] == 'zero'

    zero['name'] = real['name']
    zero['global_settings']['train_dir'] = real['global_settings']['train_dir']
    zero['environment']['trace_variant'] = 'real'
    real['environment']['trace_variant'] = 'real'
    assert zero == real


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


@pytest.mark.parametrize('field,value', UNUSED_CONTEXT_FIELDS.items())
def test_new_recipe_rejects_unused_dimensions_and_cap(field, value):
    raw = _recipe()
    raw['experiment_settings'][field] = value
    with pytest.raises(ValidationError, match=field):
        Experiment(**raw)


def test_checkpoint_only_migration_preserves_input_and_current_model_recipe():
    raw = _recipe()
    expected = Experiment(**raw).dict()
    raw['experiment_settings'].update(UNUSED_CONTEXT_FIELDS)
    original = deepcopy(raw)
    normalized = checkpoint_experiment_config(raw)
    assert raw == original
    assert set(UNUSED_CONTEXT_FIELDS) <= OBSOLETE_SAVED_SETTINGS
    assert set(UNUSED_CONTEXT_FIELDS).isdisjoint(normalized['experiment_settings'])
    assert Experiment(**normalized).dict() == expected


def test_retired_noentropy_and_noreweight_recipes_are_absent():
    root = Path(__file__).resolve().parents[1]
    for relative in (
        'learning/train_arpe_noentropy_r5_500m.yaml',
        'learning/train_no_reweight_block_r5_1b.yaml',
        'learning/no_reweight_encoder.py',
        'configs/arpe_noentropy_candidate.json',
    ):
        assert not (root / relative).exists()
