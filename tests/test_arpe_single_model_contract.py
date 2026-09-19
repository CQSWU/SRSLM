"""The public recipe only exposes the retained ARPE model and zero-trace control."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from learning.config import Experiment, checkpoint_experiment_config


def _recipe():
    root = Path(__file__).resolve().parents[1]
    return yaml.safe_load((root / "learning/train_arpe.yaml").read_text())


@pytest.mark.parametrize("field,value", [
    ("trace_context_architecture", "context"),
    ("trace_context_architecture", "paper_entropy_conv_direct_correction_centered_P_h_z_v3"),
    ("trace_context_learned_gate", "all"),
])
def test_retired_model_settings_are_rejected_even_for_saved_trace_configs(field, value):
    raw = _recipe()
    raw["experiment_settings"][field] = value
    with pytest.raises(ValidationError, match=field):
        Experiment(**checkpoint_experiment_config(raw))


@pytest.mark.parametrize("field,value", [
    ("trace_context_team_reward_coefficient", 1.0),
    ("trace_variant", "shuffled"),
])
def test_retired_environment_variants_are_rejected(field, value):
    raw = _recipe()
    raw["environment"][field] = value
    with pytest.raises(ValidationError, match=field):
        Experiment(**checkpoint_experiment_config(raw))


def test_only_inert_nontrace_serialized_defaults_are_removed():
    raw = {
        "experiment_settings": {"encoder_custom": "switcher", "trace_context_architecture": "context"},
        "environment": {"trace_context_team_reward_coefficient": 1.0},
    }
    original = deepcopy(raw)
    normalized = checkpoint_experiment_config(raw)
    assert raw == original
    assert "trace_context_architecture" not in normalized["experiment_settings"]
    assert "trace_context_team_reward_coefficient" not in normalized["environment"]
