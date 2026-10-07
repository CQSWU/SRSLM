from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from learning.config import Experiment, checkpoint_experiment_config


def _recipe():
    root = Path(__file__).resolve().parents[1]
    return yaml.safe_load((root / "learning/train_arpe_final.yaml").read_text())


@pytest.mark.parametrize(
    "field,value",
    [
        ("trace_context_architecture", "context"),
        (
            "trace_context_architecture",
            "paper_entropy_conv_direct_correction_centered_P_h_z_v3",
        ),
        ("trace_context_learned_gate", "all"),
        ("trace_encoder_input", "zero"),
        ("trace_rule_scale", 1.0),
        ("trace_gate_threshold", 0.46371241),
    ],
)
def test_new_recipes_reject_retired_model_settings(field, value):
    raw = _recipe()
    raw["experiment_settings"][field] = value
    with pytest.raises(ValidationError, match=field):
        Experiment(**raw)


@pytest.mark.parametrize(
    "field,value",
    [
        ("trace_context_team_reward_coefficient", 1.0),
        ("trace_variant", "shuffled"),
        ("trace_variant", "zero"),
        ("tau_raw", True),
    ],
)
def test_retired_environment_variants_are_rejected(field, value):
    raw = _recipe()
    raw["environment"][field] = value
    with pytest.raises(ValidationError, match=field):
        Experiment(**raw)


def test_saved_recipe_uses_current_fields_without_modifying_its_input():
    raw = _recipe()
    expected = Experiment(**raw).dict()
    model_fields = {
        "trace_context_architecture": "paper_entropy_fusion",
        "trace_context_learned_gate": "always",
        "trace_encoder_input": "raw",
        "trace_rule_scale": 1.0,
        "trace_gate_threshold": 0.46371241,
    }
    environment_fields = {
        "trace_context_team_reward_coefficient": 0.0,
        "trace_variant": "full",
        "tau_raw": True,
    }
    raw["experiment_settings"].update(model_fields)
    raw["environment"].update(environment_fields)
    original = deepcopy(raw)
    normalized = checkpoint_experiment_config(raw)
    assert raw == original
    assert model_fields.keys().isdisjoint(normalized["experiment_settings"])
    assert environment_fields.keys().isdisjoint(normalized["environment"])
    assert Experiment(**normalized).dict() == expected
