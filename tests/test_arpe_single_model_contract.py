"""The public model cannot reinterpret an old checkpoint as current ARPE."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml
from pydantic import ValidationError

from learning.config import Experiment, checkpoint_experiment_config
from learning.epom_trace_multiplier_actor_critic import EPOMTraceMultiplierActorCritic


def _recipe():
    root = Path(__file__).resolve().parents[1]
    return yaml.safe_load((root / "learning/train_arpe.yaml").read_text())


def _contract_model():
    """Use the real checkpoint loader without requiring separately held weights."""
    model = EPOMTraceMultiplierActorCritic.__new__(EPOMTraceMultiplierActorCritic)
    torch.nn.Module.__init__(model)
    model.trace_value_head = torch.nn.Linear(2, 1)
    for name, value in (
        ("paper_entropy_gate_version", 1),
        ("independent_critic_version", 1),
        ("allaction_residual_version", 2),
    ):
        model.register_buffer(name, torch.tensor(value, dtype=torch.int64))
    model.register_buffer("fixed_entropy_threshold", torch.tensor(0.46371241, dtype=torch.float64))
    return model


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


def test_model_rejects_non_strict_learned_checkpoint_load():
    model = _contract_model()
    with pytest.raises(RuntimeError, match="require strict=True"):
        model.load_state_dict(model.state_dict(), strict=False)


def test_model_loads_the_exact_semantic_contract():
    model = _contract_model()
    model.load_state_dict(model.state_dict())


@pytest.mark.parametrize("key,replacement", [
    ("paper_entropy_gate_version", torch.tensor(0, dtype=torch.int64)),
    ("independent_critic_version", torch.tensor(0, dtype=torch.int64)),
    ("allaction_residual_version", torch.tensor(1, dtype=torch.int64)),
    ("allaction_residual_version", torch.tensor(2, dtype=torch.float32)),
    ("fixed_entropy_threshold", torch.tensor(0.0, dtype=torch.float64)),
    ("fixed_gap_threshold", torch.tensor(0.0)),
])
def test_model_rejects_semantic_mismatch_before_mutating_parameters(key, replacement):
    model = _contract_model()
    original = {name: value.clone() for name, value in model.state_dict().items()}
    changed = {name: value.clone() for name, value in original.items()}
    changed["trace_value_head.weight"].fill_(99.0)
    changed[key] = replacement
    with pytest.raises(RuntimeError, match="ARPE|entropy threshold"):
        model.load_state_dict(changed)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name])
