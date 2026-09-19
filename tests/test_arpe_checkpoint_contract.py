"""Checkpoint compatibility includes the forward equation, not just actor shape."""

import pytest
import torch

from agents.policy_backbone import PolicyBackbone


def _model():
    model = torch.nn.Module()
    model.actor = torch.nn.Linear(2, 5)
    model.critic_trace_encoder = torch.nn.Linear(3, 2)
    model.critic_fusion_head = torch.nn.Linear(2, 2)
    model.trace_value_head = torch.nn.Linear(2, 1)
    model.register_buffer("fixed_entropy_threshold", torch.tensor(0.46371241))
    model.register_buffer("paper_entropy_gate_version", torch.tensor(1))
    model.register_buffer("independent_critic_version", torch.tensor(1))
    model.register_buffer("allaction_residual_version", torch.tensor(2))
    return model


def _snapshot(model):
    return {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}


def test_matching_v2_checkpoint_loads_actor_critic_and_all_buffers():
    model = _model()
    checkpoint = _snapshot(model)
    for key in checkpoint:
        if key.endswith("weight") or key.endswith("bias"):
            checkpoint[key].fill_(3)

    PolicyBackbone._load_model_state(model, checkpoint, "selected-v2-checkpoint")

    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, checkpoint[key])


@pytest.mark.parametrize(
    "key",
    [
        "actor.weight",
        "critic_trace_encoder.weight",
        "critic_fusion_head.weight",
        "trace_value_head.weight",
        "fixed_entropy_threshold",
        "paper_entropy_gate_version",
        "independent_critic_version",
        "allaction_residual_version",
    ],
)
def test_missing_actor_critic_or_contract_buffer_is_rejected(key):
    model = _model()
    checkpoint = _snapshot(model)
    del checkpoint[key]
    with pytest.raises(RuntimeError, match="missing=") as caught:
        PolicyBackbone._load_model_state(model, checkpoint, "incomplete-checkpoint")
    assert key in str(caught.value)


@pytest.mark.parametrize(
    "key,replacement",
    [
        ("fixed_entropy_threshold", 0.5),
        ("paper_entropy_gate_version", 0),
        ("independent_critic_version", 0),
        ("allaction_residual_version", 1),
    ],
)
def test_same_shape_different_forward_contract_is_rejected_before_mutation(key, replacement):
    model = _model()
    original = _snapshot(model)
    checkpoint = _snapshot(model)
    checkpoint["actor.weight"].fill_(99)
    checkpoint[key].fill_(replacement)

    with pytest.raises(RuntimeError, match="semantic_mismatches") as caught:
        PolicyBackbone._load_model_state(model, checkpoint, "wrong-forward-rule")
    assert key in str(caught.value)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, original[name])


def test_wrong_critic_shape_is_rejected_before_actor_mutation():
    model = _model()
    original = _snapshot(model)
    checkpoint = _snapshot(model)
    checkpoint["actor.weight"].fill_(99)
    checkpoint["trace_value_head.weight"] = torch.zeros(1, 512)

    with pytest.raises(RuntimeError, match="shape_mismatches"):
        PolicyBackbone._load_model_state(model, checkpoint, "wrong-critic")
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, original[key])


def test_version_buffer_dtype_mismatch_is_rejected_before_actor_mutation():
    model = _model()
    original = _snapshot(model)
    checkpoint = _snapshot(model)
    checkpoint["actor.weight"].fill_(99)
    checkpoint["allaction_residual_version"] = checkpoint["allaction_residual_version"].float()
    with pytest.raises(RuntimeError, match="semantic_mismatches"):
        PolicyBackbone._load_model_state(model, checkpoint, "wrong-buffer-dtype")
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, original[key])


def test_unexpected_critic_tensor_is_rejected():
    model = _model()
    checkpoint = _snapshot(model)
    checkpoint["critic_trace_encoder.extra.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="unexpected=") as caught:
        PolicyBackbone._load_model_state(model, checkpoint, "extra-critic")
    assert "critic_trace_encoder.extra.weight" in str(caught.value)
