"""Independent checks of the selected all-action-residual v2 training equation."""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.envs.create_env import create_env
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence

from learning.config import Experiment
from learning.epom_trace_multiplier_actor_critic import (
    EPOMTraceMultiplierActorCritic,
    INDEPENDENT_CRITIC_KIND,
    PAPER_ENTROPY_FUSION_ARCHITECTURE,
    RESIDUAL_SCALE,
    bounded_centered_residual,
    select_top2_low_pressure,
)
from train import register_custom_components, validate_config


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "learning" / "train_arpe.yaml"


def _load(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _routing(batch_size):
    pressure = torch.tensor([[0.0, 5.0, 1.0, -99.0, 0.0]]).repeat(batch_size, 1)
    legal = torch.ones(batch_size, 5)
    legal[:, 0] = 0
    ranks = torch.arange(5).repeat(batch_size, 2, 1).float()
    return pressure, legal, ranks


@pytest.fixture(scope="module")
def full_model():
    register_custom_components()
    raw = _load(FORMAL)
    base = ROOT / raw["experiment_settings"]["epom_base_weights_path"]
    if not ((base / "config.json").is_file() or (base / "cfg.json").is_file()):
        pytest.skip("EPOM-L weights are distributed separately from Git")
    raw["global_settings"]["device"] = "cpu"
    _, cfg = validate_config(raw)
    env = create_env(cfg.env, cfg=cfg, env_config={})
    try:
        model = create_actor_critic(cfg, env.observation_space, env.action_space)
        observations, _ = env.reset()
        batch = TensorDict(
            {
                key: torch.from_numpy(
                    np.stack([observation[key] for observation in observations])
                ).float()
                for key in observations[0]
            }
        )
        batch = prepare_and_normalize_obs(model, batch)
    finally:
        env.close()
    return model, batch, cfg


def test_config_locks_the_selected_arpe_contract():
    experiment = Experiment(**_load(FORMAL))
    settings, environment = experiment.experiment_settings, experiment.environment
    assert settings.trace_context_architecture == PAPER_ENTROPY_FUSION_ARCHITECTURE
    assert settings.encoder_custom == "epom_trace_context"
    assert settings.hidden_size == 512
    assert settings.trace_context_learned_gate == "entropy"
    assert settings.trace_gate_threshold == pytest.approx(0.46371241)
    assert environment.tau_radius == 5
    assert environment.tau_raw is False
    assert environment.grid_memory_obs_radius == 7
    assert environment.grid_config.map_name == "maps/train.yaml"


def test_network_restores_independent_trace_critic_and_605638_parameters(full_model):
    model, _, _ = full_model
    for encoder in (model.actor_trace_encoder, model.critic_trace_encoder):
        assert encoder.network[0].in_channels == 1
        assert encoder.network[0].out_channels == 32
        assert encoder.network[0].kernel_size == (3, 3)
        assert encoder.network[-2].in_features == 32 * 11 * 11
        assert encoder.network[-2].out_features == 32
        assert sum(p.numel() for p in encoder.parameters()) == 161_248
    for fusion in (model.trace_fusion_head, model.critic_fusion_head):
        assert fusion[0].in_features == 549
        assert fusion[0].out_features == 256
        assert sum(p.numel() for p in fusion.parameters()) == 140_800
    assert model.trace_multiplier_head[0].in_features == 256
    assert model.trace_multiplier_head[0].out_features == 5
    assert isinstance(model.trace_value_head, nn.Linear)
    assert model.trace_value_head.in_features == 256
    assert model.trace_value_head.out_features == 1
    assert sum(p.numel() for p in model.trace_multiplier_head.parameters()) == 1_285
    assert sum(p.numel() for p in model.trace_value_head.parameters()) == 257
    assert sum(p.numel() for p in model.trainable_parameters()) == 605_638
    assert model.expected_trainable_parameters == 605_638
    assert model.head_extra_size == 89
    for key, expected in (
        ("paper_entropy_gate_version", 1),
        ("independent_critic_version", 1),
        ("allaction_residual_version", 2),
    ):
        assert model.state_dict()[key].item() == expected


def test_fusion_input_is_trace32_h512_z5_and_detaches_frozen_epom():
    trace = torch.randn(3, 32, requires_grad=True)
    hidden = torch.randn(3, 512, requires_grad=True)
    logits = torch.randn(3, 5, requires_grad=True)
    fused = EPOMTraceMultiplierActorCritic.compose_paper_entropy_fusion_input(
        trace, hidden, logits
    )
    assert fused.shape == (3, 549)
    torch.testing.assert_close(fused[:, :32], trace)
    torch.testing.assert_close(fused[:, 32:544], hidden)
    torch.testing.assert_close(fused[:, 544:], logits)
    fused.sum().backward()
    assert trace.grad is not None
    assert hidden.grad is None
    assert logits.grad is None


def test_residual_uses_tanh_then_five_action_mean_not_raw_subtraction():
    raw = torch.tensor([[20.62, -30.0, 0.0, 2.0, -1.0], [4.0] * 5])
    bounded = 0.5 * torch.tanh(raw)
    expected = bounded - bounded.mean(dim=-1, keepdim=True)
    actual = bounded_centered_residual(raw)
    assert RESIDUAL_SCALE == 0.5
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.mean(dim=-1), torch.zeros(2), atol=1e-7, rtol=0)
    assert actual.abs().max().item() <= 0.8 + 1e-7
    assert not torch.allclose(actual, -raw)


def test_direct_rewards_low_pressure_of_top_two_legal_moves_only():
    base = torch.tensor([[0.0, 0.4, 0.3, 0.2, 0.1]]).repeat(3, 1)
    pressure, legal, ranks = _routing(3)
    legal[1, 1] = 0
    legal[2, 2:] = 0
    actual = select_top2_low_pressure(base, pressure, legal, ranks)
    expected = torch.tensor([[0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]])
    torch.testing.assert_close(actual, expected.to(actual.dtype))


def test_stored_tie_ranks_make_repeated_ppo_forward_identical():
    base = torch.zeros(3, 5)
    pressure = torch.ones_like(base)
    _, legal, ranks = _routing(3)
    torch.manual_seed(10)
    first = select_top2_low_pressure(base, pressure, legal, ranks)
    torch.manual_seed(100)
    second = select_top2_low_pressure(base, pressure, legal, ranks)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first.sum(dim=-1), torch.ones(3))
    torch.testing.assert_close(first[:, 0], torch.zeros(3))


def test_zero_output_keeps_direct_bonus_and_closed_gate_keeps_base():
    base = torch.tensor([[0.0, 0.4, 0.3, 0.2, 0.1], [20.0, 0.4, 0.3, 0.2, 0.1]])
    pressure, legal, ranks = _routing(2)
    final, learned_delta, gate, _ = EPOMTraceMultiplierActorCritic.apply_paper_entropy_correction_rule(
        base, torch.zeros_like(base), pressure, legal, ranks
    )
    expected = base.clone()
    expected[0, 2] += 1.0
    torch.testing.assert_close(final, expected)
    torch.testing.assert_close(learned_delta, torch.zeros_like(base))
    torch.testing.assert_close(gate, torch.tensor([[1.0], [0.0]]))


def test_v2_formula_is_direct_plus_entropy_gated_bounded_centered_residual():
    base = torch.tensor([[0.0, 0.4, 0.3, 0.2, 0.1], [20.0, 0.4, 0.3, 0.2, 0.1]])
    raw = torch.tensor([[2.0, -1.0, 0.0, 4.0, -3.0], [1.0, 2.0, 3.0, 4.0, 5.0]])
    pressure, legal, ranks = _routing(2)
    final, delta, gate, _ = EPOMTraceMultiplierActorCritic.apply_paper_entropy_correction_rule(
        base, raw, pressure, legal, ranks
    )
    residual = 0.5 * torch.tanh(raw)
    residual -= residual.mean(dim=-1, keepdim=True)
    expected = base.clone()
    expected[0, 2] += 1.0
    expected[0] += residual[0]
    torch.testing.assert_close(gate, torch.tensor([[1.0], [0.0]]))
    torch.testing.assert_close(final, expected)
    torch.testing.assert_close(delta, residual * gate)


def test_entropy_closed_rows_have_no_actor_gradient_but_open_rows_adjust_all_actions():
    base = torch.tensor([[0.0, 0.4, 0.3, 0.2, 0.1], [20.0, 0.4, 0.3, 0.2, 0.1]])
    raw = torch.tensor([[0.4, -0.2, 0.1, 0.3, -0.1], [0.7, -0.4, 0.2, 0.1, -0.3]], requires_grad=True)
    pressure, legal, ranks = _routing(2)
    legal[0, 4] = 0
    final, delta, gate, _ = EPOMTraceMultiplierActorCritic.apply_paper_entropy_correction_rule(
        base, raw, pressure, legal, ranks
    )
    F.cross_entropy(final, torch.tensor([1, 2]), reduction="sum").backward()
    torch.testing.assert_close(gate, torch.tensor([[1.0], [0.0]]))
    assert torch.count_nonzero(raw.grad[0]).item() == 5
    torch.testing.assert_close(raw.grad[1], torch.zeros(5))
    assert delta[0, 0] != 0
    assert delta[0, 4] != 0


def test_inference_cannot_silently_change_the_trained_gate():
    assert not hasattr(EPOMTraceMultiplierActorCritic, "set_inference_learned_gate_override")
    assert not hasattr(EPOMTraceMultiplierActorCritic, "set_inference_entropy_threshold_override")
    base = torch.tensor([[20.0, 0.4, 0.3, 0.2, 0.1]])
    pressure, legal, ranks = _routing(1)
    with pytest.raises(TypeError, match="entropy_threshold"):
        EPOMTraceMultiplierActorCritic.apply_paper_entropy_correction_rule(
            base, torch.ones_like(base), pressure, legal, ranks, entropy_threshold=0.0
        )


def test_actor_and_critic_have_independent_trace_gradients_and_frozen_base(full_model):
    model, _, _ = full_model
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.trace_multiplier_head[-1].weight.normal_(0.0, 0.01)
    actor_tau = torch.randn(4, 1, 11, 11, requires_grad=True)
    critic_tau = torch.randn(4, 1, 11, 11, requires_grad=True)
    hidden = torch.randn(4, 512, requires_grad=True)
    logits = torch.zeros(4, 5, requires_grad=True)
    actor_feature = model.actor_trace_encoder(actor_tau)
    fusion = model.compose_paper_entropy_fusion_input(actor_feature, hidden, logits)
    raw = model.trace_multiplier_head(model.trace_fusion_head(fusion))
    pressure, legal, ranks = _routing(4)
    adjusted = model.apply_paper_entropy_correction_rule(logits.detach(), raw, pressure, legal, ranks)[0]
    values = model._critic_values(hidden, model.critic_trace_encoder(critic_tau), logits)
    F.cross_entropy(adjusted, torch.tensor([0, 1, 2, 3])).backward()
    assert actor_tau.grad is not None
    assert critic_tau.grad is None
    assert model.critic_trace_encoder.network[0].weight.grad is None
    model.zero_grad(set_to_none=True)
    actor_tau.grad = None
    values.square().mean().backward()
    assert actor_tau.grad is None
    assert critic_tau.grad is not None
    assert model.actor_trace_encoder.network[0].weight.grad is None
    assert model.critic_trace_encoder.network[0].weight.grad.norm().item() > 0
    assert model.critic_fusion_head[0].weight.grad.norm().item() > 0
    assert model.trace_value_head.weight.grad.norm().item() > 0
    assert hidden.grad is None
    assert logits.grad is None
    assert all(parameter.grad is None for module in model._frozen_base_modules for parameter in module.parameters())


def test_full_forward_retains_direct_and_zero_initial_residual(full_model):
    model, batch, cfg = full_model
    nn.init.zeros_(model.trace_multiplier_head[-1].weight)
    nn.init.zeros_(model.trace_multiplier_head[-1].bias)
    outputs = model(batch, torch.zeros(len(batch["obs"]), cfg.hidden_size))
    assert outputs["action_logits"].shape == (len(batch["obs"]), 5)
    assert outputs["values"].shape == (len(batch["obs"]),)
    assert model.critic_kind == INDEPENDENT_CRITIC_KIND
    assert model.verify_frozen_actor_backbone()["verified"] is True
    torch.testing.assert_close(model.last_learned_delta, torch.zeros_like(model.last_learned_delta))
    torch.testing.assert_close(model.last_final_logits, model.last_direct_logits)
    torch.testing.assert_close(model.last_direct_logits, model.last_base_logits + model.last_rule_delta)
    assert (model.last_rule_delta >= 0).all()
    assert (model.last_rule_delta.sum(dim=-1) <= 1).all()


def test_provenance_reports_checkpoint_semantics_and_independent_critic(full_model):
    model, _, _ = full_model
    provenance = model.checkpoint_provenance()
    assert provenance["actor_inputs"] == [
        "full_crop_centered_trace_1x11x11",
        "frozen_epom_recurrent_hidden_512",
        "frozen_epom_base_logits_5",
    ]
    assert provenance["actor_trainable_parameters"] == 303_333
    assert provenance["critic_trainable_parameters"] == 302_305
    assert provenance["actor_critic_share_trace_trunk"] is False
    assert provenance["critic_architecture"] == INDEPENDENT_CRITIC_KIND
    assert provenance["critic_uses_trace"] is True
    assert provenance["critic_backpropagates_to_epom"] is False
    assert provenance["actor_receives_free_mask_tensor"] is False
    assert provenance["actor_receives_candidate_pressure"] is False
    assert provenance["allaction_residual_version"] == 2
    assert provenance["residual_scale"] == 0.5


def test_packed_rollout_keeps_critic_and_tie_metadata_and_backpropagates(full_model):
    """Exercise the learner's recurrent path with differently sized sequences."""

    model, batch, cfg = full_model
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.trace_multiplier_head[-1].weight.normal_(0.0, 0.01)
    # Unsorted lengths force PyTorch to reorder packed rows and retain both
    # sorted_indices and unsorted_indices, as in a recurrent PPO minibatch.
    lengths = torch.tensor([3, 1, 2])
    sequence_count, padded_length = 3, 3
    observations = TensorDict({key: value[:9] for key, value in batch.items()})
    head = model.forward_head(observations).reshape(padded_length, sequence_count, -1)
    packed_head = pack_padded_sequence(head, lengths, enforce_sorted=False)
    states = torch.zeros(sequence_count, get_rnn_size(cfg))
    packed_core, new_states = model.forward_core(packed_head, states)

    assert isinstance(packed_core, PackedSequence)
    assert packed_core.data.shape == (6, 512 + 89)
    assert new_states.shape == states.shape
    for name in ("batch_sizes", "sorted_indices", "unsorted_indices"):
        torch.testing.assert_close(getattr(packed_core, name), getattr(packed_head, name))
    torch.testing.assert_close(packed_core.data[:, -89:], packed_head.data[:, -89:])
    with torch.no_grad():
        reference_input = model._packed_like(packed_head, packed_head.data[:, :-89].detach())
        reference_core, reference_states = model.core(reference_input, states)
    torch.testing.assert_close(packed_core.data[:, :512], reference_core.data)
    torch.testing.assert_close(new_states, reference_states)

    # Sample Factory supplies unpacked row data to forward_tail. The full 89
    # extra fields must survive, including the stored 2x5 Direct tie ranks.
    result = model.forward_tail(packed_core.data, values_only=False, sample_actions=False)
    assert result["action_logits"].shape == (6, 5)
    assert result["values"].shape == (6,)
    assert torch.isfinite(result["action_logits"]).all()
    assert torch.isfinite(result["values"]).all()
    loss = F.cross_entropy(result["action_logits"], torch.arange(6) % 5)
    loss = loss + result["values"].square().mean()
    loss.backward()
    for module in (model.actor_trace_encoder, model.critic_trace_encoder,
                   model.trace_fusion_head, model.critic_fusion_head,
                   model.trace_multiplier_head, model.trace_value_head):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert all(gradient is not None and torch.isfinite(gradient).all()
                   for gradient in gradients)
    assert all(parameter.grad is None for module in model._frozen_base_modules
               for parameter in module.parameters())
    with torch.no_grad():
        model.trace_multiplier_head[-1].weight.zero_()
        model.trace_multiplier_head[-1].bias.zero_()
