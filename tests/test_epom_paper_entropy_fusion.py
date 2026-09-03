from __future__ import annotations

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
from torch import nn

from learning.config import Experiment
from learning.epom_trace_multiplier_actor_critic import (
    EPOMTraceMultiplierActorCritic,
    HLINEAR_CRITIC_KIND,
    PAPER_ENTROPY_FUSION_ARCHITECTURE,
)
from train import register_custom_components, validate_config


ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "learning" / "train_epom_trace_paper_conv_fusion_r5_500m.yaml"
SMOKE = ROOT / "learning" / "train_epom_trace_paper_conv_fusion_r5_smoke.yaml"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def full_model():
    register_custom_components()
    raw = _load(SMOKE)
    base_dir = ROOT / raw["experiment_settings"]["epom_base_weights_path"]
    if not (
        (base_dir / "config.json").is_file()
        or (base_dir / "cfg.json").is_file()
    ):
        pytest.skip("EPOM-L checkpoint is not included in the source repository")
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


@pytest.mark.parametrize(
    ("path", "steps"),
    [(FORMAL, 500_000_000), (SMOKE, 1_000_000)],
)
def test_configs_are_unique_and_lock_full_centered_epom_l(path, steps):
    experiment = Experiment(**_load(path))
    settings = experiment.experiment_settings
    environment = experiment.environment
    assert settings.trace_context_architecture == PAPER_ENTROPY_FUSION_ARCHITECTURE
    assert settings.encoder_custom == "epom_trace_context"
    assert settings.train_for_env_steps == steps
    assert settings.hidden_size == 512
    assert settings.trace_context_learned_gate == "entropy"
    assert settings.trace_gate_threshold == pytest.approx(0.46371241)
    assert experiment.async_ppo.num_workers == 12
    assert environment.tau_radius == 5
    assert environment.tau_raw is False
    assert environment.grid_memory_obs_radius == 7
    assert environment.grid_config.map_name == "maps/train_capacity_n600.yaml"


def test_network_matches_trace_fusion_figure_and_parameter_budget(full_model):
    model, _, _ = full_model
    trace_conv = model.actor_trace_encoder.network[0]
    trace_linear = model.actor_trace_encoder.network[-2]
    assert trace_conv.in_channels == 1
    assert trace_conv.out_channels == 32
    assert trace_conv.kernel_size == (3, 3)
    assert trace_linear.in_features == 32 * 11 * 11
    assert trace_linear.out_features == 32
    assert model.trace_fusion_head[0].in_features == 549
    assert model.trace_fusion_head[0].out_features == 256
    assert model.trace_multiplier_head[0].in_features == 256
    assert model.trace_multiplier_head[0].out_features == 5
    assert isinstance(model.trace_value_head, nn.Linear)
    assert model.trace_value_head.in_features == 512
    assert model.trace_value_head.out_features == 1

    trace_parameters = sum(p.numel() for p in model.actor_trace_encoder.parameters())
    fusion_parameters = sum(p.numel() for p in model.trace_fusion_head.parameters())
    actor_parameters = sum(p.numel() for p in model.trace_multiplier_head.parameters())
    critic_parameters = sum(p.numel() for p in model.trace_value_head.parameters())
    assert (trace_parameters, fusion_parameters) == (161_248, 140_800)
    assert (actor_parameters, critic_parameters) == (1_285, 513)
    assert sum((trace_parameters, fusion_parameters, actor_parameters, critic_parameters)) == 303_846
    assert model.expected_trainable_parameters == 303_846
    assert sum(p.numel() for p in model.trainable_parameters()) == 303_846


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


def test_zero_actor_output_is_exact_frozen_epom_and_rule_is_direct_p(full_model):
    model, _, _ = full_model
    assert torch.count_nonzero(model.trace_multiplier_head[-1].weight).item() == 0
    assert torch.count_nonzero(model.trace_multiplier_head[-1].bias).item() == 0
    base = torch.zeros(2, 5)
    base[1, 0] = 20.0
    raw_correction = torch.tensor(
        [[2.0, -1.0, 0.0, 4.0, -3.0], [1.0, 2.0, 3.0, 4.0, 5.0]]
    )
    zero_correction = model.trace_multiplier_head(torch.randn(2, 256))
    initial, _, initial_gate, _ = model.apply_paper_entropy_correction_rule(
        base, zero_correction
    )
    torch.testing.assert_close(initial, base)
    torch.testing.assert_close(initial_gate, torch.tensor([[1.0], [0.0]]))

    final, delta, gate, _ = model.apply_paper_entropy_correction_rule(
        base, raw_correction
    )
    torch.testing.assert_close(gate, torch.tensor([[1.0], [0.0]]))
    torch.testing.assert_close(final[0], base[0] - raw_correction[0])
    torch.testing.assert_close(delta[0], -raw_correction[0])
    assert torch.equal(final[1], base[1])


def test_entropy_gate_off_rows_contribute_zero_trace_actor_gradient(full_model):
    """Rows that do not use Trace at inference must not train its Actor."""

    model, _, _ = full_model
    # Uniform logits open the entropy gate; a sharp frozen policy closes it.
    base = torch.zeros(2, 5)
    base[1, 0] = 20.0
    raw_correction = torch.tensor(
        [[0.4, -0.2, 0.1, 0.3, -0.1], [0.7, -0.4, 0.2, 0.1, -0.3]],
        requires_grad=True,
    )
    final, _, gate, _ = model.apply_paper_entropy_correction_rule(
        base, raw_correction
    )
    loss = F.cross_entropy(final, torch.tensor([1, 2]), reduction="sum")
    loss.backward()

    torch.testing.assert_close(gate, torch.tensor([[1.0], [0.0]]))
    assert raw_correction.grad is not None
    assert torch.count_nonzero(raw_correction.grad[0]).item() > 0
    torch.testing.assert_close(
        raw_correction.grad[1], torch.zeros_like(raw_correction.grad[1])
    )


def test_inference_override_disables_only_the_entropy_gate(full_model):
    model, _, _ = full_model
    base = torch.zeros(2, 5)
    base[1, 0] = 20.0
    correction = torch.tensor(
        [[0.4, -0.2, 0.1, 0.3, -0.1], [0.7, -0.4, 0.2, 0.1, -0.3]]
    )

    model.set_inference_learned_gate_override("checkpoint")
    checkpoint_logits, _, checkpoint_gate, checkpoint_entropy = (
        model.apply_effective_paper_correction(base, correction)
    )
    torch.testing.assert_close(checkpoint_gate, torch.tensor([[1.0], [0.0]]))
    torch.testing.assert_close(checkpoint_logits[1], base[1])

    model.set_inference_learned_gate_override("all")
    all_logits, all_delta, all_gate, all_entropy = (
        model.apply_effective_paper_correction(base, correction)
    )
    torch.testing.assert_close(all_gate, torch.ones(2, 1))
    torch.testing.assert_close(all_logits, base - correction)
    torch.testing.assert_close(all_delta, -correction)
    torch.testing.assert_close(all_entropy, checkpoint_entropy)
    provenance = model.checkpoint_provenance()
    assert provenance["checkpoint_learned_gate_mode"] == "entropy"
    assert provenance["inference_learned_gate_override"] == "all"
    assert provenance["entropy_gate_applies_to_correction"] is False
    assert provenance["logit_rule"] == "z_prime_equals_z_minus_p"

    # Do not leak this inference-only setting to the remaining module tests.
    model.set_inference_learned_gate_override("checkpoint")


def test_inference_override_rejects_unknown_modes(full_model):
    model, _, _ = full_model
    with pytest.raises(ValueError, match="checkpoint.*all"):
        model.set_inference_learned_gate_override("invalid")




def test_actor_gradient_and_linear_critic_are_independent(full_model):
    model, _, _ = full_model
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.trace_multiplier_head[-1].weight.normal_(0.0, 0.01)
    tau = torch.randn(4, 1, 11, 11, requires_grad=True)
    hidden = torch.randn(4, 512, requires_grad=True)
    logits = torch.zeros(4, 5, requires_grad=True)
    trace_feature = model.actor_trace_encoder(tau)
    fused = model.compose_paper_entropy_fusion_input(trace_feature, hidden, logits)
    actor_feature = model.trace_fusion_head(fused)
    raw_scores = model.trace_multiplier_head(actor_feature)
    adjusted = model.apply_paper_entropy_correction_rule(
        logits.detach(), raw_scores
    )[0]
    values = model._critic_values(hidden, hidden.new_empty((4, 0)))
    loss = F.cross_entropy(adjusted, torch.tensor([0, 1, 2, 3])) + values.square().mean()
    loss.backward()

    assert model.actor_trace_encoder.network[0].weight.grad.norm().item() > 0.0
    assert model.trace_fusion_head[0].weight.grad.norm().item() > 0.0
    assert model.trace_multiplier_head[0].weight.grad.norm().item() > 0.0
    assert model.trace_value_head.weight.grad.norm().item() > 0.0
    assert hidden.grad is None
    assert logits.grad is None
    assert tau.grad is not None
    assert all(
        parameter.grad is None
        for module in model._frozen_base_modules
        for parameter in module.parameters()
    )


def test_full_forward_and_frozen_base_contract(full_model):
    model, batch, cfg = full_model
    nn.init.zeros_(model.trace_multiplier_head[-1].weight)
    nn.init.zeros_(model.trace_multiplier_head[-1].bias)
    outputs = model(batch, torch.zeros(len(batch["obs"]), cfg.hidden_size))
    assert outputs["action_logits"].shape == (len(batch["obs"]), 5)
    assert outputs["values"].shape == (len(batch["obs"]),)
    assert model.reweight_mode == PAPER_ENTROPY_FUSION_ARCHITECTURE
    assert model._resolved_critic_kind() == HLINEAR_CRITIC_KIND
    assert model.verify_frozen_actor_backbone()["verified"] is True
    assert all(
        not parameter.requires_grad
        for module in model._frozen_base_modules
        for parameter in module.parameters()
    )
    torch.testing.assert_close(
        model.last_multipliers,
        torch.zeros_like(model.last_multipliers),
    )


def test_provenance_matches_figure_and_excludes_mask(full_model):
    model, _, _ = full_model
    provenance = model.checkpoint_provenance()
    assert provenance["trace_architecture"] == (
        "paper_entropy_conv_direct_correction_centered_P_h_z_v3"
    )
    assert provenance["actor_inputs"] == [
        "full_crop_centered_trace_1x11x11",
        "frozen_epom_recurrent_hidden_512",
        "frozen_epom_base_logits_5",
    ]
    assert provenance["trace_encoder_parameters"] == 161_248
    assert provenance["feature_fusion_parameters"] == 140_800
    assert provenance["actor_head_parameters"] == 1_285
    assert provenance["critic_head_parameters"] == 513
    assert provenance["actor_critic_share_trace_trunk"] is False
    assert provenance["critic_architecture"] == HLINEAR_CRITIC_KIND
    assert provenance["critic_inputs"] == ["frozen_epom_hidden_512"]
    assert provenance["critic_uses_trace"] is False
    assert provenance["learned_network_uses_free_mask"] is False
    assert provenance["actor_receives_free_mask_tensor"] is False
    assert provenance["actor_receives_candidate_pressure"] is False
    assert provenance["actor_output"] == "five_direct_logit_corrections_p"
    assert provenance["logit_rule"] == (
        "z_prime_equals_z_minus_entropy_gated_p"
    )
    assert provenance["correction_source"] == "feature_fusion_linear_output_5"
    assert provenance["correction_centering"] == "none"
    assert provenance["correction_bound"] == "unbounded_logit_space"
    assert provenance["zero_initial_correction"] is True
    assert provenance["entropy_gate_applies_to_correction"] is True
    assert "candidate_pressure_source" not in provenance
    assert "multiplier_range" not in provenance
