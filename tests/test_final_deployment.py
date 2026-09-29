"""Final inference is explicit and does not relabel or modify trained weights."""

from copy import deepcopy
import json
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from agents.arpe import ARPE, ArpeCandidateArtifact
from agents.switcher_core import SwitcherController
from learning.config import Experiment
from learning.epom_trace_multiplier_actor_critic import (
    EPOMTraceMultiplierActorCritic, bounded_centered_residual, select_top2_low_pressure,
)
from learning.inference_correction import InferenceCorrection
from planning.aoreplan_branch import AORePlanStep

ROOT = Path(__file__).resolve().parents[1]




def test_inference_gate_is_strict_and_scales_after_centering():
    profile = InferenceCorrection()
    entropy = torch.tensor([0.0, 0.01, 0.010001, 1.0], dtype=torch.float64)
    logits = torch.arange(20, dtype=torch.float64).reshape(4, 5)
    raw = logits / 3
    residual = bounded_centered_residual(raw)
    final, learned, direct_gate, _ = profile.apply(logits, residual, entropy)
    assert profile.gate(entropy).flatten().tolist() == [False, False, True, True]
    assert torch.equal(final[:2], logits[:2])
    assert torch.equal(learned[2:], 12 * residual[2:])
    assert torch.equal(final[2:], logits[2:] + 12 * residual[2:])
    assert torch.count_nonzero(direct_gate) == 0


@pytest.mark.parametrize("gate_mode", ["entropy", "always"])
def test_native_training_gate_remains_separate_from_inference(gate_mode):
    logits = torch.tensor([[100., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]])
    raw = torch.tensor([[1., -1., 2., -2., 0.]]).repeat(2, 1)
    pressure = torch.tensor([[0., 4., 1., 2., 3.]]).repeat(2, 1)
    legal = torch.tensor([[0., 1., 1., 1., 1.]]).repeat(2, 1)
    ranks = torch.arange(5).float().repeat(2, 2, 1)
    model = SimpleNamespace(
        learned_gate_mode=gate_mode, rule_scale=1.0, training=False,
        _base_entropy=EPOMTraceMultiplierActorCritic._base_entropy,
        apply_paper_entropy_correction_rule=EPOMTraceMultiplierActorCritic.apply_paper_entropy_correction_rule,
    )
    apply = EPOMTraceMultiplierActorCritic._apply_configured_correction_rule
    native, _, gate, _ = apply(model, logits, raw, pressure, legal, ranks)
    expected_gate = torch.ones(2, 1) if gate_mode == "always" else torch.tensor([[0.], [1.]])
    assert torch.equal(gate, expected_gate)
    route = select_top2_low_pressure(logits, pressure, legal, ranks)
    torch.testing.assert_close(native, logits + gate * route + gate * bounded_centered_residual(raw))
    model.inference_correction = InferenceCorrection()
    final, delta, direct_gate, _ = apply(model, logits, raw, pressure, legal, ranks)
    assert torch.equal(final[0], logits[0])
    assert torch.equal(delta[1], 12 * bounded_centered_residual(raw)[1])
    assert not direct_gate.any()
    assert model.learned_gate_mode == gate_mode and model.rule_scale == 1.0
    model.training = True
    with pytest.raises(RuntimeError, match="training"):
        apply(model, logits, raw, pressure, legal, ranks)


def _observations(position=(0, 0), target=(0, 5)):
    return [{"obstacles": np.zeros((11, 11), np.float32),
             "agents": np.zeros((11, 11), np.float32),
             "xy": position, "target_xy": target}]


class _Candidate:
    def act(self, *_args):
        return [1]

    def after_reset(self):
        pass


class _Planner:
    action = 4
    reverse = False

    def reset(self):
        pass

    def propose(self, _observations):
        return AORePlanStep(actions=(self.action,), planned_mask=(True,),
                            reverse_mask=(self.reverse,), static_astar_invoked_mask=(self.reverse,))

    def commit(self, selected):
        self.committed = tuple(selected)


@pytest.mark.parametrize("branch,expected_action", [(0, 1), (1, 3)])
def test_wait_only_controller_leaves_reverse_actions_to_switcher(branch, expected_action):
    planner = _Planner()
    controller = SwitcherController(_Candidate(), planner)
    first = controller.prepare_actions(_observations())
    assert first.switch_allowed_mask == (True,)
    controller.resolve_actions([1])
    planner.action = 3
    planner.reverse = True
    second = controller.prepare_actions(_observations((0, 1)))
    # This move returns to the previous position, even after a static A* query.
    # It must still be selected by the network, not an extra reverse rule.
    assert second.switch_allowed_mask == (True,)
    assert not second.switcher_state["aoreplan_action"][:, 0].any()
    assert controller.resolve_actions([branch]).actions == (expected_action,)
    assert planner.committed == (branch == 1,)
    planner.action = 0
    third = controller.prepare_actions(_observations((0, 1)))
    assert third.switch_allowed_mask == (False,)
    assert third.switcher_state["aoreplan_action"][:, 0].all()
    assert controller.resolve_actions([]).actions == (1,)
    assert planner.committed == (False,)
    stats = controller.get_stats()
    assert stats["total_action_count"] == 3
    assert stats["switcher_choice_count"] == 2
    assert stats["aoreplan_wait_bypass_count"] == 1
    assert stats["switcher_decision_scope"] == "aoreplan_nonwait_only"
    assert "final_reverse_arpe_bypass_count" not in stats
    controller.after_reset()
    assert controller.get_stats()["total_action_count"] == 0
    planner.action = 3
    assert controller.prepare_actions(_observations((0, 1))).switch_allowed_mask == (True,)




def test_training_and_deployment_share_wait_only_controller():
    import inspect
    from agents.srslm import SRSLM, SRSLMConfig
    from pomapf_env.switcher_arpe_env import ArpeSwitcherEnv

    assert ArpeSwitcherEnv.controller_class is SwitcherController
    assert inspect.signature(SRSLM).parameters["controller_factory"].default is SwitcherController
    assert "final_reverse_guard_enabled" not in SRSLMConfig.__fields__


def _reference_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_as_run_arpe_adapter_matches_when_reference_is_supplied(monkeypatch):
    reference_dir = os.environ.get("SRSLM_REFERENCE_SOURCE")
    if not reference_dir:
        pytest.skip("Optional immutable as-run reference is not part of the public source")
    reference_dir = Path(reference_dir)
    common = _reference_module(reference_dir / "deployment_adapter/common.py", "_srslm_reference_common")
    monkeypatch.setitem(sys.modules, "common", common)
    adapter = _reference_module(reference_dir / "deployment_adapter/gate_adapter.py", "_srslm_reference_adapter")
    rng = np.random.default_rng(19)
    model = SimpleNamespace(_base_entropy=EPOMTraceMultiplierActorCritic._base_entropy)
    for _ in range(20):
        logits = torch.from_numpy(rng.normal(size=(17, 5))).float()
        raw = torch.from_numpy(rng.normal(size=(17, 5))).float()
        expected = adapter._boost_rule(model, logits, raw, None, None, None)
        actual = InferenceCorrection().apply(logits, bounded_centered_residual(raw), model._base_entropy(logits))
        assert all(torch.equal(left, right) for left, right in zip(actual, expected))


def test_final_manifest_and_training_example_have_distinct_gates():
    mapping = json.loads((ROOT / "configs/arpe_final_candidate.json").read_text())
    artifact = ArpeCandidateArtifact.from_mapping(mapping, ROOT)
    assert artifact.inference == InferenceCorrection()
    assert not any("sha" in key.lower() for key in mapping)
    cfg = Experiment(**yaml.safe_load((ROOT / "learning/train_arpe_final.yaml").read_text()))
    assert cfg.experiment_settings.trace_context_learned_gate == "always"
    assert cfg.experiment_settings.trace_rule_scale == 1
    assert cfg.experiment_settings.train_for_env_steps == 1_000_000_000


def test_final_weight_loading_and_small_cpu_forward_when_available():
    """No download and no rollout; release verification supplies three bundles."""
    mapping = json.loads((ROOT / "configs/arpe_final_candidate.json").read_text())
    artifact = ArpeCandidateArtifact.from_mapping(mapping, ROOT)
    switcher_dir = ROOT / "weights/SRSLM-Switcher-Final-1B"
    if not artifact.checkpoint_path.is_file() or not (switcher_dir / "config.json").is_file():
        pytest.skip("Final model weights are distributed separately")
    from agents.srslm import SRSLM, SRSLMConfig
    from agents.arpe import ARPEConfig
    from agents.switcher import SwitcherConfig
    from agents.switcher_core import build_switcher_state
    from pomapf_env.trace_routing import TIE_KEY
    from sample_factory.algo.utils.tensor_dict import TensorDict
    from sample_factory.model.model_utils import get_rnn_size

    source_bytes = artifact.config_path.read_bytes()
    candidate_cfg = ARPEConfig(
        path_to_weights=str(artifact.weights_path), milestone_checkpoint=str(artifact.checkpoint_path),
        base_weights_path=str(artifact.base_weights_path), base_checkpoint_path=str(artifact.base_checkpoint_path),
        inference=artifact.inference.as_dict(),
    )
    policy = SRSLM(SRSLMConfig(
        device="cpu", candidate=candidate_cfg,
        switcher=SwitcherConfig(path_to_weights=str(switcher_dir), device="cpu"),
    ), project_root=ROOT)
    model = policy.candidate.ppo
    assert model.learned_gate_mode == "always" and model.rule_scale == 1
    assert int(model.paper_entropy_gate_version) == 0
    assert policy.candidate.policy.algo_cfg.action_sampling == "direct_numpy"
    before = {key: value.clone() for key, value in model.state_dict().items()}
    config_before = deepcopy(model.cfg.full_config)
    batch = TensorDict(obs=torch.zeros(2, 3, 15, 15), xy=torch.zeros(2, 2),
                       target_xy=torch.ones(2, 2), tau=torch.zeros(2, 1, 11, 11),
                       tau_free_mask=torch.ones(2, 1, 11, 11))
    batch[TIE_KEY] = torch.arange(5).float().repeat(2, 2, 1)
    with torch.no_grad():
        result = model(batch, torch.zeros(2, get_rnn_size(policy.candidate.policy.cfg)))
    expected = model.last_base_logits + (
        (model.last_base_entropy > .01).unsqueeze(-1) * 12
        * bounded_centered_residual(model.last_raw_correction)
    )
    assert torch.equal(result["action_logits"], expected)
    assert not model.last_rule_delta.any() and not model.last_gate.any()
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key]), key
    assert model.cfg.full_config == config_before
    assert artifact.config_path.read_bytes() == source_bytes
    state = build_switcher_state(_observations() * 2, [1, 2], [4, 3])
    state["switch_allowed"] = np.ones((2, 1), dtype=np.float32)
    assert policy.switcher.choose(state).shape == (2,)
