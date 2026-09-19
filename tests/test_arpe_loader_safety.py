"""Reject silent runtime reinterpretation before a paper policy can act."""

from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from agents.arpe import ARPE, ArpeCandidateArtifact
from agents.epom_trace_context import (
    EPOMTraceContext,
    EPOMTraceContextConfig,
    _validate_r5_trace_contract,
)
from agents.policy_backbone import PolicyBackbone, PolicyBackboneConfig


@pytest.mark.parametrize("field,value", [
    ("learned_gate_override", "all"),
    ("entropy_threshold_override", 0.9),
    ("checkpoint_kind", "auto"),
])
def test_retired_inference_overrides_are_not_accepted(field, value):
    with pytest.raises(ValidationError):
        EPOMTraceContextConfig(path_to_weights="unused", **{field: value})


def test_checkpoint_selection_never_silently_falls_back_to_best(tmp_path):
    (tmp_path / "best_0001.pth").touch()
    backbone = PolicyBackbone.__new__(PolicyBackbone)
    with pytest.raises(ValidationError):
        PolicyBackboneConfig(path_to_weights=str(tmp_path), checkpoint_kind="auto")
    with pytest.raises(ValueError, match="explicit checkpoint kind"):
        backbone._load_checkpoint(tmp_path, torch.device("cpu"), "auto")
    with pytest.raises(FileNotFoundError):
        backbone._load_checkpoint(tmp_path, torch.device("cpu"), "latest")


@pytest.mark.parametrize("settings", [
    {"checkpoint_kind": "milestone"},
    {"checkpoint_kind": "milestone", "milestone_checkpoint": " "},
    {"checkpoint_kind": "latest", "milestone_checkpoint": "ignored.pth"},
])
def test_milestone_selection_cannot_be_missing_or_ignored(settings):
    with pytest.raises(ValidationError):
        EPOMTraceContextConfig(path_to_weights="unused", **settings)


def test_milestone_cannot_pair_another_runs_weights_with_current_config(tmp_path):
    current_dir = tmp_path / "run-a" / "checkpoint_p0"
    candidate = tmp_path / "run-b" / "checkpoint_p0" / "checkpoint.pth"
    candidate.parent.mkdir(parents=True)
    candidate.touch()
    adapter = EPOMTraceContext.__new__(EPOMTraceContext)
    adapter.algo_cfg = EPOMTraceContextConfig(
        path_to_weights=str(current_dir.parent),
        checkpoint_kind="milestone",
        milestone_checkpoint=str(candidate),
    )
    with pytest.raises(ValueError, match="declared run"):
        adapter._load_checkpoint(current_dir, torch.device("cpu"), "milestone")


def _trace_config():
    return {
        "experiment_settings": {"trace_context_architecture": "paper_entropy_fusion"},
        "environment": {"tau_radius": 5, "tau_raw": False, "trace_variant": "real"},
    }


@pytest.mark.parametrize("architecture", ["context", "paper_entropy_multiplier", None])
def test_trace_contract_does_not_infer_a_legacy_architecture(architecture):
    config = _trace_config()
    if architecture is None:
        del config["experiment_settings"]["trace_context_architecture"]
    else:
        config["experiment_settings"]["trace_context_architecture"] = architecture
    with pytest.raises(RuntimeError):
        _validate_r5_trace_contract(config)


def test_trace_contract_preserves_centered_real_and_zero_controls():
    config = _trace_config()
    for variant in ("real", "zero"):
        config["environment"]["trace_variant"] = variant
        contract = _validate_r5_trace_contract(config)
        assert contract["tau_raw"] is False
        assert contract["tau_size"] == 11
        assert contract["trace_variant"] == variant


@pytest.mark.parametrize("invalid", ["nan", "wrong_dtype", "not_tensor"])
def test_all_tensor_validation_precedes_parameter_mutation(invalid):
    model = torch.nn.Linear(2, 5)
    original = {key: value.detach().clone() for key, value in model.state_dict().items()}
    checkpoint = {key: torch.full_like(value, 99) for key, value in original.items()}
    if invalid == "nan":
        checkpoint["bias"][0] = float("nan")
    elif invalid == "wrong_dtype":
        checkpoint["bias"] = checkpoint["bias"].double()
    else:
        checkpoint["bias"] = [99] * 5
    with pytest.raises(RuntimeError):
        PolicyBackbone._load_model_state(model, checkpoint, "invalid-policy")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key])


def _artifact(tmp_path):
    mapping = {
        "weights_path": "weights/trace",
        "checkpoint_path": "weights/trace/checkpoint_p0/trace.pth",
        "base_weights_path": "weights/base",
        "base_checkpoint_path": "weights/base/checkpoint_p0/base.pth",
    }
    artifact = ArpeCandidateArtifact.from_mapping(mapping, tmp_path)
    for index, path in enumerate((artifact.config_path, artifact.checkpoint_path,
                                  artifact.base_config_path, artifact.base_checkpoint_path)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(index))
    return artifact, mapping


def _policy(artifact, digests):
    actor = torch.nn.Linear(2, 5)
    actor.checkpoint_provenance = lambda: {
        "base_config_sha256": digests[str(artifact.base_config_path)],
        "base_checkpoint_sha256": digests[str(artifact.base_checkpoint_path)],
    }
    return SimpleNamespace(
        ppo=actor,
        config_sha256=digests[str(artifact.config_path)],
        checkpoint_sha256=digests[str(artifact.checkpoint_path)],
    )


@pytest.mark.parametrize("field", ["config_path", "checkpoint_path", "base_config_path", "base_checkpoint_path"])
def test_declared_artifacts_must_be_the_ones_actually_loaded(tmp_path, field):
    artifact, _ = _artifact(tmp_path)
    inspected = artifact.inspect_files()
    actual = dict(inspected)
    actual[str(getattr(artifact, field))] = "a-different-file-hash"
    with pytest.raises(RuntimeError, match="differ from its declaration"):
        ARPE(_policy(artifact, actual), artifact, verified_file_hashes=inspected)


def test_rehash_does_not_relabel_a_loaded_model_with_new_artifact_hashes(tmp_path):
    artifact, _ = _artifact(tmp_path)
    digests = artifact.inspect_files()
    policy = ARPE(_policy(artifact, digests), artifact, verified_file_hashes=digests)
    assert policy.verify_frozen()["verified"]
    artifact.base_checkpoint_path.write_text("replaced after load")
    with pytest.raises(RuntimeError, match="differ from its declaration"):
        policy.verify_frozen(rehash_files=True)


@pytest.mark.parametrize("field,value", [("kind", "other"), ("schema", "v0"), ("frozen", False)])
def test_candidate_declaration_rejects_another_policy_contract(tmp_path, field, value):
    _, mapping = _artifact(tmp_path)
    mapping[field] = value
    with pytest.raises(ValueError):
        ArpeCandidateArtifact.from_mapping(mapping, tmp_path)


def test_historical_action_sampling_is_explicit_and_unchanged():
    assert EPOMTraceContextConfig(path_to_weights="unused").action_sampling == "torch"
    assert EPOMTraceContextConfig(
        path_to_weights="unused", action_sampling="direct_numpy"
    ).action_sampling == "direct_numpy"
