"""Current method/API names change without rewriting checkpoint identities."""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
from unittest.mock import patch

import pytest

import run_experiments as runner
from agents.arpe import ARPE, ARPEConfig, ArpeCandidateArtifact
from agents.switcher_core import ARPE_BRANCH
from pomapf_env.switcher_arpe_env import switcher_observation_space

ROOT = Path(__file__).resolve().parents[1]


def test_new_name_is_the_only_current_method_and_cli_namespace():
    assert "ARPE" in runner.SUPPORTED_ALGORITHMS
    assert "CAAR" not in runner.SUPPORTED_ALGORITHMS
    assert runner.canonical_algorithm_name("arpe") == "ARPE"
    assert runner.canonical_algorithm_name("CAAR") is None
    parameters = inspect.signature(runner.build_algorithm).parameters
    assert "arpe_candidate_manifest" in parameters
    assert "arpe_weights_path" in parameters
    assert "caar_candidate_manifest" not in parameters
    with patch("sys.argv", ["run_experiments.py", "--algorithms", "ARPE",
                           "--arpe-candidate-manifest", "configs/arpe_final_candidate.json"]):
        args = runner.parse_args()
    assert args.arpe_candidate_manifest == "configs/arpe_final_candidate.json"
    with pytest.raises(ValueError, match="Unsupported"):
        runner.build_algorithm("CAAR", ROOT, 0)


def test_selected_declaration_bytes_and_legacy_serialized_identity_are_exact():
    payload = (ROOT / "configs" / "arpe_final_candidate.json").read_bytes()
    assert hashlib.sha256(payload).hexdigest() == (
        "41f9f27429a30186ffef55a622284703cfb859fccb440934f4e585022eb885ca"
    )
    data = json.loads(payload)
    assert data["kind"] == "epom_trace_context_caar_milestone"
    assert data["schema"] == "switcher_candidate_caar_v1"
    artifact = ArpeCandidateArtifact.from_mapping(data, ROOT)
    saved = artifact.as_dict()
    assert saved["label"] == "ARPE"
    for key in ("checkpoint_sha256", "config_sha256",
                "base_checkpoint_sha256", "base_config_sha256"):
        assert saved[key] == data[key]


def test_checkpoint_input_order_and_old_backbone_are_not_rebranded_as_arpe():
    assert ARPE_BRANCH == 0
    space = switcher_observation_space()
    assert "caar_action" in space.spaces  # state_dict-compatible serialized input
    assert "arpe_action" not in space.spaces
    assert space["caar_action"].shape == (5,)
    assert ARPEConfig.__fields__["name"].default == "ARPE"
    assert importlib.util.find_spec("learning.no_reweight_encoder") is None
    assert importlib.util.find_spec("agents.caar") is None
    assert importlib.util.find_spec("agents.switcher_caar_candidate") is None
    assert importlib.util.find_spec("pomapf_env.switcher_caar_env") is None


def test_historical_certificate_name_mapping_is_narrow_and_nonmutating():
    from copy import deepcopy
    from scripts.artifact_utils import same_training_certificate
    saved = {"network_contract": {"branch_0": "CAAR", "branch_1": "AORePlan"},
             "checkpoint_sha256": "a" * 64, "source_manifest": {"sha256": "b" * 64}}
    untouched = deepcopy(saved)
    rebuilt = deepcopy(saved)
    rebuilt["network_contract"]["branch_0"] = "ARPE"
    assert same_training_certificate(saved, rebuilt)
    assert saved == untouched
    for path, value in [("checkpoint_sha256", "c" * 64),
                        ("source_manifest", {"sha256": "c" * 64})]:
        wrong = deepcopy(rebuilt)
        wrong[path] = value
        assert not same_training_certificate(saved, wrong)
    for branch in ("EPOM", "NoReweight", "Other"):
        wrong = deepcopy(saved)
        wrong["network_contract"]["branch_0"] = branch
        assert not same_training_certificate(wrong, rebuilt)
    wrong = deepcopy(rebuilt)
    wrong["network_contract"]["branch_1"] = "RePlan"
    assert not same_training_certificate(saved, wrong)
