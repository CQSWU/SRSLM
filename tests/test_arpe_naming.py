"""Current method/API names change without rewriting checkpoint identities."""
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
    assert "arpe_weights_path" not in parameters
    assert "caar_candidate_manifest" not in parameters
    with patch("sys.argv", ["run_experiments.py", "--algorithms", "ARPE",
                           "--arpe-candidate-manifest", "configs/arpe_final_candidate.json"]):
        args = runner.parse_args()
    assert args.arpe_candidate_manifest == "configs/arpe_final_candidate.json"
    with pytest.raises(ValueError, match="Unsupported"):
        runner.build_algorithm("CAAR", ROOT, 0)
    with patch("sys.argv", ["run_experiments.py", "--arpe-weights-path", "ignored"]):
        with pytest.raises(SystemExit):
            runner.parse_args()


def test_selected_declaration_contains_portable_weight_paths():
    data = json.loads(
        (ROOT / "configs" / "arpe_final_candidate.json").read_text()
    )
    assert data["kind"] == "epom_trace_context_caar_milestone"
    assert data["schema"] == "switcher_candidate_caar_v1"
    artifact = ArpeCandidateArtifact.from_mapping(data, ROOT)
    saved = artifact.as_dict()
    assert saved["label"] == "ARPE"
    assert saved["weights_path"] == data["weights_path"]
    assert saved["checkpoint_path"] == data["checkpoint_path"]


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
