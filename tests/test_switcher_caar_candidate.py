from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from agents.switcher_caar_candidate import CaarCandidateArtifact


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _artifact_tree(root: Path) -> dict[str, object]:
    candidate = root / "weights" / "candidate"
    base = root / "weights" / "base"
    candidate_checkpoint = candidate / "checkpoint_p0" / "checkpoint_1.pth"
    base_checkpoint = base / "checkpoint_p0" / "checkpoint_2.pth"
    files = {
        candidate / "config.json": b"candidate-config",
        candidate_checkpoint: b"candidate-checkpoint",
        base / "config.json": b"base-config",
        base_checkpoint: b"base-checkpoint",
    }
    for path, payload in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return {
        "kind": "epom_trace_context_caar_milestone",
        "schema": "switcher_candidate_caar_v1",
        "weights_path": "weights/candidate",
        "checkpoint_path": "weights/candidate/checkpoint_p0/checkpoint_1.pth",
        "checkpoint_sha256": _digest(files[candidate_checkpoint]),
        "config_sha256": _digest(files[candidate / "config.json"]),
        "base_weights_path": "weights/base",
        "base_checkpoint_path": "weights/base/checkpoint_p0/checkpoint_2.pth",
        "base_checkpoint_sha256": _digest(files[base_checkpoint]),
        "base_config_sha256": _digest(files[base / "config.json"]),
        "checkpoint_selection": "exact_milestone",
        "frozen": True,
    }


def test_dynamic_caar_artifact_is_relative_hash_pinned_and_reproducible(tmp_path):
    declaration = _artifact_tree(tmp_path)
    artifact = CaarCandidateArtifact.from_mapping(declaration, tmp_path)

    assert artifact.weights_relative == "weights/candidate"
    assert artifact.checkpoint_relative.endswith("checkpoint_1.pth")
    assert len(artifact.verify_files()) == 4
    saved = artifact.as_dict()
    assert saved["weights_path"] == declaration["weights_path"]
    assert saved["checkpoint_sha256"] == declaration["checkpoint_sha256"]

    artifact.checkpoint_path.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="Frozen CAAR input changed"):
        artifact.verify_files()


def test_caar_artifact_rejects_absolute_and_outside_weight_paths(tmp_path):
    declaration = _artifact_tree(tmp_path)
    declaration["checkpoint_path"] = str(
        (tmp_path / "weights" / "candidate" / "checkpoint_p0" / "checkpoint_1.pth").resolve()
    )
    with pytest.raises(ValueError, match="must be relative"):
        CaarCandidateArtifact.from_mapping(declaration, tmp_path)

    declaration = _artifact_tree(tmp_path)
    declaration["weights_path"] = "other/candidate"
    with pytest.raises(ValueError, match="below project/weights"):
        CaarCandidateArtifact.from_mapping(declaration, tmp_path)


def test_caar_checkpoint_must_live_inside_declared_run(tmp_path):
    declaration = _artifact_tree(tmp_path)
    declaration["checkpoint_path"] = (
        "weights/base/checkpoint_p0/checkpoint_2.pth"
    )
    with pytest.raises(ValueError, match="inside weights_path"):
        CaarCandidateArtifact.from_mapping(declaration, tmp_path)
