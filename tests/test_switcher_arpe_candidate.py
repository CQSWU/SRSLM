from __future__ import annotations

from pathlib import Path

from agents.arpe import ArpeCandidateArtifact


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
        "base_weights_path": "weights/base",
        "base_checkpoint_path": "weights/base/checkpoint_p0/checkpoint_2.pth",
        "frozen": True,
    }


def test_arpe_artifact_accepts_simple_paths_and_reports_provenance(tmp_path):
    declaration = _artifact_tree(tmp_path)
    artifact = ArpeCandidateArtifact.from_mapping(declaration, tmp_path)

    assert artifact.weights_relative == "weights/candidate"
    assert artifact.checkpoint_relative.endswith("checkpoint_1.pth")
    assert artifact.weights_path == tmp_path / declaration["weights_path"]
    assert artifact.base_checkpoint_path == tmp_path / declaration["base_checkpoint_path"]
    saved = artifact.as_dict()
    assert saved["weights_path"] == declaration["weights_path"]

    # Paths remain usable when users supply a different checkpoint.
    artifact.checkpoint_path.write_bytes(b"changed")
    reloaded = ArpeCandidateArtifact.from_mapping(saved, tmp_path)
    assert reloaded == artifact
    assert reloaded.checkpoint_path.read_bytes() == b"changed"


def test_arpe_artifact_accepts_absolute_paths(tmp_path):
    declaration = _artifact_tree(tmp_path)
    declaration["checkpoint_path"] = str(
        (tmp_path / "weights" / "candidate" / "checkpoint_p0" / "checkpoint_1.pth").resolve()
    )
    artifact = ArpeCandidateArtifact.from_mapping(declaration, tmp_path)
    assert artifact.checkpoint_path.is_absolute()
    assert artifact.checkpoint_path.is_file()
