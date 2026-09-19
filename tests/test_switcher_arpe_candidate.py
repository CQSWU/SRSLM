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
    assert len(artifact.inspect_files()) == 4
    saved = artifact.as_dict()
    assert saved["weights_path"] == declaration["weights_path"]

    # Replacing a user checkpoint is allowed; its new digest is simply recorded.
    before = artifact.inspect_files()[str(artifact.checkpoint_path)]
    artifact.checkpoint_path.write_bytes(b"changed")
    after = artifact.inspect_files()[str(artifact.checkpoint_path)]
    assert after != before


def test_arpe_artifact_accepts_absolute_paths(tmp_path):
    declaration = _artifact_tree(tmp_path)
    declaration["checkpoint_path"] = str(
        (tmp_path / "weights" / "candidate" / "checkpoint_p0" / "checkpoint_1.pth").resolve()
    )
    artifact = ArpeCandidateArtifact.from_mapping(declaration, tmp_path)
    assert artifact.checkpoint_path.is_absolute()
    assert artifact.checkpoint_path.is_file()
