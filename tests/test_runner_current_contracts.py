import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import run_experiments as runner
from agents.ao_replan import AORePlan, AORePlanConfig
from planning.ao_replan_algo import AORePlanWrapper


def test_public_names_are_explicit_and_retired_names_are_rejected():
    assert runner.SUPPORTED_ALGORITHMS == ("AORePlan", "ARPE", "SRSLM")
    assert set(runner.ALGORITHM_ALIASES.values()) == set(runner.SUPPORTED_ALGORITHMS)
    for name in (
        "DCC",
        "DHC",
        "Follower",
        "NoReweight",
        "SRSLM-NoWaitDetect",
        "SRSLM-WaitDetectOnly",
        "v8b",
        "RePlan",
        "EPOM-L",
        "EPOM-Lifelong-FT",
        "Direct",
        "SRSLM-NoRule",
        "SRSLM-OnlyRule",
        "SRSLM-NoWait",
        "SRSLM-OnlyWait",
        "AORePlan-SoftNoCheck",
    ):
        with pytest.raises(Exception):
            runner.parse_algorithms(name)


def test_public_cli_defaults_to_aoreplan_only():
    with patch.object(sys, "argv", ["run_experiments.py"]):
        args = runner.parse_args()
    assert args.algorithms == ["AORePlan"]
    assert runner.parse_algorithms("all") == ["AORePlan"]
    assert runner.parse_algorithms("AORePlan,ARPE,SRSLM") == [
        "AORePlan",
        "ARPE",
        "SRSLM",
    ]


def test_learned_policies_are_always_episode_fresh():
    for name in ("ARPE", "SRSLM"):
        assert not runner.should_cache_algorithm(name, True)
    assert runner.should_cache_algorithm("AORePlan", True)


def test_retired_direct_cli_variants_are_rejected():
    with (
        patch.object(
            sys,
            "argv",
            [
                "run_experiments.py",
                "--algorithms",
                "Direct",
                "--direct-transform",
                "clipped_relu",
            ],
        ),
        pytest.raises(SystemExit),
    ):
        runner.parse_args()


def _artifact():
    return SimpleNamespace(
        weights_path=Path("weights/caar").resolve(),
        checkpoint_path=Path("weights/caar/checkpoint_p0/model.pth").resolve(),
    )


def test_arpe_loads_exact_frozen_candidate():
    artifact = _artifact()
    with (
        patch.object(runner, "_load_arpe_candidate_artifact", return_value=artifact),
        patch("agents.arpe.ARPE.load") as load,
    ):
        runner.build_algorithm("ARPE", ".", 42, arpe_candidate_manifest="manifest.json")
    assert load.call_args.args[0] is artifact
    assert load.call_args.kwargs["seed"] == 42
    assert load.call_args.kwargs["action_sampling"] == "direct_numpy"


@pytest.mark.parametrize("collision", ["block_both", "soft"])
def test_default_aoreplan_keeps_static_occupancy_check(collision):
    agent = AORePlan(AORePlanConfig())
    agent.set_grid_config(SimpleNamespace(collision_system=collision))
    assert agent.WRAPPER_CLASS is AORePlanWrapper
    wrapper = object.__new__(agent.WRAPPER_CLASS)
    wrapper.static_astar = SimpleNamespace(get_action=lambda _index, _observation: 4)
    wrapper.last_static_astar_invoked_mask = [False]
    wrapper.moves = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
    agents = np.zeros((11, 11), dtype=int)
    agents[5, 6] = 1
    obs = {"obstacles": np.zeros((11, 11)), "agents": agents}
    assert wrapper._static_astar_action(0, obs) == 0


def test_public_manifest_uses_portable_paths_without_required_hashes():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "configs/arpe_final_candidate.json").read_text())
    from agents.arpe import ArpeCandidateArtifact

    artifact = ArpeCandidateArtifact.from_mapping(data, root)
    assert artifact.weights_path == (root / data["weights_path"]).resolve()
    assert artifact.checkpoint_path == (root / data["checkpoint_path"]).resolve()
    for key in ("checkpoint_sha256", "base_checkpoint_sha256"):
        assert key not in data


def test_map_snapshot_returns_selected_grids_without_archival_hashes(tmp_path):
    import yaml

    registry = tmp_path / "maps.yaml"
    registry.write_text(yaml.safe_dump({"tiny": "...\n.#.\n..."}))
    names, grids = runner.load_map_list_snapshot(registry)
    assert names == {"tiny": "tiny"}
    assert grids == {"tiny": "...\n.#.\n..."}
    selection = tmp_path / "selection.yaml"
    selection.write_text(yaml.safe_dump({"tiny": None}))
    assert runner.load_map_list_snapshot(selection, registry) == (names, grids)


def test_saved_metadata_keeps_protocol_without_weight_or_host_reports(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_experiments.py",
            "--algorithms",
            "SRSLM",
            "--agents",
            "4",
            "--seeds",
            "42",
            "--max-steps",
            "16",
            "--workers",
            "1",
            "--map-types",
            "random",
            "--collision-system",
            "soft",
        ],
    )
    monkeypatch.setattr(runner, "run_experiments", lambda *_a, **_k: ([], 0.0))
    monkeypatch.setattr(
        runner, "save_results", lambda _rows, meta, *_a: captured.update(meta)
    )
    runner.main()
    assert captured["algorithms"] == ["SRSLM"]
    assert captured["agent_counts"] == [4]
    assert captured["seeds"] == [42]
    assert captured["max_steps"] == 16
    assert captured["collision_system"] == "soft"
    assert not (
        {"runtime_provenance", "switcher_weights_path", "main_dir", "map_list_sha256"}
        & captured.keys()
    )


@pytest.mark.parametrize("on_target", ["restart", "nothing"])
def test_progress_output_omits_retired_labels_without_changing_results(
    monkeypatch, capsys, on_target
):
    from concurrent.futures import ThreadPoolExecutor

    row = {
        "algorithm": "ARPE",
        "map_name": "test-map",
        "num_agents": 100,
        "seed": 0,
        "error": None,
        "on_target": on_target,
        "avg_throughput": 1.2,
        "run_time_seconds": 0.4,
        "ISR": 1.0,
        "CSR": 0.5,
        "ep_length": 512,
        "congestion_rate": 0.2,
        "reverse_action_rate": 0.1,
        "learning_ratio": 0.4,
        "planner_ratio": 0.6,
        "caar_action_ratio": 0.4,
        "guided_agent_step_ratio": 0.2,
    }
    monkeypatch.setattr(runner, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(runner, "run_single_experiment", lambda _task: dict(row))
    results, _elapsed = runner.run_experiments([{}], workers=1)
    output = capsys.readouterr().out
    for label in ("caar=", "planner=", "caar_actions=", "guided="):
        assert label not in output
    assert "congestion=20.0%" in output
    assert (
        "throughput=1.2000" if on_target == "restart" else "isr=100.0% csr=50.0%"
    ) in output
    for key, value in row.items():
        assert results[0][key] == value
