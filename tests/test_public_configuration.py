from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agents.arpe import ArpeCandidateArtifact
from learning.config import Environment, Experiment
from agents.switcher_core import switcher_observation_space

ROOT = Path(__file__).resolve().parents[1]


def test_training_recipe_can_use_custom_maps_populations_and_hardware():
    config = yaml.safe_load((ROOT / "learning/train_arpe_final.yaml").read_text())
    config["environment"]["grid_config"].update(
        num_agents=16,
        max_episode_steps=128,
        on_target="finish",
        collision_system="priority",
        map_name="maps/my_maps.yaml",
    )
    config["async_ppo"].update(
        num_workers=32,
        batch_size=8192,
        rollout=64,
        recurrence=64,
        num_envs_per_worker=4,
        num_epochs=2,
        ppo_clip_ratio=0.2,
    )
    config["experiment_settings"].update(learning_rate=0.0003, gamma=0.97)
    config["environment"]["training_num_agents_by_worker"] = [16, 32]
    saved = deepcopy(config)
    experiment = Experiment(**config)
    assert experiment.async_ppo.num_workers == 32
    assert experiment.environment.grid_config.num_agents == 16
    assert config == saved


def test_arpe_configuration_resolves_explicit_candidate_paths(tmp_path):
    config = {
        "weights_path": "weights/arpe",
        "checkpoint_path": "downloaded/arpe.pth",
        "base_weights_path": "weights/base",
        "base_checkpoint_path": "downloaded/base.pth",
    }
    original = config.copy()
    artifact = ArpeCandidateArtifact.from_mapping(config, tmp_path)
    assert artifact.project_root == tmp_path.resolve()
    assert artifact.weights_path == (tmp_path / "weights/arpe").resolve()
    assert artifact.checkpoint_path == (tmp_path / "downloaded/arpe.pth").resolve()
    assert artifact.base_weights_path == (tmp_path / "weights/base").resolve()
    assert artifact.base_checkpoint_path == (tmp_path / "downloaded/base.pth").resolve()
    assert artifact.as_dict() == config
    assert config == original


@pytest.mark.parametrize(
    "context,expected",
    [
        (None, 16),
        ({}, 16),
        ({"worker_index": None}, 16),
        ({"worker_index": 3}, 32),
        (SimpleNamespace(worker_index=2), 16),
    ],
)
def test_worker_population_assignment_keeps_recipe_unchanged(context, expected):
    environment = Environment(training_num_agents_by_worker=[16, 32])
    before = environment.dict()
    worker = environment.for_worker(context)
    assert worker.grid_config.num_agents == expected
    worker.grid_config.seed = 99
    assert environment.dict() == before


@pytest.mark.parametrize("index", [-1, "invalid"])
def test_worker_population_rejects_invalid_index(index):
    environment = Environment(training_num_agents_by_worker=[16, 32])
    with pytest.raises(ValueError, match="worker index"):
        environment.for_worker({"worker_index": index})


def test_worker_without_population_schedule_keeps_default():
    environment = Environment()
    assert environment.for_worker({"worker_index": 42}) is environment


def test_switcher_observation_space_is_independent_and_matches_saved_shapes():
    import numpy as np

    expected = {
        "obs": (3, 11, 11),
        "xy": (2,),
        "target_xy": (2,),
        "caar_action": (5,),
        "aoreplan_action": (5,),
    }
    training = switcher_observation_space()
    inference = switcher_observation_space()
    assert set(training.spaces) == set(expected)
    for key, shape in expected.items():
        assert training[key].shape == shape
        assert training[key].dtype == np.float32
        low, high = (-1024, 1024) if key in {"xy", "target_xy"} else (0, 1)
        assert np.all(training[key].low == low)
        assert np.all(training[key].high == high)
        assert inference[key] == training[key]
    inference["extra"] = inference["xy"]
    assert "extra" not in training.spaces
