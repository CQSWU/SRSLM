from copy import deepcopy
from functools import partial
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from sample_factory.algo.utils.context import global_env_registry

import train
import train_switcher
from pomapf_env.switcher_arpe_env import ArpeSwitcherEnv
from agents.arpe import ArpeCandidateArtifact


def test_training_example_uses_current_candidate_and_wait_only_controller():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "learning/train_switcher.yaml").read_text())
    manifest = json.loads((root / "configs/arpe_final_candidate.json").read_text())
    assert config["candidate_policy"] == manifest
    original = deepcopy(config)
    flat = train_switcher.prepare_switcher_config(config)
    assert config == original
    assert flat.env == train_switcher.ENV_NAME
    assert flat.encoder_custom == "switcher"
    expected = ArpeCandidateArtifact.from_mapping(manifest, root).as_dict()
    assert flat.candidate_policy == expected
    assert flat.full_config["candidate_policy"] == expected
    assert flat.population_curriculum == {"populations": [50, 100, 200], "cycles": 10}
    assert "population_curriculum" not in flat.full_config


def test_population_stages_use_equal_agent_step_budgets():
    stages = train_switcher.population_stages(
        {"populations": [50, 100, 200], "cycles": 10}, 1_000_000_000
    )
    assert [n for n, _ in stages] == [50, 100, 200] * 10
    targets = [0] + [target for _, target in stages]
    budgets = [b - a for a, b in zip(targets, targets[1:])]
    assert max(budgets) - min(budgets) == 1
    assert targets[-1] == 1_000_000_000


@pytest.mark.parametrize("populations,cycles", [([], 1), ([0], 1), ([50], 0)])
def test_invalid_curriculum_is_rejected(populations, cycles):
    with pytest.raises(ValueError):
        train_switcher.population_stages(
            {"populations": populations, "cycles": cycles}, 100
        )


def _curriculum_config(tmp_path):
    return SimpleNamespace(
        train_dir=str(tmp_path),
        experiment="curriculum",
        train_for_env_steps=300,
        population_curriculum={"populations": [50, 100, 200], "cycles": 1},
    )


def test_curriculum_resume_skips_finished_stages_and_keeps_one_run(
    tmp_path, monkeypatch
):
    cfg = _curriculum_config(tmp_path)
    checkpoints = tmp_path / cfg.experiment / "checkpoint_p0"
    checkpoints.mkdir(parents=True)
    (checkpoints / "checkpoint_000001_000000120.pth").touch()
    calls = []

    def launch(command):
        calls.append(command)
        target = int(command[command.index("--train_for_env_steps") + 1])
        (checkpoints / f"checkpoint_000002_{target:09d}.pth").touch()
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(train_switcher.subprocess, "run", launch)
    assert train_switcher.run_curriculum(tmp_path / "recipe.yaml", cfg) == 0
    assert [cmd[cmd.index("--training_population") + 1] for cmd in calls] == [
        "100",
        "200",
    ]
    assert all(cmd[cmd.index("--run_name") + 1] == cfg.experiment for cmd in calls)
    assert train_switcher.run_curriculum(tmp_path / "recipe.yaml", cfg) == 0
    assert len(calls) == 2


def test_curriculum_stops_on_failure_or_incomplete_checkpoint(tmp_path, monkeypatch):
    cfg = _curriculum_config(tmp_path)
    monkeypatch.setattr(
        train_switcher.subprocess, "run", lambda _: SimpleNamespace(returncode=7)
    )
    assert train_switcher.run_curriculum(tmp_path / "recipe.yaml", cfg) == 7
    monkeypatch.setattr(
        train_switcher.subprocess, "run", lambda _: SimpleNamespace(returncode=0)
    )
    with pytest.raises(RuntimeError, match="before the stage target"):
        train_switcher.run_curriculum(tmp_path / "recipe.yaml", cfg)


def test_stage_updates_population_on_resume_without_touching_checkpoint(
    tmp_path, monkeypatch
):
    root = Path(__file__).resolve().parents[1]
    args = [
        "--config_path",
        str(root / "learning/train_switcher.yaml"),
        "--run_name",
        "curriculum",
        "--train_dir",
        str(tmp_path),
        "--train_for_env_steps",
        "200",
        "--training_population",
        "100",
    ]
    saved_path = tmp_path / "curriculum" / "config.json"
    saved_path.parent.mkdir()
    saved_path.write_text(
        json.dumps(
            {
                "train_for_env_steps": 100,
                "full_config": {"environment": {"grid_config": {"num_agents": 50}}},
            }
        )
    )
    checkpoint = saved_path.parent / "checkpoint_p0" / "checkpoint_000001_000000100.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"model-and-optimizer")
    seen = []
    monkeypatch.setattr(train_switcher, "register_switcher_components", lambda: None)
    monkeypatch.setattr(train_switcher, "run_rl", lambda cfg: seen.append(cfg) or 0)
    assert train_switcher.main(args) == 0
    cfg = seen[0]
    saved = json.loads(saved_path.read_text())
    assert saved["full_config"]["environment"]["grid_config"]["num_agents"] == 100
    assert (
        saved["full_config"]["environment"]["training_num_agents_by_worker"]
        == [100] * cfg.num_workers
    )
    assert saved["train_for_env_steps"] == 200
    assert saved["restart_behavior"] == "resume"
    assert saved["use_env_info_cache"] is False
    assert cfg.cli_args["full_config"] == saved["full_config"]
    assert checkpoint.read_bytes() == b"model-and-optimizer"


def test_validate_only_has_no_training_or_resume_writes(tmp_path, monkeypatch, capsys):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(train_switcher, "register_switcher_components", lambda: None)
    monkeypatch.setattr(
        train_switcher, "run_curriculum", lambda *args: pytest.fail("started training")
    )
    monkeypatch.setattr(
        train,
        "_sync_resume_cli_overrides",
        lambda *args: pytest.fail("wrote saved config"),
    )
    assert (
        train_switcher.main(
            [
                "--config_path",
                str(root / "learning/train_switcher.yaml"),
                "--train_dir",
                str(tmp_path),
                "--validate-only",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert len(report["population_stages"]) == 30
    assert [n for n, _ in report["population_stages"][:3]] == [50, 100, 200]
    assert not list(tmp_path.iterdir())


@pytest.fixture
def isolated_registry(monkeypatch):
    registry = global_env_registry()
    original = registry.copy()
    monkeypatch.setattr(train, "_CUSTOM_COMPONENTS_REGISTERED", False)
    try:
        yield registry
    finally:
        registry.clear()
        registry.update(original)


@pytest.mark.parametrize(
    "name",
    [train_switcher.ENV_NAME],
)
def test_generic_switcher_factory_fails_before_configuration_or_loading(
    name, isolated_registry
):
    train.register_custom_components()
    assert isolated_registry[name] is train.create_pogema_env
    with pytest.raises(RuntimeError, match="train_switcher.py"):
        isolated_registry[name](name, cfg=None)


def test_unused_environment_base_is_removed():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "pomapf_env/switcher_env.py").exists()


class _FrozenCandidate:
    def __init__(self, artifact, *, seed, device):
        self.artifact = artifact
        self.reset_count = 0
        self.steps = 0

    def after_reset(self):
        self.reset_count += 1

    def set_grid_config(self, cfg):
        self.grid_config = cfg

    def set_env(self, env):
        self.env = env

    def after_step(self, dones):
        self.steps += 1

    def act(self, observations, *args):
        return [4] * len(observations)


def _configuration(environment_name, collision):
    root = Path(__file__).resolve().parents[1]
    declaration = json.loads((root / "configs/arpe_final_candidate.json").read_text())
    return SimpleNamespace(
        full_config={
            "candidate_policy": declaration,
            "environment": {
                "name": environment_name,
                "switcher_caar_device": "cpu",
                "switcher_team_reward_coefficient": 1.0,
                "grid_config": {
                    "map": ".......\n.......\n.......\n.......\n.......\n.......\n.......",
                    "num_agents": 2,
                    "seed": 0,
                    "obs_radius": 5,
                    "on_target": "restart",
                    "collision_system": collision,
                    "max_episode_steps": 3,
                },
            },
        }
    )


@pytest.mark.parametrize("collision", ["block_both", "soft"])
def test_dedicated_registry_constructs_real_env_and_preserves_runtime(
    collision,
    isolated_registry,
    monkeypatch,
):

    env_type = ArpeSwitcherEnv
    monkeypatch.setattr(
        train_switcher,
        "ArpeSwitcherEnv",
        partial(env_type, candidate_factory=_FrozenCandidate),
    )
    train.register_custom_components()
    assert isolated_registry[train_switcher.ENV_NAME] is train.create_pogema_env
    train_switcher.register_switcher_components()
    factory = isolated_registry[train_switcher.ENV_NAME]
    assert factory is not train.create_pogema_env
    cfg = _configuration(train_switcher.ENV_NAME, collision)
    with pytest.raises(ValueError, match="cannot construct"):
        factory("POMAPF-v0", cfg=cfg)
    missing = SimpleNamespace(full_config=deepcopy(cfg.full_config))
    missing.full_config.pop("candidate_policy")
    with pytest.raises(RuntimeError, match="no candidate_policy paths"):
        factory(train_switcher.ENV_NAME, cfg=missing)

    env = factory(train_switcher.ENV_NAME, cfg=cfg, env_config={"worker_index": 0})
    assert type(env) is env_type
    assert env.candidate_artifact.as_dict() == env.candidate.artifact.as_dict()
    rewards_seen = []
    base_step = env.base_env.step

    def record_base_step(actions):
        result = base_step(actions)
        rewards_seen.append(np.asarray(result[1], dtype=np.float32))
        return result

    monkeypatch.setattr(env.base_env, "step", record_base_step)
    try:
        with pytest.raises(RuntimeError, match="reset"):
            env.step([0, 1])
        observations, infos = env.reset(seed=0)
        assert len(observations) == len(infos) == 2
        for index in range(8):
            for observation in observations:
                assert env.observation_space.contains(observation)
            allowed = tuple(env._prepared.switch_allowed_mask)
            expected = tuple(a != 0 for a in env._prepared.aoreplan_actions)
            assert allowed == expected
            observations, rewards, terminated, truncated, infos = env.step(
                [index % 2, (index + 1) % 2]
            )
            raw = rewards_seen[-1]
            np.testing.assert_allclose(rewards, raw + raw.mean(), rtol=0, atol=0)
            assert (
                len(observations)
                == len(rewards)
                == len(terminated)
                == len(truncated)
                == len(infos)
                == 2
            )
            assert np.all(np.isfinite(rewards))

        assert env.candidate.steps == 8
        assert env.candidate.reset_count >= 4
    finally:
        env.close()
