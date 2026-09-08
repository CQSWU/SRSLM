"""Exercise the actual public episode functions without loading models or a GPU."""
import ast
import contextlib
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "run_experiments.py"


def load_function(name, namespace):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace[name]


def episode_fixture(steps, on_target="restart", metric_offset=0.0, fail_at=None):
    clock = [0.0]
    events = []
    envs = []

    class Env:
        def __init__(self, grid_config):
            self.grid_config = grid_config
            self.grid = SimpleNamespace(
                positions_xy=np.array([[2, 2], [2, 3]]), obstacles=np.zeros((5, 5))
            )
            self.unwrapped = self
            self.steps = self.solved = self.resets = 0
            self.closed = False

        def reset(self):
            clock[0] += 1000.0
            self.resets += 1
            events.append("reset")
            return ([{}, {}], {})

        def step(self, actions):
            assert actions == [0, 0]
            clock[0] += 100.0  # Not policy decision time.
            self.steps += 1
            events.append("env_step")
            goals = [self.steps % 3 == 0, self.steps % 5 == 0]
            self.solved += sum(goals)
            if on_target == "restart":
                self.was_on_goal = goals
            done = self.steps == steps
            metrics = {"untouched_metric": 0.25}
            if on_target == "restart":
                metrics["avg_throughput"] = self.solved / steps + metric_offset
            else:
                metrics["CSR"] = 1.0
            infos = [{"metrics": metrics}, {}] if done else [{}, {}]
            # Deliberately unrelated rewards: completion must use was_on_goal.
            return ([{}, {}], [11.0, -7.0], [False, False], [done, done], infos)

        def close(self):
            self.closed = True

    class Algo:
        def after_reset(self):
            events.append("algo_reset")
            clock[0] += 500.0

        def set_grid_config(self, cfg):
            events.append("set_config")

        def set_env(self, env):
            events.append("set_env")
            self.env = env

        def after_step(self, dones):
            events.append("after_step")
            clock[0] += 7.0

        def act(self, *args):
            events.append("act")
            if fail_at is not None and self.env.steps + 1 == fail_at:
                raise RuntimeError("intentional act failure")
            clock[0] += 0.125
            return [0, 0]

    class Tracker:
        def __init__(self, *args, **kwargs):
            pass

        def capture(self, *args, **kwargs):
            clock[0] += 13.0

        def commit(self, *args, **kwargs):
            clock[0] += 17.0

        def metrics(self):
            return {"untouched_tracker_metric": 3}

    class Holder:
        def __init__(self):
            self.data = {}

        def after_step(self, infos):
            clock[0] += 19.0
            self.data.update(infos[0].get("metrics", {}))

        def get_final(self):
            return self.data

    def make_env(grid_config, **kwargs):
        assert kwargs == {"with_animations": False, "auto_reset": False}
        env = Env(grid_config)
        envs.append(env)
        return env

    namespace = dict(
        np=np, POMAPFConfig=lambda **kw: SimpleNamespace(MOVES=[(0, 0)], **kw),
        make_pomapf=make_env, _MoveFailureTracker=Tracker, ResultsHolder=Holder,
        time=SimpleNamespace(perf_counter=lambda: clock[0], time=lambda: clock[0]),
        torch=SimpleNamespace(no_grad=contextlib.nullcontext, manual_seed=lambda _: None),
    )
    run = load_function("run_algorithm", namespace)
    arguments = dict(map_name="fixture", max_episode_steps=steps, seed=42, num_agents=2,
                     obs_radius=5, animate=False, on_target=on_target,
                     collision_system="soft", map_text=".....\n.....\n.....")
    return run, Algo(), arguments, envs, events, namespace


@pytest.mark.parametrize("steps", [128, 512, 600, 1025, 4096])
def test_actual_windows_include_short_tail_and_only_time_act(steps):
    run, algo, args, envs, events, _ = episode_fixture(steps)
    result = run(algo, **args)
    assert result["policy_decision_calls"] == steps
    assert result["policy_decision_seconds"] == steps * 0.125
    assert result["policy_decision_ms_per_joint_action"] == 125.0
    assert result["policy_decision_timing_scope"] == "algo.act_only_perf_counter_v1"
    total = steps // 3 + steps // 5
    assert result["completed_targets_observed"] == total
    assert result["avg_throughput"] == total / steps
    segments = result["throughput_segments"]
    assert len(segments) == (steps + 511) // 512
    for i, segment in enumerate(segments):
        start, end = i * 512 + 1, min((i + 1) * 512, steps)
        count = end // 3 - (start - 1) // 3 + end // 5 - (start - 1) // 5
        assert segment == {
            "start_step": start, "end_step": end, "step_count": end - start + 1,
            "completed_targets": count, "throughput": count / (end - start + 1),
            "cumulative_throughput": (end // 3 + end // 5) / end,
        }
    assert sum(s["completed_targets"] for s in segments) == total
    assert result["untouched_metric"] == 0.25 and result["untouched_tracker_metric"] == 3
    assert envs[0].resets == 1 and envs[0].closed
    assert events == ["reset", "algo_reset", "set_config", "set_env"] + ["act", "env_step", "after_step"] * steps


def test_non_lifelong_has_no_fake_goal_count_or_windows():
    run, algo, args, envs, _, _ = episode_fixture(128, on_target="finish")
    result = run(algo, **args)
    assert result["CSR"] == 1.0 and result["policy_decision_calls"] == 128
    assert result["throughput_segments"] == []
    assert result["completed_targets_observed"] is None
    assert not hasattr(envs[0], "was_on_goal") and envs[0].closed


@pytest.mark.parametrize("offset", [0.01, float("nan")])
def test_goal_counts_must_match_existing_pogema_metric(offset):
    run, algo, args, envs, _, _ = episode_fixture(512, metric_offset=offset)
    with pytest.raises(RuntimeError, match="Segment goal events disagree"):
        run(algo, **args)
    assert envs[0].closed


@pytest.mark.parametrize("failure", ["build", "act"])
def test_actual_outer_failure_has_empty_metrics_and_no_scope_error(failure):
    _, algo, args, envs, _, namespace = episode_fixture(128, fail_at=4)

    def build(*args, **kwargs):
        if failure == "build":
            raise RuntimeError("intentional build failure")
        return algo

    namespace.update(quiet_model_logs=lambda: None, canonical_algorithm_name=lambda s: s,
                     random=random, Path=Path, json=json,
                     should_cache_algorithm=lambda *args: False, build_algorithm=build)
    run = load_function("run_single_experiment", namespace)
    task = {**args, "algorithm": "EPOM-Lifelong-FT", "main_dir": ".", "max_steps": 128}
    result = run(task)
    assert result["error"] == f"intentional {failure} failure"
    assert "NameError" not in result["traceback"]
    assert result["policy_decision_seconds"] is None
    assert result["policy_decision_calls"] == 0
    assert result["policy_decision_ms_per_joint_action"] is None
    assert result["throughput_segments"] == []
    assert result["completed_targets_observed"] is None
    assert all(env.closed for env in envs)


def test_metadata_distinguishes_call_wallclock_from_hardware_isolation():
    metadata = load_function("runtime_metric_metadata", {})()
    assert metadata["version"] == "end_to_end_episode_wall_v1"
    assert "does not establish CPU isolation" in metadata["policy_decision_seconds"]
    assert "no extra accelerator synchronization" in metadata["policy_decision_seconds"]
    assert metadata["throughput_segments"]["event"] == "PogemaLifeLong.was_on_goal_after_each_env_step"
    assert metadata["throughput_segments"]["window_steps"] == 512


def test_outer_result_api_retains_nested_windows_and_existing_metrics():
    _, algo, args, _, _, namespace = episode_fixture(600)
    namespace.update(quiet_model_logs=lambda: None, canonical_algorithm_name=lambda s: s,
                     random=random, Path=Path, json=json,
                     should_cache_algorithm=lambda *args: False,
                     build_algorithm=lambda *args, **kwargs: algo)
    run = load_function("run_single_experiment", namespace)
    result = run({**args, "algorithm": "EPOM-Lifelong-FT", "main_dir": ".", "max_steps": 600})
    assert "error" not in result
    assert result["algorithm"] == "EPOM-Lifelong-FT" and result["seed"] == 42
    assert result["untouched_metric"] == 0.25 and result["untouched_tracker_metric"] == 3
    assert [s["step_count"] for s in result["throughput_segments"]] == [512, 88]
    assert result["run_time_seconds"] > result["policy_decision_seconds"]
    assert json.loads(json.dumps(result))["throughput_segments"] == result["throughput_segments"]
