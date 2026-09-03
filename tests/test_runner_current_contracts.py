import inspect
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from run_experiments import (
    ALGORITHM_ALIASES,
    SUPPORTED_ALGORITHMS,
    _find_caar_weights,
    _find_no_wait_detect_switcher_weights,
    _find_switcher_weights,
    build_algorithm,
    parse_algorithms,
    run_single_experiment,
    static_astar_metric_metadata,
    validate_srslm_stats,
    validate_srslm_ablation_stats,
)


def valid_srslm_stats():
    return {
        "hybrid_mode": "aoreplan_wait_bypass_switcher_v3",
        "switch_pair": ["CAAR", "AORePlan"],
        "switcher_training": "PPO",
        "value_predictor_loaded": False,
        "switcher_feature_schema": "srslm_switcher_state_v3",
        "selector_kind": "ppo_two_branch_categorical",
        "switcher_decision_scope": "aoreplan_nonwait_only",
        "joint_conflict_prediction_enabled": False,
        "total_action_count": 10,
        "switcher_choice_count": 7,
        "switcher_model_choice_count": 7,
        "selected_ao_count": 4,
        "switcher_model_selected_ao_count": 4,
        "executed_ao_count": 4,
        "executed_caar_count": 6,
        "aoreplan_wait_bypass_count": 3,
        "branch_action_agreement_count": 2,
        "static_astar_query_count": 1,
        "aoreplan_commit_count": 4,
        "switcher_checkpoint_sha256": "a" * 64,
        "switcher_stochastic": True,
        "switcher_choice_rate": 0.7,
        "selected_ao_rate": 4 / 7,
        "executed_ao_rate": 0.4,
        "aoreplan_wait_bypass_rate": 0.3,
        "branch_action_agreement_rate": 0.2,
        "switcher_sampled_ao_rate": 4 / 7,
        "switcher_ao_probability_mean": 0.5,
        "switcher_ao_probability_p05": 0.1,
        "switcher_ao_probability_p95": 0.9,
    }


def test_only_current_public_names_are_supported():
    assert SUPPORTED_ALGORITHMS == (
        "RePlan",
        "AORePlan",
        "NoReweight",
        "Direct",
        "CAAR",
        "SRSLM-NoWaitDetect",
        "SRSLM-WaitDetectOnly",
        "SRSLM",
        "EPOM-Lifelong-FT",
    )
    assert set(ALGORITHM_ALIASES.values()) == set(SUPPORTED_ALGORITHMS)
    for retired in (
        "Replan",
        "AO-RePlan",
        "NoTau",
        "CAAR-RS",
        "CAAR-RG",
        "CAAR-Yield",
        "CAAR-PB",
        "CAAR-RA",
        "SRSLM-PPO",
        "SRSLM-PPO-Only",
        "DCC",
        "DHC",
        "MATS-LP",
        "SCRIMP",
        "Follower",
        "CHS-Reconstructed",
        "AS",
        "EPOM-Direct",
        "SRSLM-v8b",
    ):
        assert retired not in SUPPORTED_ALGORITHMS
    for retired_alias in (
        "ao-replan",
        "notau",
        "caar-rs",
        "caar-rg",
        "caar-yield",
        "caar-pb",
        "caar-ra",
        "srslm-ppo",
        "srslm-ppo-only",
    ):
        assert retired_alias not in ALGORITHM_ALIASES
    assert ALGORITHM_ALIASES["replan"] == "RePlan"
    assert not {
        "CAAR-RG",
        "CAAR-Yield",
        "CAAR-PB",
        "CAAR-RA",
    }.intersection(parse_algorithms("all"))


def test_builder_exposes_only_current_artifact_inputs():
    assert tuple(inspect.signature(build_algorithm).parameters) == (
        "algo_name",
        "main_dir",
        "seed",
        "caar_weights_path",
        "switcher_weights_path",
        "no_wait_detect_switcher_weights_path",
        "no_reweight_weights_path",
        "epom_weights_path",
    )


def test_runner_source_excludes_retired_adapter_surface():
    source = (
        Path(__file__).resolve().parents[1] / "run_experiments.py"
    ).read_text(encoding="utf-8")
    for retired in (
        "DCC",
        "DHC",
        "MATS-LP",
        "SCRIMP",
        "Follower",
        "CHS-Reconstructed",
        "AssistantSwitcher",
        "EPOM-Direct",
        "Direct-0P",
        "SRSLM-v8b",
    ):
        assert retired not in source


def test_switcher_default_resolution_is_wait_aware_release():
    with patch("run_experiments._find_weight_run_dir", return_value=None) as finder:
        try:
            _find_switcher_weights(".")
        except FileNotFoundError as exc:
            assert "wait-aware" in str(exc)
        else:
            raise AssertionError("missing wait-aware Switcher silently fell back")
    searched = str(finder.call_args.args[0]).replace("\\", "/")
    assert searched.endswith("weights/SRSLM-switcher-wait-aware-caar-100m")


def test_caar_default_resolution_is_current_trace_branch():
    with patch("run_experiments._find_weight_run_dir", return_value=None) as finder:
        try:
            _find_caar_weights(".")
        except FileNotFoundError as exc:
            assert "current CAAR" in str(exc)
        else:
            raise AssertionError("missing current CAAR silently fell back")
    searched = str(finder.call_args.args[0]).replace("\\", "/")
    assert searched.endswith(
        "weights/EPOM-TracePaperConvDirectCorrection-R5-500m"
    )


def test_no_wait_detect_has_an_independent_weight_root():
    with patch("run_experiments._find_weight_run_dir", return_value=None) as finder:
        try:
            _find_no_wait_detect_switcher_weights(".")
        except FileNotFoundError as exc:
            assert "independently trained" in str(exc)
        else:
            raise AssertionError("missing all-state checkpoint silently fell back")
    searched = str(finder.call_args.args[0]).replace("\\", "/")
    assert searched.endswith("weights/SRSLM-switcher-caar-nowait-100m")
    assert "wait-aware" not in searched


def test_srslm_validator_accepts_only_wait_bypass_contract():
    stats = valid_srslm_stats()
    validate_srslm_stats(stats)
    broken = dict(stats, switcher_choice_count=8)
    try:
        validate_srslm_stats(broken)
    except RuntimeError as exc:
        assert "do not sum" in str(exc)
    else:
        raise AssertionError("invalid SRSLM routing was accepted")


def test_srslm_wait_ablation_validators_enforce_distinct_scopes():
    all_state = dict(
        valid_srslm_stats(),
        hybrid_mode="all_state_switcher_v3",
        ablation_name="SRSLM-NoWaitDetect",
        switcher_decision_scope="all_states",
        wait_detection_enabled=False,
        learned_switcher_called=True,
        total_action_count=10,
        switcher_choice_count=10,
        switcher_model_choice_count=10,
        selected_ao_count=4,
        switcher_model_selected_ao_count=4,
        executed_ao_count=4,
        executed_caar_count=6,
        aoreplan_wait_bypass_count=0,
    )
    validate_srslm_ablation_stats("SRSLM-NoWaitDetect", all_state)

    wait_only = dict(
        all_state,
        hybrid_mode="aoreplan_wait_detect_only_v3",
        ablation_name="SRSLM-WaitDetectOnly",
        switcher_training="none",
        selector_kind="deterministic_wait_detect_only",
        switcher_decision_scope="none",
        wait_detection_enabled=True,
        learned_switcher_called=False,
        switcher_choice_count=0,
        switcher_model_choice_count=0,
        selected_ao_count=0,
        switcher_model_selected_ao_count=0,
        executed_ao_count=7,
        executed_caar_count=3,
        aoreplan_wait_bypass_count=3,
        switcher_stochastic=False,
        switcher_choice_rate=0.0,
        selected_ao_rate=0.0,
        executed_ao_rate=0.7,
        aoreplan_wait_bypass_rate=0.3,
    )
    validate_srslm_ablation_stats("SRSLM-WaitDetectOnly", wait_only)

    broken = dict(all_state, switcher_choice_count=7)
    try:
        validate_srslm_ablation_stats("SRSLM-NoWaitDetect", broken)
    except RuntimeError as exc:
        assert "all-state" in str(exc)
    else:
        raise AssertionError("all-state actor masking was accepted")


def test_srslm_builder_uses_current_classes_and_names():
    caar_module = types.ModuleType("agents.caar")
    switcher_module = types.ModuleType("agents.switcher")
    srslm_module = types.ModuleType("agents.srslm")

    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Algorithm:
        def __init__(self, cfg, **kwargs):
            self.cfg = cfg

    caar_module.CAARConfig = Config
    switcher_module.SwitcherConfig = Config
    srslm_module.SRSLMConfig = Config
    srslm_module.SRSLM = Algorithm
    with patch.dict(sys.modules, {
        "agents.caar": caar_module,
        "agents.switcher": switcher_module,
        "agents.srslm": srslm_module,
    }):
        algorithm = build_algorithm(
            "SRSLM",
            ".",
            42,
            caar_weights_path="caar",
            switcher_weights_path="switcher",
        )
    # The wait-aware artifact contract binds the CAAR checkpoint inside the
    # Switcher bundle, so SRSLM itself owns only the Switcher path.
    assert not hasattr(algorithm.cfg, "caar")
    assert algorithm.cfg.switcher.path_to_weights.endswith("switcher")
    assert not hasattr(algorithm.cfg, "rule_guard_enabled")
    assert not hasattr(algorithm.cfg, "value_margin")


def test_wait_ablation_builders_use_distinct_artifact_contracts():
    caar_module = types.ModuleType("agents.caar")
    ablation_module = types.ModuleType("agents.srslm_ablation")
    switcher_module = types.ModuleType("agents.switcher")

    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Algorithm:
        def __init__(self, cfg):
            self.cfg = cfg

    caar_module.CAARConfig = Config
    ablation_module.SRSLMNoWaitDetectConfig = Config
    ablation_module.SRSLMWaitDetectOnlyConfig = Config
    ablation_module.SRSLMNoWaitDetect = Algorithm
    ablation_module.SRSLMWaitDetectOnly = Algorithm
    switcher_module.AllStateSwitcherConfig = Config
    with patch.dict(
        sys.modules,
        {
            "agents.caar": caar_module,
            "agents.srslm_ablation": ablation_module,
            "agents.switcher": switcher_module,
        },
    ):
        no_wait = build_algorithm(
            "SRSLM-NoWaitDetect",
            ".",
            42,
            caar_weights_path="caar",
            no_wait_detect_switcher_weights_path="all-state",
        )
        wait_only = build_algorithm(
            "SRSLM-WaitDetectOnly",
            ".",
            42,
            caar_weights_path="caar",
        )

    assert no_wait.cfg.switcher.path_to_weights.endswith("all-state")
    assert wait_only.cfg.caar.path_to_weights.endswith("caar")
    assert not hasattr(wait_only.cfg, "switcher")


def test_no_reweight_builder_uses_current_class_names():
    module = types.ModuleType("agents.caar")

    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Algorithm:
        def __init__(self, cfg):
            self.cfg = cfg

    module.NoReweightConfig = Config
    module.NoReweight = Algorithm
    with patch.dict(sys.modules, {"agents.caar": module}):
        algorithm = build_algorithm(
            "NoReweight",
            ".",
            42,
            no_reweight_weights_path="weights/current",
        )
    assert algorithm.cfg.path_to_weights == "weights/current"
    assert algorithm.cfg.checkpoint_kind == "latest"


def test_aoreplan_result_uses_new_metric_names():
    algorithm = SimpleNamespace(
        reverse_action_rate=0.25,
        reverse_action_count=2,
        reverse_action_denominator=8,
        reverse_metric_version="previous_timestep_position_target_segment_v3",
        static_astar_query_count=3,
        static_astar_query_denominator=10,
        static_astar_query_rate=0.3,
        no_path_fallback_count=4,
    )
    task = {
        "algorithm": "AORePlan",
        "main_dir": ".",
        "seed": 42,
        "cache_algorithms": False,
        "map_name": "test-map",
        "num_agents": 4,
        "max_steps": 512,
        "obs_radius": 5,
        "animate": False,
        "on_target": "restart",
        "collision_system": "block_both",
        "map_text": "....\n....",
    }
    run_result = {
        "algorithm": "AORePlan",
        "avg_throughput": 1.0,
        "environment_step_count_observed": 512,
    }
    with patch("run_experiments.build_algorithm", return_value=algorithm), patch(
        "run_experiments.run_algorithm", return_value=run_result
    ):
        result = run_single_experiment(task)
    assert result["static_astar_query_count"] == 3
    assert result["static_astar_query_denominator"] == 10
    assert result["static_astar_query_rate"] == 0.3
    assert result["no_path_fallback_count"] == 4
    assert "probe_invocation_rate" not in result


def test_static_astar_metadata_matches_reported_fields():
    metadata = static_astar_metric_metadata()
    assert metadata["version"] == "aoreplan_static_astar_query_v3"
    assert "static_astar_query_rate_denominator" in metadata
    assert "no_path_fallback_count" in metadata
