"""Unified lifelong MAPF benchmark runner."""

import argparse

import hashlib

import json

import logging

import multiprocessing

import os

import platform

import random

import time

import urllib.request

from concurrent.futures import ProcessPoolExecutor, as_completed

from contextlib import suppress

from datetime import datetime

from pathlib import Path


import numpy as np
import torch


from agents.utils_agents import ResultsHolder
from pogema.svg_animation.animation_wrapper import (
    AnimationConfig,
    AnimationMonitor,
)
from pomapf_env.env import make_pomapf
from pomapf_env.pomapf_config import POMAPFConfig



DEFAULT_MAPS = {

    "mazes": "mazes-s0_wc8_od55",

    "random": "random-s0_d0.15",

    "sc1": "sc1-AcrosstheCape",

    "street": "street-Berlin_0",

    "wc3": "wc3-Battleground",

}


SUPPORTED_ALGORITHMS = (

    "Replan",

    "AO-RePlan",

    "DHC",

    "DCC",

    "SCRIMP",

    "CAAR",

    "NoTau",

    "CAAR-RG",

    "CAAR-Yield",

    "CAAR-PB",

    "CAAR-RA",

    "CAAR-RS",

    "CAAR-LS",

    "EPOM",

    "AS",

)


# Experimental hybrids have distinct integrity contracts from CAAR-RG. Keep
# them opt-in so the historical default/"all" batch remains unambiguous.
DEFAULT_ALGORITHMS = tuple(
    algorithm
    for algorithm in SUPPORTED_ALGORITHMS
    if algorithm not in (
        "CAAR-Yield",
        "CAAR-PB",
        "CAAR-RA",
        "CAAR-RS",
        "CAAR-LS",
    )
)
HYBRID_STRATEGY_KIND = "hybrid_waypoint_guidance"
HYBRID_MODE = "replan_waypoint_v4"
HYBRID_BASE_ALGORITHM = "CAAR"
HYBRID_ACTION_POLICY = "CAAR"
HYBRID_GUIDE_ALGORITHM = "Replan"
HYBRID_COMPONENTS = {
    "base_algorithm": HYBRID_BASE_ALGORITHM,
    "action_policy": HYBRID_ACTION_POLICY,
    "guide_algorithm": HYBRID_GUIDE_ALGORITHM,
}
HYBRID_DEPLOYMENT_CONTRACT = {
    "trigger": (
        "no_net_goal_progress_over_8_step_window "
        "AND net_displacement<=2 "
        "AND local_visible_agents>=5_for_3_steps"
    ),
    "stall_window_steps": 8,
    "stall_goal_progress_metric": "true_goal_manhattan_net_improvement",
    "stall_max_net_goal_progress": 0,
    "stall_displacement_metric": "window_start_to_end_manhattan",
    "stall_max_net_displacement": 2,
    "congestion_min_agents": 5,
    "visible_agent_count_includes_self": True,
    "congestion_confirm_steps": 3,
    "static_visible_prefix_min_blockers": 2,
    "static_visible_prefix_max_blockers": 3,
    "static_visible_prefix_blocker_metric": "last_reference_blocker_count",
    "static_visible_prefix_blockers_exclude_self": True,
    "waypoint_encoding_confirm_steps": 2,
    "waypoint_encoding_metric": "caar_axis_clamped_goal_cell",
    "waypoint_encoding_must_be_stable": True,
    "route_backoff_steps": 4,
    "waypoint_abort_steps": 8,
    "max_planning_steps": 10000,
    "dynamic_agents_used": True,
    "static_replan_counterfactual_reference": True,
    "agent_induced_detour_required": True,
    "activation_condition": (
        "2<=static_visible_prefix_blockers<=3 "
        "AND dynamic_agent_aware_waypoint_encoding_differs_from_static_reference "
        "AND dynamic_waypoint_goal_encoding_stable_for_2_steps"
    ),
    "counterfactual_reference_algorithm": "ordinary_replan",
    "actual_guide_algorithm": "ordinary_replan",
    "shared_accumulated_static_memory": True,
    "counterfactual_reference_uses_dynamic_agents": False,
    "actual_guide_uses_dynamic_agents": True,
    "path_requirement": "complete_four_connected_to_true_goal",
    "guide_output": "last_contiguous_visible_prefix_point",
    "visibility_metric": "chebyshev_obs_radius",
    "same_goal_encoding_rejected": True,
    "simultaneous_per_agent_guidance": True,
    "temporary_target_scope": "caar_observation_only",
    "environment_goal_mutated": False,
    "action_source": "caar_only",
    "planner_actions_executed": False,
    "ao_replan_used": False,
    "probe_used": False,
}
HYBRID_INTEGRITY_IMPLEMENTATION_FILES = (
    "run_experiments.py",
    "agents/caar.py",
    "agents/caar_rg.py",
    "agents/replan.py",
    "agents/utils_agents.py",
    "learning/caar_actor_critic.py",
    "learning/caar_encoder.py",
    "learning/config.py",
    "learning/encoder.py",
    "planning/replan_algo.py",
    "planning/replan_waypoint.py",
    "planning/planner.cpp",
    "planning/planner.cpython-310-x86_64-linux-gnu.so",
    "pomapf_env/custom_maps.py",
    "pomapf_env/env.py",
    "pomapf_env/pomapf_config.py",
    "pomapf_env/stigmergic.py",
    "pomapf_env/wrappers.py",
    "train.py",
    "uv.lock",
)

YIELD_STRATEGY_KIND = "hybrid_active_yielding"
YIELD_MODE = "ao_safe_yield_pocket_v3"
YIELD_BASE_ALGORITHM = "CAAR"
YIELD_ACTION_POLICY = "CAAR"
YIELD_GUIDE_ALGORITHM = "AO-RePlan+LocalSafePocketBFS"
YIELD_COMPONENTS = {
    "base_algorithm": YIELD_BASE_ALGORITHM,
    "action_policy": YIELD_ACTION_POLICY,
    "diagnostic_algorithm": "AO-RePlan",
    "waypoint_selector": "local_safe_pocket_bfs",
}
YIELD_DEPLOYMENT_CONTRACT = {
    "trigger": (
        "no_net_goal_progress_over_8_step_window "
        "AND net_displacement<=2 "
        "AND local_visible_agents>=5_for_3_steps "
        "AND ao_teammate_block_confirmed"
    ),
    "stall_window_steps": 8,
    "stall_goal_progress_metric": "true_goal_manhattan_net_improvement",
    "stall_max_net_goal_progress": 0,
    "stall_max_net_displacement": 2,
    "congestion_min_agents": 5,
    "visible_agent_count_includes_self": True,
    "congestion_confirm_steps": 3,
    "component_min_agents": 5,
    "component_link_distance": 2,
    "narrow_static_degree_max": 2,
    "ao_replan_role": "shadow_teammate_attribution_and_release_check",
    "ao_replan_uses_real_target": True,
    "ao_teammate_confidence_threshold": 2,
    "direct_block_confirm_steps": 2,
    "all_ao_proposals_cancelled": True,
    "ao_actions_executed": False,
    "ordinary_replan_used": False,
    "pocket_search": "four_connected_local_bfs",
    "pocket_max_path_steps": 3,
    "pocket_min_agent_clearance": 2,
    "pocket_fov_inner_margin": 1,
    "straight_corridor_pockets_rejected": True,
    "forward_goal_half_plane_rejected": True,
    "activation_confirm_steps": 2,
    "simultaneous_different_components": True,
    "global_single_agent_cap": False,
    "same_component_min_graph_distance": 3,
    "reservation_rule": "path_tube_plus_one_cell_halo_disjoint",
    "yield_no_progress_abort_steps": 6,
    "yield_total_timeout_steps": 12,
    "hold_min_steps": 3,
    "post_reach_release_monitoring": True,
    "release_monitoring_requires_yielder_at_pocket": False,
    "release_seed_direction_check": (
        "ao_static_probe_next_cell_unoccupied_and_confidence_zero"
    ),
    "dynamic_static_first_action_agreement_required": False,
    "whole_component_clear_required": False,
    "release_conflict_scope": "original_seed_to_yielder_edges",
    "hold_timeout_steps": 16,
    "release_confirm_steps": 2,
    "backoff_steps": 4,
    "temporary_target_scope": "caar_observation_only",
    "environment_goal_mutated": False,
    "action_source": "caar_only",
    "caar_forward_passes_per_environment_step": 1,
}
YIELD_INTEGRITY_IMPLEMENTATION_FILES = (
    "run_experiments.py",
    "agents/ao_replan.py",
    "agents/caar.py",
    "agents/caar_yield.py",
    "agents/utils_agents.py",
    "learning/caar_actor_critic.py",
    "learning/caar_encoder.py",
    "learning/config.py",
    "learning/encoder.py",
    "planning/ao_replan_algo.py",
    "planning/planner.cpp",
    "planning/planner.cpython-310-x86_64-linux-gnu.so",
    "planning/yield_pocket.py",
    "pomapf_env/custom_maps.py",
    "pomapf_env/env.py",
    "pomapf_env/pomapf_config.py",
    "pomapf_env/stigmergic.py",
    "pomapf_env/wrappers.py",
    "scripts/run_caar_yield_pilot.sh",
    "train.py",
    "uv.lock",
)

PB_STRATEGY_KIND = "hybrid_probe_virtual_obstacles"
PB_MODE = "ao_probe_virtual_block_v2"
PB_BASE_ALGORITHM = "CAAR"
PB_ACTION_POLICY = "CAAR"
PB_GUIDE_ALGORITHM = "AO-RePlan Local Probe"
PB_COMPONENTS = {
    "base_algorithm": PB_BASE_ALGORITHM,
    "action_policy": PB_ACTION_POLICY,
    "diagnostic_algorithm": "AO-RePlan Local Probe",
    "observation_modifier": "persistent_per_agent_virtual_obstacles",
}
PB_DEPLOYMENT_CONTRACT = {
    "trigger": "visible_probe_path_contains_at_least_2_teammates",
    "initial_trigger_requires_stall": False,
    "probe_target": "real_target",
    "probe_ignores_dynamic_agents": True,
    "probe_path_requirement": "complete_four_connected_to_true_goal",
    "probe_visible_prefix_metric": "contiguous_chebyshev_fov_prefix",
    "probe_static_knowledge_scope": "current_local_obstacles_unknown_outside_free",
    "path_blockers_source": "local_agents_observation",
    "path_blockers_exclude_self": True,
    "initial_min_path_blockers": 2,
    "virtual_block_cell": "probe_path_first_move_after_current",
    "candidate_validation": "fresh_stateless_probe_after_tentative_block",
    "candidate_no_path_policy": "rollback_without_commit",
    "persistent_until": (
        "real_target_reached_or_changed_or_agent_done_or_inactive_or_reset"
    ),
    "repeat_trigger": (
        "after_existing_block_and_all_positions_in_fresh_8_step_window_"
        "remain_within_manhattan_radius_2_of_window_anchor AND current_"
        "visible_probe_path_contains_at_least_2_teammates"
    ),
    "repeat_requires_fresh_congestion_evidence": True,
    "repeat_min_path_blockers": 2,
    "repeat_path_blocker_metric": (
        "current_probe_contiguous_visible_prefix_distinct_teammate_cells"
    ),
    "repeat_probe_obstacle_scope": "current_local_obstacles_plus_owned_blocks",
    "repeat_uses_goal_distance": False,
    "repeat_stall_region_metric": (
        "maximum_manhattan_displacement_from_window_anchor"
    ),
    "repeat_stall_window_steps": 8,
    "repeat_stall_max_anchor_displacement": 2,
    "repeat_window_reset_after_attempt": True,
    "repeat_window_consumed_on_probe_no_path": True,
    "repeat_window_consumed_without_congestion": True,
    "maximum_new_blocks_per_attempt": 1,
    "maximum_virtual_blocks_per_real_target": 4,
    "maximum_virtual_blocks_scope": "per_agent_per_real_target_episode",
    "at_block_cap": "stop_appending_until_persistent_state_is_cleared",
    "ordinary_replan_used": False,
    "ao_probe_actions_executed": False,
    "temporary_obstacle_scope": "caar_observation_copy_only",
    "environment_obstacles_mutated": False,
    "environment_goal_mutated": False,
    "action_source": "caar_only",
    "caar_forward_passes_per_environment_step": 1,
}
PB_INTEGRITY_IMPLEMENTATION_FILES = (
    "run_experiments.py",
    "agents/ao_replan.py",
    "agents/caar.py",
    "agents/caar_probe_block.py",
    "agents/utils_agents.py",
    "learning/caar_actor_critic.py",
    "learning/caar_encoder.py",
    "learning/config.py",
    "learning/encoder.py",
    "planning/ao_replan_algo.py",
    "planning/planner.cpp",
    "planning/planner.cpython-310-x86_64-linux-gnu.so",
    "pomapf_env/custom_maps.py",
    "pomapf_env/env.py",
    "pomapf_env/pomapf_config.py",
    "pomapf_env/stigmergic.py",
    "pomapf_env/wrappers.py",
    "scripts/run_caar_probe_block_pilot.sh",
    "train.py",
    "uv.lock",
)

RA_STRATEGY_KIND = "learned_relative_advantage_hybrid"
RA_MODE = "learned_relative_advantage_raw_plan_v1"
RA_BASE_ALGORITHM = "CAAR"
RA_PLAN_ALGORITHM = "Raw AO-RePlan"
RA_ACTION_POLICY = "CAAR-or-Raw-AO-RePlan"
RA_COMPONENTS = {
    "base_algorithm": RA_BASE_ALGORITHM,
    "candidate_algorithm": RA_PLAN_ALGORITHM,
    "selector": "binary_relative_advantage_gate",
}
RA_DEPLOYMENT_CONTRACT = {
    "decision_order": "CAAR_then_raw_AO-RePlan_then_gate",
    "gate_actions": {"0": "CAAR", "1": "raw_plan"},
    "deterministic_logit_margin": 0.0,
    "hard_caar_fallbacks": (
        "raw_plan_none_or_invalid_or_reverse_or_equal_to_caar"
    ),
    "reverse_plan_actions_executed": False,
    "probe_used": False,
    "simultaneous_plan_agents_allowed": True,
    "global_single_agent_cap": False,
    "joint_filter": (
        "reject_only_vertex_or_edge_swap_conflicts_new_relative_to_all_caar"
    ),
    "proposal_feedback": "commit_if_raw_plan_matches_final_physical_action",
}
RA_INTEGRITY_IMPLEMENTATION_FILES = (
    "run_experiments.py",
    "agents/caar.py",
    "agents/caar_ra.py",
    "agents/utils_agents.py",
    "learning/caar_actor_critic.py",
    "learning/caar_encoder.py",
    "learning/caar_plan_gate_env.py",
    "learning/config.py",
    "learning/encoder.py",
    "planning/ao_replan_algo.py",
    "planning/raw_aoreplan_candidates.py",
    "planning/planner.cpp",
    "planning/planner.cpython-310-x86_64-linux-gnu.so",
    "pomapf_env/custom_maps.py",
    "pomapf_env/env.py",
    "pomapf_env/pomapf_config.py",
    "pomapf_env/stigmergic.py",
    "pomapf_env/wrappers.py",
    "train.py",
    "uv.lock",
)

RS_STRATEGY_KIND = "deterministic_rule_only_hybrid"
RS_MODE = "rule_only_valid_nonreverse_raw_plan_else_caar_cooldown4_v1"
RS_BASE_ALGORITHM = "CAAR"
RS_PLAN_ALGORITHM = "Raw AO-RePlan"
RS_ACTION_POLICY = "CAAR-or-Raw-AO-RePlan"
RS_COMPONENTS = {
    "base_algorithm": RS_BASE_ALGORITHM,
    "candidate_algorithm": RS_PLAN_ALGORITHM,
    "selector": "valid_nonreverse_raw_plan_else_caar_cooldown4",
}
RS_DEPLOYMENT_CONTRACT = {
    "selector_kind": "deterministic_rule_only",
    "default_action_source": "raw_AO-RePlan",
    "hard_caar_fallbacks": "raw_plan_none_or_invalid_or_reverse",
    "value_predictor_loaded": False,
    "learned_gate_loaded": False,
    "reverse_plan_actions_executed": False,
    "cooldown_steps": 4,
    "cooldown_scope": "per_agent",
    "cooldown_includes_trigger_step": True,
    "probe_used": False,
    "simultaneous_plan_agents_allowed": True,
    "global_single_agent_cap": False,
    "joint_filter": "none",
    "proposal_feedback": (
        "commit_if_valid_nonreverse_raw_plan_matches_final_physical_action"
    ),
}
RS_INTEGRITY_IMPLEMENTATION_FILES = (
    "run_experiments.py",
    "agents/caar.py",
    "agents/caar_rule_switch.py",
    "agents/utils_agents.py",
    "learning/caar_actor_critic.py",
    "learning/caar_encoder.py",
    "learning/config.py",
    "learning/encoder.py",
    "learning/grid_memory.py",
    "planning/ao_replan_algo.py",
    "planning/raw_aoreplan_candidates.py",
    "planning/planner.cpp",
    "planning/planner.cpython-310-x86_64-linux-gnu.so",
    "pomapf_env/custom_maps.py",
    "pomapf_env/env.py",
    "pomapf_env/pomapf_config.py",
    "pomapf_env/stigmergic.py",
    "pomapf_env/wrappers.py",
    "train.py",
    "uv.lock",
)

LS_STRATEGY_KIND = "learned_absolute_return_hybrid"
LS_MODE = "per_step_absolute_return_lswitcher_reverse_to_caar_v1"
LS_REVERSE_COOLDOWN_MODE = (
    "per_step_absolute_return_lswitcher_reverse_to_caar_cooldown_v1"
)
LS_PREDICTOR_ONLY_MODE = (
    "per_step_absolute_return_lswitcher_predictor_only_v1"
)
LS_BASE_ALGORITHM = "CAAR"
LS_PLAN_ALGORITHM = "Raw AO-RePlan"
LS_ACTION_POLICY = "CAAR-or-Raw-AO-RePlan"
LS_COMPONENTS = {
    "base_algorithm": LS_BASE_ALGORITHM,
    "candidate_algorithm": LS_PLAN_ALGORITHM,
    "selector": (
        "two_independent_absolute_mc_return_estimators_with_shared_trace"
    ),
}
LS_DEPLOYMENT_CONTRACT = {
    "value_targets": "separate_absolute_raw_monte_carlo_returns",
    "estimator_input": "matrix_observation_plus_shared_traffic_trace",
    "selector": "V_AO_gt_V_CAAR_plus_margin",
    "tie_policy": "CAAR",
    "nonfinite_policy": "CAAR",
    "comparison_cadence": "every_step_per_agent",
    "switch_constraint": "none",
    "hard_caar_fallbacks": "raw_plan_none_or_invalid_or_reverse",
    "safety_override_changes_nominal_selection": False,
    "reverse_plan_actions_executed": False,
    "probe_used": False,
    "simultaneous_plan_agents_allowed": True,
    "global_single_agent_cap": False,
    "proposal_feedback": (
        "commit_if_valid_nonreverse_raw_plan_matches_final_physical_action"
    ),
}


def _ls_mode(
    reverse_caar_cooldown_steps=4,
    reverse_caar_override_enabled=True,
):
    if not bool(reverse_caar_override_enabled):
        return LS_PREDICTOR_ONLY_MODE
    return (
        LS_REVERSE_COOLDOWN_MODE
        if int(reverse_caar_cooldown_steps) > 0
        else LS_MODE
    )


def _ls_switch_constraint(reverse_caar_cooldown_steps=4):
    return (
        "reverse_caar_cooldown"
        if int(reverse_caar_cooldown_steps) > 0
        else "none"
    )
LS_INTEGRITY_IMPLEMENTATION_FILES = (
    "run_experiments.py",
    "agents/caar.py",
    "agents/caar_lswitcher.py",
    "agents/utils_agents.py",
    "learning/encoder.py",
    "learning/epom_encoder.py",
    "planning/ao_replan_algo.py",
    "planning/raw_aoreplan_candidates.py",
    "planning/planner.cpp",
    "planning/planner.cpython-310-x86_64-linux-gnu.so",
    "policy_estimation/model.py",
    "pomapf_env/env.py",
    "pomapf_env/pomapf_config.py",
    "pomapf_env/stigmergic.py",
    "pomapf_env/wrappers.py",
    "uv.lock",
)


ALGORITHM_ALIASES = {

    "replan": "Replan",

    "ao-replan": "AO-RePlan",

    "dhc": "DHC",

    "dcc": "DCC",

    "scrimp": "SCRIMP",

    "caar": "CAAR",

    "notau": "NoTau",

    "caar-rg": "CAAR-RG",

    "caarrg": "CAAR-RG",

    "caar_rg": "CAAR-RG",

    "caar-yield": "CAAR-Yield",

    "caaryield": "CAAR-Yield",

    "caar_yield": "CAAR-Yield",

    "caar-pb": "CAAR-PB",

    "caarpb": "CAAR-PB",

    "caar_pb": "CAAR-PB",

    "caar-ra": "CAAR-RA",

    "caarra": "CAAR-RA",

    "caar_ra": "CAAR-RA",

    "caar-rs": "CAAR-RS",

    "caarrs": "CAAR-RS",

    "caar_rs": "CAAR-RS",

    "caar-rule": "CAAR-RS",

    "caar-ls": "CAAR-LS",

    "caarls": "CAAR-LS",

    "caar_ls": "CAAR-LS",

    "epom": "EPOM",

    "as": "AS",

}


ALGORITHM_COLUMN_WIDTH = max(13, *(len(algorithm) for algorithm in SUPPORTED_ALGORITHMS))


_worker_algo_cache = {}



def canonical_algorithm_name(value):

    return ALGORITHM_ALIASES.get(value.strip().lower())


def hybrid_contract_metadata(algorithms):
    """Describe CAAR actions guided by ordinary-Replan path waypoints."""
    if "CAAR-RG" not in algorithms:
        return None
    return {
        "strategy_kind": HYBRID_STRATEGY_KIND,
        "hybrid_mode": HYBRID_MODE,
        "base_algorithm": HYBRID_BASE_ALGORITHM,
        "action_policy": HYBRID_ACTION_POLICY,
        "guide_algorithm": HYBRID_GUIDE_ALGORITHM,
        "hybrid_components": dict(HYBRID_COMPONENTS),
        "deployment": dict(HYBRID_DEPLOYMENT_CONTRACT),
    }


def yield_contract_metadata(algorithms):
    """Describe AO-assisted active yielding with CAAR-only actions."""
    if "CAAR-Yield" not in algorithms:
        return None
    return {
        "strategy_kind": YIELD_STRATEGY_KIND,
        "hybrid_mode": YIELD_MODE,
        "base_algorithm": YIELD_BASE_ALGORITHM,
        "action_policy": YIELD_ACTION_POLICY,
        "guide_algorithm": YIELD_GUIDE_ALGORITHM,
        "hybrid_components": dict(YIELD_COMPONENTS),
        "deployment": dict(YIELD_DEPLOYMENT_CONTRACT),
    }


def pb_contract_metadata(algorithms):
    """Describe Probe-guided virtual obstacles with CAAR-only actions."""
    if "CAAR-PB" not in algorithms:
        return None
    return {
        "strategy_kind": PB_STRATEGY_KIND,
        "hybrid_mode": PB_MODE,
        "base_algorithm": PB_BASE_ALGORITHM,
        "action_policy": PB_ACTION_POLICY,
        "guide_algorithm": PB_GUIDE_ALGORITHM,
        "hybrid_components": dict(PB_COMPONENTS),
        "deployment": dict(PB_DEPLOYMENT_CONTRACT),
    }


def ra_contract_metadata(algorithms):
    """Describe the learned CAAR/raw-Plan relative-advantage policy."""
    if "CAAR-RA" not in algorithms:
        return None
    return {
        "strategy_kind": RA_STRATEGY_KIND,
        "hybrid_mode": RA_MODE,
        "base_algorithm": RA_BASE_ALGORITHM,
        "action_policy": RA_ACTION_POLICY,
        "guide_algorithm": RA_PLAN_ALGORITHM,
        "hybrid_components": dict(RA_COMPONENTS),
        "deployment": dict(RA_DEPLOYMENT_CONTRACT),
    }


def rs_contract_metadata(algorithms):
    """Describe the deterministic CAAR/raw-Plan rule-only policy."""
    if "CAAR-RS" not in algorithms:
        return None
    return {
        "strategy_kind": RS_STRATEGY_KIND,
        "hybrid_mode": RS_MODE,
        "base_algorithm": RS_BASE_ALGORITHM,
        "action_policy": RS_ACTION_POLICY,
        "guide_algorithm": RS_PLAN_ALGORITHM,
        "hybrid_components": dict(RS_COMPONENTS),
        "deployment": dict(RS_DEPLOYMENT_CONTRACT),
    }


def ls_contract_metadata(
    algorithms,
    *,
    value_margin=0.0,
    reverse_caar_cooldown_steps=None,
    reverse_caar_override_enabled=True,
    road_topology_adaptive_cooldown_enabled=False,
    road_open4_threshold=0.68,
    road_dense_obstacle_threshold=0.70,
    road_reverse_caar_cooldown_steps=8,
    road_caar_only_density_threshold=None,
):
    """Describe the per-step absolute-return learnable switcher."""
    if "CAAR-LS" not in algorithms:
        return None
    reverse_override_enabled = bool(reverse_caar_override_enabled)
    cooldown_steps = int(
        4 if reverse_caar_cooldown_steps is None and reverse_override_enabled
        else 0 if reverse_caar_cooldown_steps is None
        else reverse_caar_cooldown_steps
    )
    road_adaptive = bool(road_topology_adaptive_cooldown_enabled)
    road_cooldown_steps = int(road_reverse_caar_cooldown_steps)
    road_open4_threshold = float(road_open4_threshold)
    road_dense_obstacle_threshold = float(
        road_dense_obstacle_threshold
    )
    road_density_threshold = (
        None
        if road_caar_only_density_threshold is None
        else float(road_caar_only_density_threshold)
    )
    for label, threshold in (
        ("road_open4_threshold", road_open4_threshold),
        ("road_dense_obstacle_threshold", road_dense_obstacle_threshold),
    ):
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(f"{label} must be finite and within [0, 1]")
    if road_cooldown_steps < 0:
        raise ValueError("road reverse CAAR cooldown must be non-negative")
    if road_density_threshold is not None and (
        not np.isfinite(road_density_threshold)
        or road_density_threshold < 0.0
    ):
        raise ValueError("road CAAR-only density threshold is invalid")
    if cooldown_steps > 0 and not reverse_override_enabled:
        raise ValueError(
            "reverse CAAR cooldown requires the reverse override"
        )
    if road_adaptive and road_cooldown_steps > 0 and not reverse_override_enabled:
        raise ValueError(
            "road reverse CAAR cooldown requires the reverse override"
        )
    deployment = dict(LS_DEPLOYMENT_CONTRACT)
    deployment["value_margin"] = float(value_margin)
    deployment["switch_constraint"] = _ls_switch_constraint(cooldown_steps)
    deployment["reverse_caar_cooldown_steps"] = cooldown_steps
    deployment["reverse_caar_cooldown_includes_trigger_step"] = True
    deployment["reverse_caar_override_enabled"] = reverse_override_enabled
    deployment["road_topology_adaptive_cooldown_enabled"] = road_adaptive
    deployment["road_open4_threshold"] = road_open4_threshold
    deployment["road_dense_obstacle_threshold"] = (
        road_dense_obstacle_threshold
    )
    deployment["road_reverse_caar_cooldown_steps"] = road_cooldown_steps
    deployment["road_caar_only_density_threshold"] = (
        road_density_threshold
    )
    deployment["road_topology_source"] = "grid_config.map"
    deployment["road_topology_uses_map_name"] = False
    deployment["road_agent_density_definition"] = "num_agents/free_cells"
    deployment["hard_caar_fallbacks"] = (
        "raw_plan_none_or_invalid_or_reverse"
        if reverse_override_enabled
        else "raw_plan_none_or_invalid"
    )
    if road_density_threshold is not None:
        deployment["hard_caar_fallbacks"] += "_or_road_density_gate"
    deployment["reverse_plan_actions_executed"] = not reverse_override_enabled
    deployment["safety_override_changes_nominal_selection"] = bool(
        cooldown_steps or road_adaptive or road_density_threshold is not None
    )
    return {
        "strategy_kind": LS_STRATEGY_KIND,
        "hybrid_mode": _ls_mode(
            cooldown_steps,
            reverse_override_enabled,
        ),
        "base_algorithm": LS_BASE_ALGORITHM,
        "action_policy": LS_ACTION_POLICY,
        "guide_algorithm": LS_PLAN_ALGORITHM,
        "hybrid_components": dict(LS_COMPONENTS),
        "deployment": deployment,
    }


def switch_contract_metadata(algorithms):
    """Compatibility alias for the CAAR/Replan hybrid contract."""
    return hybrid_contract_metadata(algorithms)



def quiet_model_logs():

    logging.getLogger("rl").setLevel(logging.ERROR)

    with suppress(Exception):

        from sample_factory.utils.utils import log


        log.setLevel(logging.ERROR)



def _has_config(path):

    return (path / "config.json").exists()



def _has_checkpoints(path):

    for d in path.glob("checkpoint_p*"):

        if d.is_dir() and any(d.glob("*.pth")):

            return True

    return False



def _find_caar_weights(main_dir):
    root = Path(main_dir).resolve()
    candidate = _find_weight_run_dir(root / "weights" / "CAAR")
    if candidate is not None:
        return str(candidate)
    return str(root / "weights" / "CAAR" / "CAAR")


def _find_notau_weights(main_dir):
    root = Path(main_dir).resolve()
    candidate = _find_weight_run_dir(root / "weights" / "NoTau")
    if candidate is not None:
        return str(candidate)
    return str(root / "weights" / "NoTau" / "NoTau")


def _find_caar_ra_weights(main_dir):
    root = Path(main_dir).resolve()
    candidate = _find_weight_run_dir(root / "weights" / "CAAR-RA")
    if candidate is not None:
        return str(candidate)
    return str(root / "weights" / "CAAR-RA" / "CAAR-RA")


def _find_caar_ls_caar_estimator_checkpoint(main_dir):
    root = Path(main_dir).resolve()
    return str(root / "weights" / "CAAR-LS" / "caar_estimator.pth")


def _find_caar_ls_ao_estimator_checkpoint(main_dir):
    root = Path(main_dir).resolve()
    return str(root / "weights" / "CAAR-LS" / "ao_estimator.pth")


def _find_dhc_weights(main_dir):
    root = Path(main_dir).resolve()
    return str(root / "otherpolicy" / "DHC" / "models" / "337500.pth")


def _find_dcc_weights(main_dir):
    root = Path(main_dir).resolve()
    return str(root / "otherpolicy" / "DCC" / "saved_models" / "128000.pth")


def _find_scrimp_weights(main_dir):
    root = Path(main_dir).resolve()
    return str(root / "otherpolicy" / "SCRIMP" / "final" / "net_checkpoint.pkl")


def _find_epom_weights(main_dir):
    root = Path(main_dir).resolve()
    candidate = root / "weights" / "EPOM" / "EPOM"
    return str(candidate)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_provenance():
    """Record host/runtime details without using them as result identity."""
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "accelerator_visible_devices": os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        ),
        "ppu_sdk_version": os.environ.get("PPU_SDK_VERSION"),
    }


def _project_path(main_dir, value):
    path = Path(value)
    if not path.is_absolute():
        path = Path(main_dir) / path
    return path.resolve()


def cache_algorithm_metadata(algorithms, requested):
    """Describe requested and effective per-algorithm instance caching."""
    requested = bool(requested)
    effective_by_algorithm = {
        algorithm: requested and algorithm not in ("CAAR-LS", "CAAR-RS")
        for algorithm in algorithms
    }
    exceptions = {}
    if requested and "CAAR-LS" in effective_by_algorithm:
        exceptions["CAAR-LS"] = (
            "disabled_to_preserve_episode_fresh_caar_normalization"
        )
    if requested and "CAAR-RS" in effective_by_algorithm:
        exceptions["CAAR-RS"] = (
            "disabled_to_preserve_episode_fresh_caar_normalization"
        )
    return {
        "requested": requested,
        "effective_by_algorithm": effective_by_algorithm,
        "exceptions": exceptions,
    }


def _latest_caar_checkpoint(weights_path):
    checkpoint_dir = Path(weights_path) / "checkpoint_p0"
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pth"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No latest CAAR checkpoint under {checkpoint_dir}."
        )
    return checkpoints[-1].resolve()


def _latest_gate_checkpoint(weights_path):
    checkpoint_dir = Path(weights_path) / "checkpoint_p0"
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_*.pth"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No latest CAAR-RA gate checkpoint under {checkpoint_dir}."
        )
    return checkpoints[-1].resolve()


def hybrid_integrity_metadata(
    args,
    map_list_sha256=None,
    map_registry_sha256=None,
):
    """Hash the exact CAAR/Replan waypoint hybrid used by an evaluation."""

    root = Path(args.main_dir).resolve()
    code_root = Path(__file__).resolve().parent
    caar_weights = _project_path(
        root,
        args.caar_weights_path or _find_caar_weights(root),
    )
    caar_checkpoint = _latest_caar_checkpoint(caar_weights)
    caar_config = caar_weights / "config.json"
    implementation = {
        relative_path: code_root / relative_path
        for relative_path in HYBRID_INTEGRITY_IMPLEMENTATION_FILES
    }
    required_files = {
        "caar_checkpoint": caar_checkpoint,
        "caar_config": caar_config,
        **implementation,
    }
    if args.map_list:
        required_files["map_list"] = _project_path(root, args.map_list)
    missing = [
        f"{label}={path}"
        for label, path in required_files.items()
        if not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Cannot create CAAR-RG integrity metadata; missing "
            + ", ".join(missing)
        )
    return {
        "strategy_kind": HYBRID_STRATEGY_KIND,
        "hybrid_mode": HYBRID_MODE,
        "hybrid_components": dict(HYBRID_COMPONENTS),
        "hybrid_action_policy": HYBRID_ACTION_POLICY,
        "hybrid_guide_algorithm": HYBRID_GUIDE_ALGORITHM,
        "caar_checkpoint_path": str(caar_checkpoint),
        "caar_checkpoint_sha256": _sha256_file(caar_checkpoint),
        "caar_config_path": str(caar_config.resolve()),
        "caar_config_sha256": _sha256_file(caar_config),
        "implementation_sha256": {
            label: _sha256_file(path)
            for label, path in implementation.items()
        },
        "map_list_sha256": (
            map_list_sha256
            if map_list_sha256 is not None
            else (
                _sha256_file(required_files["map_list"])
                if "map_list" in required_files
                else None
            )
        ),
        "map_registry_sha256": (
            map_registry_sha256
            if map_registry_sha256 is not None
            else (
                _sha256_file(required_files["map_list"])
                if "map_list" in required_files
                else None
            )
        ),
    }


def yield_integrity_metadata(
    args,
    map_list_sha256=None,
    map_registry_sha256=None,
):
    """Hash the exact AO-assisted safe-yield policy and CAAR checkpoint."""

    root = Path(args.main_dir).resolve()
    code_root = Path(__file__).resolve().parent
    caar_weights = _project_path(
        root,
        args.caar_weights_path or _find_caar_weights(root),
    )
    caar_checkpoint = _latest_caar_checkpoint(caar_weights)
    caar_config = caar_weights / "config.json"
    implementation = {
        relative_path: code_root / relative_path
        for relative_path in YIELD_INTEGRITY_IMPLEMENTATION_FILES
    }
    required_files = {
        "caar_checkpoint": caar_checkpoint,
        "caar_config": caar_config,
        **implementation,
    }
    if args.map_list:
        required_files["map_list"] = _project_path(root, args.map_list)
    missing = [
        f"{label}={path}"
        for label, path in required_files.items()
        if not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Cannot create CAAR-Yield integrity metadata; missing "
            + ", ".join(missing)
        )
    return {
        "strategy_kind": YIELD_STRATEGY_KIND,
        "hybrid_mode": YIELD_MODE,
        "hybrid_components": dict(YIELD_COMPONENTS),
        "hybrid_action_policy": YIELD_ACTION_POLICY,
        "hybrid_guide_algorithm": YIELD_GUIDE_ALGORITHM,
        "caar_checkpoint_path": str(caar_checkpoint),
        "caar_checkpoint_sha256": _sha256_file(caar_checkpoint),
        "caar_config_path": str(caar_config.resolve()),
        "caar_config_sha256": _sha256_file(caar_config),
        "implementation_sha256": {
            label: _sha256_file(path)
            for label, path in implementation.items()
        },
        "map_list_sha256": (
            map_list_sha256
            if map_list_sha256 is not None
            else (
                _sha256_file(required_files["map_list"])
                if "map_list" in required_files
                else None
            )
        ),
        "map_registry_sha256": map_registry_sha256,
    }


def pb_integrity_metadata(
    args,
    map_list_sha256=None,
    map_registry_sha256=None,
):
    """Hash the exact Probe-block policy and CAAR checkpoint."""

    root = Path(args.main_dir).resolve()
    code_root = Path(__file__).resolve().parent
    caar_weights = _project_path(
        root,
        args.caar_weights_path or _find_caar_weights(root),
    )
    caar_checkpoint = _latest_caar_checkpoint(caar_weights)
    caar_config = caar_weights / "config.json"
    implementation = {
        relative_path: code_root / relative_path
        for relative_path in PB_INTEGRITY_IMPLEMENTATION_FILES
    }
    required_files = {
        "caar_checkpoint": caar_checkpoint,
        "caar_config": caar_config,
        **implementation,
    }
    if args.map_list:
        required_files["map_list"] = _project_path(root, args.map_list)
    missing = [
        f"{label}={path}"
        for label, path in required_files.items()
        if not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Cannot create CAAR-PB integrity metadata; missing "
            + ", ".join(missing)
        )
    return {
        "strategy_kind": PB_STRATEGY_KIND,
        "hybrid_mode": PB_MODE,
        "hybrid_components": dict(PB_COMPONENTS),
        "hybrid_action_policy": PB_ACTION_POLICY,
        "hybrid_guide_algorithm": PB_GUIDE_ALGORITHM,
        "caar_checkpoint_path": str(caar_checkpoint),
        "caar_checkpoint_sha256": _sha256_file(caar_checkpoint),
        "caar_config_path": str(caar_config.resolve()),
        "caar_config_sha256": _sha256_file(caar_config),
        "implementation_sha256": {
            label: _sha256_file(path)
            for label, path in implementation.items()
        },
        "map_list_sha256": (
            map_list_sha256
            if map_list_sha256 is not None
            else (
                _sha256_file(required_files["map_list"])
                if "map_list" in required_files
                else None
            )
        ),
        "map_registry_sha256": map_registry_sha256,
    }


def ra_integrity_metadata(
    args,
    map_list_sha256=None,
    map_registry_sha256=None,
):
    """Hash both frozen policies and all CAAR-RA decision-path sources."""

    root = Path(args.main_dir).resolve()
    code_root = Path(__file__).resolve().parent
    caar_weights = _project_path(
        root,
        args.caar_weights_path or _find_caar_weights(root),
    )
    gate_weights = _project_path(
        root,
        args.caar_ra_weights_path or _find_caar_ra_weights(root),
    )
    caar_checkpoint = _latest_caar_checkpoint(caar_weights)
    gate_checkpoint = _latest_gate_checkpoint(gate_weights)
    caar_config = caar_weights / "config.json"
    gate_config = gate_weights / "config.json"
    implementation = {
        relative_path: code_root / relative_path
        for relative_path in RA_INTEGRITY_IMPLEMENTATION_FILES
    }
    required_files = {
        "caar_checkpoint": caar_checkpoint,
        "caar_config": caar_config,
        "gate_checkpoint": gate_checkpoint,
        "gate_config": gate_config,
        **implementation,
    }
    if getattr(args, "map_list", None):
        required_files["map_list"] = _project_path(root, args.map_list)
    missing = [
        f"{label}={path}"
        for label, path in required_files.items()
        if not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Cannot create CAAR-RA integrity metadata; missing "
            + ", ".join(missing)
        )

    implementation_hashes = {
        label: _sha256_file(path)
        for label, path in implementation.items()
    }
    artifact_hashes = {
        "caar_checkpoint": _sha256_file(caar_checkpoint),
        "caar_config": _sha256_file(caar_config),
        "gate_checkpoint": _sha256_file(gate_checkpoint),
        "gate_config": _sha256_file(gate_config),
        **{
            f"implementation:{label}": digest
            for label, digest in implementation_hashes.items()
        },
    }
    aggregate_sha256 = hashlib.sha256(
        json.dumps(
            artifact_hashes,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "strategy_kind": RA_STRATEGY_KIND,
        "hybrid_mode": RA_MODE,
        "hybrid_components": dict(RA_COMPONENTS),
        "hybrid_action_policy": RA_ACTION_POLICY,
        "hybrid_guide_algorithm": RA_PLAN_ALGORITHM,
        "caar_checkpoint_path": str(caar_checkpoint),
        "caar_checkpoint_sha256": artifact_hashes["caar_checkpoint"],
        "caar_config_path": str(caar_config.resolve()),
        "caar_config_sha256": artifact_hashes["caar_config"],
        "gate_checkpoint_path": str(gate_checkpoint),
        "gate_checkpoint_sha256": artifact_hashes["gate_checkpoint"],
        "gate_config_path": str(gate_config.resolve()),
        "gate_config_sha256": artifact_hashes["gate_config"],
        "implementation_sha256": implementation_hashes,
        "aggregate_sha256": aggregate_sha256,
        "map_list_sha256": (
            map_list_sha256
            if map_list_sha256 is not None
            else (
                _sha256_file(required_files["map_list"])
                if "map_list" in required_files
                else None
            )
        ),
        "map_registry_sha256": map_registry_sha256,
    }


def rs_integrity_metadata(
    args,
    map_list_sha256=None,
    map_registry_sha256=None,
):
    """Hash CAAR and the complete deterministic rule-only action path."""

    root = Path(args.main_dir).resolve()
    code_root = Path(__file__).resolve().parent
    caar_weights = _project_path(
        root,
        args.caar_weights_path or _find_caar_weights(root),
    )
    caar_checkpoint = _latest_caar_checkpoint(caar_weights)
    caar_config = caar_weights / "config.json"
    implementation = {
        relative_path: code_root / relative_path
        for relative_path in RS_INTEGRITY_IMPLEMENTATION_FILES
    }
    required_files = {
        "caar_checkpoint": caar_checkpoint,
        "caar_config": caar_config,
        **implementation,
    }
    if getattr(args, "map_list", None):
        required_files["map_list"] = _project_path(root, args.map_list)
    missing = [
        f"{label}={path}"
        for label, path in required_files.items()
        if not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Cannot create CAAR-RS integrity metadata; missing "
            + ", ".join(missing)
        )

    implementation_hashes = {
        label: _sha256_file(path)
        for label, path in implementation.items()
    }
    artifact_hashes = {
        "caar_checkpoint": _sha256_file(caar_checkpoint),
        "caar_config": _sha256_file(caar_config),
        **{
            f"implementation:{label}": digest
            for label, digest in implementation_hashes.items()
        },
    }
    aggregate_sha256 = hashlib.sha256(
        json.dumps(
            artifact_hashes,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "strategy_kind": RS_STRATEGY_KIND,
        "hybrid_mode": RS_MODE,
        "hybrid_components": dict(RS_COMPONENTS),
        "hybrid_action_policy": RS_ACTION_POLICY,
        "hybrid_guide_algorithm": RS_PLAN_ALGORITHM,
        "caar_checkpoint_path": str(caar_checkpoint),
        "caar_checkpoint_sha256": artifact_hashes["caar_checkpoint"],
        "caar_config_path": str(caar_config.resolve()),
        "caar_config_sha256": artifact_hashes["caar_config"],
        "implementation_sha256": implementation_hashes,
        "aggregate_sha256": aggregate_sha256,
        "map_list_sha256": (
            map_list_sha256
            if map_list_sha256 is not None
            else (
                _sha256_file(required_files["map_list"])
                if "map_list" in required_files
                else None
            )
        ),
        "map_registry_sha256": map_registry_sha256,
    }


def ls_integrity_metadata(
    args,
    map_list_sha256=None,
    map_registry_sha256=None,
):
    """Hash both absolute-return estimators and the complete LS action path."""

    root = Path(args.main_dir).resolve()
    cooldown_steps = int(
        getattr(args, "caar_ls_reverse_caar_cooldown_steps", 0)
    )
    reverse_override_enabled = bool(
        getattr(args, "caar_ls_reverse_caar_override_enabled", True)
    )
    road_adaptive = bool(
        getattr(
            args,
            "caar_ls_road_topology_adaptive_cooldown_enabled",
            False,
        )
    )
    road_open4_threshold = float(
        getattr(args, "caar_ls_road_open4_threshold", 0.68)
    )
    road_dense_obstacle_threshold = float(
        getattr(args, "caar_ls_road_dense_obstacle_threshold", 0.70)
    )
    road_cooldown_steps = int(
        getattr(args, "caar_ls_road_reverse_caar_cooldown_steps", 8)
    )
    road_density_threshold = getattr(
        args,
        "caar_ls_road_caar_only_density_threshold",
        None,
    )
    if road_density_threshold is not None:
        road_density_threshold = float(road_density_threshold)
    code_root = Path(__file__).resolve().parent
    caar_weights = _project_path(
        root,
        args.caar_weights_path or _find_caar_weights(root),
    )
    caar_estimator_checkpoint = _project_path(
        root,
        args.caar_ls_caar_estimator_checkpoint
        or _find_caar_ls_caar_estimator_checkpoint(root),
    )
    ao_estimator_checkpoint = _project_path(
        root,
        args.caar_ls_ao_estimator_checkpoint
        or _find_caar_ls_ao_estimator_checkpoint(root),
    )
    caar_checkpoint = _latest_caar_checkpoint(caar_weights)
    caar_config = caar_weights / "config.json"
    implementation = {
        relative_path: code_root / relative_path
        for relative_path in LS_INTEGRITY_IMPLEMENTATION_FILES
    }
    required_files = {
        "caar_checkpoint": caar_checkpoint,
        "caar_config": caar_config,
        "caar_estimator_checkpoint": caar_estimator_checkpoint,
        "ao_estimator_checkpoint": ao_estimator_checkpoint,
        **implementation,
    }
    if getattr(args, "map_list", None):
        required_files["map_list"] = _project_path(root, args.map_list)
    missing = [
        f"{label}={path}"
        for label, path in required_files.items()
        if not Path(path).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Cannot create CAAR-LS integrity metadata; missing "
            + ", ".join(missing)
        )

    implementation_hashes = {
        label: _sha256_file(path)
        for label, path in implementation.items()
    }
    artifact_hashes = {
        "caar_checkpoint": _sha256_file(caar_checkpoint),
        "caar_config": _sha256_file(caar_config),
        "caar_estimator_checkpoint": _sha256_file(
            caar_estimator_checkpoint
        ),
        "ao_estimator_checkpoint": _sha256_file(
            ao_estimator_checkpoint
        ),
        **{
            f"implementation:{label}": digest
            for label, digest in implementation_hashes.items()
        },
    }
    deployment_identity = {
        "hybrid_mode": _ls_mode(
            cooldown_steps,
            reverse_override_enabled,
        ),
        "value_margin": float(args.caar_ls_value_margin),
        "reverse_caar_override_enabled": reverse_override_enabled,
        "reverse_caar_cooldown_steps": cooldown_steps,
        "road_topology_adaptive_cooldown_enabled": road_adaptive,
        "road_open4_threshold": road_open4_threshold,
        "road_dense_obstacle_threshold": road_dense_obstacle_threshold,
        "road_reverse_caar_cooldown_steps": road_cooldown_steps,
        "road_caar_only_density_threshold": road_density_threshold,
    }
    aggregate_sha256 = hashlib.sha256(
        json.dumps(
            {
                "artifacts": artifact_hashes,
                "deployment": deployment_identity,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "strategy_kind": LS_STRATEGY_KIND,
        "hybrid_mode": _ls_mode(
            cooldown_steps,
            reverse_override_enabled,
        ),
        "hybrid_components": dict(LS_COMPONENTS),
        "hybrid_action_policy": LS_ACTION_POLICY,
        "hybrid_guide_algorithm": LS_PLAN_ALGORITHM,
        "comparison_cadence": "every_step_per_agent",
        "switch_constraint": _ls_switch_constraint(cooldown_steps),
        "value_margin": float(args.caar_ls_value_margin),
        "reverse_caar_override_enabled": reverse_override_enabled,
        "reverse_caar_cooldown_steps": cooldown_steps,
        "reverse_caar_cooldown_includes_trigger_step": True,
        "road_topology_adaptive_cooldown_enabled": road_adaptive,
        "road_open4_threshold": road_open4_threshold,
        "road_dense_obstacle_threshold": road_dense_obstacle_threshold,
        "road_reverse_caar_cooldown_steps": road_cooldown_steps,
        "road_caar_only_density_threshold": road_density_threshold,
        "caar_checkpoint_path": str(caar_checkpoint),
        "caar_checkpoint_sha256": artifact_hashes["caar_checkpoint"],
        "caar_config_path": str(caar_config.resolve()),
        "caar_config_sha256": artifact_hashes["caar_config"],
        "caar_estimator_checkpoint_path": str(caar_estimator_checkpoint),
        "caar_estimator_checkpoint_sha256": artifact_hashes[
            "caar_estimator_checkpoint"
        ],
        "ao_estimator_checkpoint_path": str(ao_estimator_checkpoint),
        "ao_estimator_checkpoint_sha256": artifact_hashes[
            "ao_estimator_checkpoint"
        ],
        "implementation_sha256": implementation_hashes,
        "aggregate_sha256": aggregate_sha256,
        "map_list_sha256": (
            map_list_sha256
            if map_list_sha256 is not None
            else (
                _sha256_file(required_files["map_list"])
                if "map_list" in required_files
                else None
            )
        ),
        "map_registry_sha256": map_registry_sha256,
    }


def switch_integrity_metadata(
    args,
    map_list_sha256=None,
    map_registry_sha256=None,
):
    """Compatibility alias for hybrid integrity metadata."""
    return hybrid_integrity_metadata(
        args,
        map_list_sha256=map_list_sha256,
        map_registry_sha256=map_registry_sha256,
    )


def _find_weight_run_dir(root):
    root = Path(root)
    if not root.exists():
        return None
    if _has_config(root) and _has_checkpoints(root):
        return root

    configs = sorted(root.rglob("config.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for config_path in configs:
        candidate = config_path.parent
        if _has_checkpoints(candidate):
            return candidate
    return None



def _caar_cfg(caar_path, seed):

    from agents.caar import CAARConfig


    return CAARConfig(

        path_to_weights=caar_path,

        seed=seed,

        checkpoint_kind="latest",

    )



def _replan_cfg(seed, max_planning_steps=10000):

    from agents.replan import RePlanConfig


    return RePlanConfig(

        name="RePlan",

        fix_loops=True,

        add_none_if_loop=False,

        no_path_random=True,

        use_best_move=True,

        fix_nones=True,

        max_planning_steps=max_planning_steps,

        seed=seed,

    )



def _ao_replan_cfg(

    seed,

    max_planning_steps=10000,

):

    from agents.ao_replan import AORePlanConfig


    return AORePlanConfig(

        name="AORePlan",

        no_path_random=True,

        use_best_move=True,

        fix_nones=True,

        handoff_on_reverse=False,

        max_planning_steps=max_planning_steps,

        seed=seed,

    )


def build_algorithm(

    algo_name,

    main_dir,

    seed,

    caar_weights_path=None,

    caar_ra_weights_path=None,

    caar_ls_caar_estimator_checkpoint=None,

    caar_ls_ao_estimator_checkpoint=None,

    caar_ls_value_margin=0.0,

    caar_ls_reverse_caar_override_enabled=True,

    caar_ls_reverse_caar_cooldown_steps=4,

    caar_ls_road_topology_adaptive_cooldown_enabled=False,

    caar_ls_road_open4_threshold=0.68,

    caar_ls_road_dense_obstacle_threshold=0.70,

    caar_ls_road_reverse_caar_cooldown_steps=8,

    caar_ls_road_caar_only_density_threshold=None,

    notau_weights_path=None,

    dhc_weights_path=None,

    dcc_weights_path=None,

    scrimp_weights_path=None,

    epom_weights_path=None,

):

    algo_name = canonical_algorithm_name(algo_name) or algo_name


    if algo_name == "Replan":

        from agents.replan import RePlan


        return RePlan(_replan_cfg(seed))


    if algo_name == "AO-RePlan":

        from agents.ao_replan import AORePlan


        return AORePlan(_ao_replan_cfg(seed))

    if algo_name == "DHC":

        from agents.dhc import DHC, DHCConfig

        return DHC(

            DHCConfig(

                path_to_weights=dhc_weights_path or _find_dhc_weights(main_dir),

                seed=seed,

            )

        )







    if algo_name == "DCC":

        from agents.dcc import DCC, DCCConfig

        return DCC(

            DCCConfig(

                path_to_weights=dcc_weights_path or _find_dcc_weights(main_dir),

                seed=seed,

            )

        )


    if algo_name == "SCRIMP":

        from agents.scrimp import SCRIMP, SCRIMPConfig

        return SCRIMP(

            SCRIMPConfig(

                path_to_weights=scrimp_weights_path or _find_scrimp_weights(main_dir),

                seed=seed,

            )

        )


    if algo_name == "CAAR-RG":
        from agents.caar import CAARConfig
        from agents.caar_rg import CAARRG, CAARRGConfig

        return CAARRG(
            CAARRGConfig(
                caar=CAARConfig(
                    path_to_weights=(
                        caar_weights_path or _find_caar_weights(main_dir)
                    ),
                    checkpoint_kind="latest",
                ),
                seed=seed,
            )
        )


    if algo_name == "CAAR-Yield":
        from agents.caar import CAARConfig
        from agents.caar_yield import CAARYield, CAARYieldConfig

        return CAARYield(
            CAARYieldConfig(
                caar=CAARConfig(
                    path_to_weights=(
                        caar_weights_path or _find_caar_weights(main_dir)
                    ),
                    checkpoint_kind="latest",
                ),
                seed=seed,
            )
        )


    if algo_name == "CAAR-PB":
        from agents.caar import CAARConfig
        from agents.caar_probe_block import (
            CAARProbeBlock,
            CAARProbeBlockConfig,
        )

        return CAARProbeBlock(
            CAARProbeBlockConfig(
                caar=CAARConfig(
                    path_to_weights=(
                        caar_weights_path or _find_caar_weights(main_dir)
                    ),
                    checkpoint_kind="latest",
                ),
                initial_min_path_blockers=2,
                repeat_stall_window_steps=8,
                repeat_stall_max_anchor_displacement=2,
                max_virtual_blocks_per_target=4,
                seed=seed,
            )
        )


    if algo_name == "CAAR-RA":
        from agents.caar import CAARConfig
        from agents.caar_ra import CAARRA, CAARRAConfig

        return CAARRA(
            CAARRAConfig(
                caar=CAARConfig(
                    path_to_weights=(
                        caar_weights_path or _find_caar_weights(main_dir)
                    ),
                    checkpoint_kind="latest",
                    device="auto",
                ),
                gate_path_to_weights=(
                    caar_ra_weights_path or _find_caar_ra_weights(main_dir)
                ),
                gate_checkpoint_kind="latest",
                gate_device="auto",
                logit_margin=0.0,
                filter_new_conflicts=True,
                seed=seed,
            )
        )


    if algo_name == "CAAR-RS":
        from agents.caar import CAARConfig
        from agents.caar_rule_switch import CAARRS, CAARRSConfig

        return CAARRS(
            CAARRSConfig(
                caar=CAARConfig(
                    path_to_weights=str(
                        _project_path(
                            main_dir,
                            caar_weights_path or _find_caar_weights(main_dir),
                        )
                    ),
                    checkpoint_kind="latest",
                    device="auto",
                ),
                reverse_caar_cooldown_steps=int(
                    caar_ls_reverse_caar_cooldown_steps
                ),
                seed=seed,
            )
        )


    if algo_name == "CAAR-LS":
        from agents.caar import CAARConfig
        from agents.caar_lswitcher import CAARLS, CAARLSConfig

        resolved_caar_weights = str(
            _project_path(
                main_dir,
                caar_weights_path or _find_caar_weights(main_dir),
            )
        )
        resolved_caar_estimator = str(
            _project_path(
                main_dir,
                caar_ls_caar_estimator_checkpoint
                or _find_caar_ls_caar_estimator_checkpoint(main_dir),
            )
        )
        resolved_ao_estimator = str(
            _project_path(
                main_dir,
                caar_ls_ao_estimator_checkpoint
                or _find_caar_ls_ao_estimator_checkpoint(main_dir),
            )
        )

        return CAARLS(
            CAARLSConfig(
                caar=CAARConfig(
                    path_to_weights=resolved_caar_weights,
                    checkpoint_kind="latest",
                    device="auto",
                ),
                caar_estimator_checkpoint_path=resolved_caar_estimator,
                ao_estimator_checkpoint_path=resolved_ao_estimator,
                estimator_device="auto",
                value_margin=float(caar_ls_value_margin),
                reverse_caar_override_enabled=bool(
                    caar_ls_reverse_caar_override_enabled
                ),
                reverse_caar_cooldown_steps=int(
                    caar_ls_reverse_caar_cooldown_steps
                ),
                road_topology_adaptive_cooldown_enabled=bool(
                    caar_ls_road_topology_adaptive_cooldown_enabled
                ),
                road_open4_threshold=float(
                    caar_ls_road_open4_threshold
                ),
                road_dense_obstacle_threshold=float(
                    caar_ls_road_dense_obstacle_threshold
                ),
                road_reverse_caar_cooldown_steps=int(
                    caar_ls_road_reverse_caar_cooldown_steps
                ),
                road_caar_only_density_threshold=(
                    None
                    if caar_ls_road_caar_only_density_threshold is None
                    else float(caar_ls_road_caar_only_density_threshold)
                ),
                seed=seed,
            )
        )



    if algo_name == "EPOM":

        from agents.epom import EPOM, EPOMConfig


        return EPOM(

            EPOMConfig(

                path_to_weights=epom_weights_path or _find_epom_weights(main_dir),

                seed=seed,

                device="auto",

            )

        )


    if algo_name == "AS":

        from agents.assistant_switcher import (

            AssistantSwitcher,

            AssistantSwitcherConfig,

        )

        from agents.epom import EPOMConfig

        from agents.replan import RePlanConfig


        return AssistantSwitcher(

            AssistantSwitcherConfig(

                planning=RePlanConfig(

                    name="RePlan",

                    fix_loops=True,

                    add_none_if_loop=True,

                    no_path_random=False,

                    use_best_move=False,

                    fix_nones=False,

                    seed=seed,

                ),

                learning=EPOMConfig(
                    path_to_weights=epom_weights_path or _find_epom_weights(main_dir),
                    seed=seed,
                    device="auto",
                ),

                seed=seed,

            )

        )


    if algo_name == "NoTau":

        from agents.caar import NoTau, NoTauConfig

        notau_path = notau_weights_path or _find_notau_weights(main_dir)


        return NoTau(

            NoTauConfig(

                path_to_weights=notau_path,

                seed=seed,

                checkpoint_kind="auto" if notau_weights_path else "best",

                device="auto",

            )

        )


    if algo_name == "CAAR":

        caar_path = caar_weights_path or _find_caar_weights(main_dir)

        from agents.caar import CAAR


        return CAAR(

            _caar_cfg(

                caar_path,

                seed,

            )

        )


    raise ValueError(f"Unsupported algorithm: {algo_name}")


def validate_caar_yield_stats(stats):
    """Fail a result if the CAAR-only action-source contract was violated."""

    required = (
        "environment_step_count",
        "caar_forward_pass_count",
        "total_actions",
        "caar_action_count",
        "ao_planned_proposal_count",
        "ao_cancelled_proposal_count",
        "ao_action_execution_count",
        "none_action_count",
        "target_mutation_leak_count",
        "adjacent_yielder_violation_count",
        "pocket_conflict_violation_count",
        "invalid_state_transition_count",
    )
    missing = [key for key in required if key not in stats]
    if missing:
        raise RuntimeError(
            "CAAR-Yield diagnostics are incomplete: " + ", ".join(missing)
        )
    equalities = (
        (
            "one CAAR forward per environment step",
            stats["caar_forward_pass_count"],
            stats["environment_step_count"],
        ),
        (
            "all environment actions come from CAAR",
            stats["caar_action_count"],
            stats["total_actions"],
        ),
        (
            "all AO proposals are cancelled",
            stats["ao_cancelled_proposal_count"],
            stats["ao_planned_proposal_count"],
        ),
    )
    violations = [
        f"{label}: {actual} != {expected}"
        for label, actual, expected in equalities
        if actual != expected
    ]
    zero_fields = (
        "ao_action_execution_count",
        "none_action_count",
        "target_mutation_leak_count",
        "adjacent_yielder_violation_count",
        "pocket_conflict_violation_count",
        "invalid_state_transition_count",
    )
    violations.extend(
        f"{key}: {stats[key]} != 0"
        for key in zero_fields
        if stats[key] != 0
    )
    if violations:
        raise RuntimeError(
            "CAAR-Yield safety contract failed: " + "; ".join(violations)
        )


def validate_caar_pb_stats(stats):
    """Fail a result if CAAR-PB mutates raw inputs or executes another policy."""

    required = (
        "hybrid_mode",
        "environment_step_count",
        "caar_forward_pass_count",
        "total_actions",
        "caar_action_count",
        "ao_action_execution_count",
        "ordinary_replan_action_count",
        "none_action_count",
        "target_mutation_leak_count",
        "obstacle_mutation_leak_count",
        "agent_mutation_leak_count",
        "initial_congestion_trigger_count",
        "repeat_stall_trigger_count",
        "repeat_window_reset_count",
        "repeat_congestion_confirmed_count",
        "repeat_stall_without_congestion_count",
        "repeat_probe_no_path_count",
        "initial_virtual_block_count",
        "repeat_virtual_block_count",
        "virtual_block_commit_count",
        "max_virtual_blocks_per_agent",
        "virtual_block_cap_reached_count",
        "repeat_min_path_blockers_configured",
        "repeat_stall_window_steps_configured",
        "repeat_stall_radius_configured",
        "max_virtual_blocks_configured",
    )
    missing = [key for key in required if key not in stats]
    if missing:
        raise RuntimeError(
            "CAAR-PB diagnostics are incomplete: " + ", ".join(missing)
        )
    violations = []
    if stats["hybrid_mode"] != PB_MODE:
        violations.append("hybrid mode does not match the v2 contract")
    if stats["caar_forward_pass_count"] != stats["environment_step_count"]:
        violations.append("CAAR forward count differs from environment steps")
    if stats["caar_action_count"] != stats["total_actions"]:
        violations.append("not all environment actions came from CAAR")
    for key in (
        "ao_action_execution_count",
        "ordinary_replan_action_count",
        "none_action_count",
        "target_mutation_leak_count",
        "obstacle_mutation_leak_count",
        "agent_mutation_leak_count",
    ):
        if stats[key] != 0:
            violations.append(f"{key}: {stats[key]} != 0")
    classified_repeat_count = (
        stats["repeat_congestion_confirmed_count"]
        + stats["repeat_stall_without_congestion_count"]
        + stats["repeat_probe_no_path_count"]
    )
    if classified_repeat_count != stats["repeat_stall_trigger_count"]:
        violations.append(
            "repeat attempts are not fully classified and window-consumed"
        )
    if stats["repeat_window_reset_count"] != stats["repeat_stall_trigger_count"]:
        violations.append("not every repeat attempt consumed its 8-step window")
    if (
        stats["repeat_virtual_block_count"]
        > stats["repeat_congestion_confirmed_count"]
    ):
        violations.append(
            "repeat commits exceed freshly confirmed congestion attempts"
        )
    if stats["virtual_block_commit_count"] != (
        stats["initial_virtual_block_count"]
        + stats["repeat_virtual_block_count"]
    ):
        violations.append("virtual block commit accounting is inconsistent")
    attempt_count = (
        stats["initial_congestion_trigger_count"]
        + stats["repeat_congestion_confirmed_count"]
    )
    if stats["virtual_block_commit_count"] > attempt_count:
        violations.append(
            "virtual block commits exceed initial plus repeat attempts"
        )
    if stats["max_virtual_blocks_per_agent"] > 4:
        violations.append("per-target virtual block cap exceeded 4")
    configured = (
        stats["repeat_min_path_blockers_configured"],
        stats["repeat_stall_window_steps_configured"],
        stats["repeat_stall_radius_configured"],
        stats["max_virtual_blocks_configured"],
    )
    if configured != (2, 8, 2, 4):
        violations.append(
            "configured repeat blockers/window/radius/block cap differ "
            "from 2/8/2/4"
        )
    if violations:
        raise RuntimeError(
            "CAAR-PB safety contract failed: " + "; ".join(violations)
        )


def validate_caar_ra_stats(stats):
    """Validate hard safety and accounting invariants of learned CAAR-RA."""

    required = (
        "hybrid_mode",
        "environment_step_count",
        "total_action_count",
        "total_actions",
        "caar_action_count",
        "reverse_count",
        "none_count",
        "plan_requested_count",
        "eligible_plan_requested_count",
        "plan_executed_count",
        "reverse_executed_count",
        "final_none_action_count",
        "conflict_rejected_count",
        "max_concurrent_plan",
    )
    missing = [key for key in required if key not in stats]
    if missing:
        raise RuntimeError(
            "CAAR-RA diagnostics are incomplete: " + ", ".join(missing)
        )

    violations = []
    if stats["hybrid_mode"] != RA_MODE:
        violations.append("hybrid mode does not match the CAAR-RA contract")
    total = stats["total_action_count"]
    if total != stats["total_actions"]:
        violations.append("total action aliases disagree")
    if stats["plan_executed_count"] + stats["caar_action_count"] != total:
        violations.append("final actions are not fully classified")
    if stats["reverse_executed_count"] != 0:
        violations.append("a reverse raw Plan proposal was executed")
    if stats["final_none_action_count"] != 0:
        violations.append("a final environment action was None")
    if not (
        0
        <= stats["eligible_plan_requested_count"]
        <= stats["plan_requested_count"]
        <= total
    ):
        violations.append("Plan request accounting is inconsistent")
    if stats["plan_executed_count"] + stats["conflict_rejected_count"] != (
        stats["eligible_plan_requested_count"]
    ):
        violations.append("eligible Plan requests are not fully classified")
    if not 0 <= stats["reverse_count"] <= total:
        violations.append("reverse proposal count is outside action bounds")
    if not 0 <= stats["none_count"] <= total:
        violations.append("None proposal count is outside action bounds")
    if not 0 <= stats["max_concurrent_plan"] <= stats["plan_executed_count"]:
        violations.append("maximum concurrent Plan count is inconsistent")
    if violations:
        raise RuntimeError(
            "CAAR-RA safety contract failed: " + "; ".join(violations)
        )


def validate_caar_rs_stats(stats):
    """Validate deterministic rule-only selection and safety accounting."""

    required = (
        "hybrid_mode",
        "selector_kind",
        "value_predictor_loaded",
        "value_predictor_call_count",
        "value_comparison_count",
        "reverse_caar_cooldown_steps",
        "reverse_caar_cooldown_includes_trigger_step",
        "reverse_caar_cooldown_trigger_count",
        "reverse_caar_cooldown_action_count",
        "reverse_caar_cooldown_followup_action_count",
        "max_reverse_caar_cooldown_remaining",
        "total_action_count",
        "total_actions",
        "nominal_caar_count",
        "nominal_ao_count",
        "executed_caar_count",
        "executed_ao_count",
        "forced_caar_count",
        "reverse_count",
        "reverse_override_count",
        "reverse_ao_executed_count",
        "probe_call_count",
        "final_none_action_count",
    )
    missing = [key for key in required if key not in stats]
    if missing:
        raise RuntimeError(
            "CAAR-RS diagnostics are incomplete: " + ", ".join(missing)
        )

    violations = []
    total = stats["total_action_count"]
    if stats["hybrid_mode"] != RS_MODE:
        violations.append("hybrid mode does not match the CAAR-RS contract")
    if stats["selector_kind"] != "deterministic_rule_only":
        violations.append("selector is not deterministic rule-only")
    if stats["value_predictor_loaded"] is not False:
        violations.append("a value predictor was loaded")
    if stats["value_predictor_call_count"] != 0:
        violations.append("a value predictor was called")
    if stats["value_comparison_count"] != 0:
        violations.append("a learned value comparison was recorded")
    if stats["reverse_caar_cooldown_steps"] != 4:
        violations.append("the required four-step cooldown is not enabled")
    if stats["reverse_caar_cooldown_includes_trigger_step"] is not True:
        violations.append("cooldown trigger-step semantics are invalid")
    if total != stats["total_actions"]:
        violations.append("total action aliases disagree")
    if stats["nominal_caar_count"] != 0 or stats["nominal_ao_count"] != total:
        violations.append("raw Plan is not the default nominal source")
    if stats["executed_caar_count"] + stats["executed_ao_count"] != total:
        violations.append("final actions are not fully classified")
    if stats["forced_caar_count"] != stats["executed_caar_count"]:
        violations.append("rule fallback accounting is inconsistent")
    if stats["reverse_override_count"] != stats["reverse_count"]:
        violations.append("reverse rule accounting is inconsistent")
    if stats["reverse_ao_executed_count"] != 0:
        violations.append("a reverse raw Plan proposal executed")
    if stats["probe_call_count"] != 0:
        violations.append("Probe was called")
    if stats["final_none_action_count"] != 0:
        violations.append("a final environment action was None")
    triggers = stats["reverse_caar_cooldown_trigger_count"]
    cooldown_actions = stats["reverse_caar_cooldown_action_count"]
    cooldown_followups = stats[
        "reverse_caar_cooldown_followup_action_count"
    ]
    if cooldown_actions != triggers + cooldown_followups:
        violations.append("cooldown action accounting is inconsistent")
    if not 0 <= triggers <= stats["reverse_count"]:
        violations.append("cooldown trigger count is invalid")
    if not 0 <= cooldown_actions <= triggers * 4:
        violations.append("cooldown action count exceeds its bound")
    if not 0 <= cooldown_followups <= triggers * 3:
        violations.append("cooldown followup count exceeds its bound")
    if not 0 <= stats["max_reverse_caar_cooldown_remaining"] <= 3:
        violations.append("maximum cooldown state is invalid")
    for key in (
        "executed_caar_count",
        "executed_ao_count",
        "forced_caar_count",
        "reverse_count",
        "reverse_override_count",
    ):
        if not 0 <= stats[key] <= total:
            violations.append(f"{key} is outside total-action bounds")
    if violations:
        raise RuntimeError(
            "CAAR-RS safety contract failed: " + "; ".join(violations)
        )


def validate_caar_ls_stats(stats):
    """Validate CAAR-LS safety, provenance, and accounting invariants."""

    required = (
        "hybrid_mode",
        "comparison_cadence",
        "switch_constraint",
        "environment_step_count",
        "total_action_count",
        "total_actions",
        "value_comparison_count",
        "branch_switch_count",
        "nominal_caar_count",
        "nominal_ao_count",
        "executed_caar_count",
        "executed_ao_count",
        "reverse_count",
        "reverse_override_count",
        "reverse_ao_executed_count",
        "none_count",
        "none_override_count",
        "invalid_plan_count",
        "invalid_override_count",
        "nonfinite_value_count",
        "nonfinite_caar_value_count",
        "nonfinite_ao_value_count",
        "forced_caar_count",
        "reverse_caar_override_enabled",
        "reverse_caar_cooldown_steps",
        "reverse_caar_cooldown_includes_trigger_step",
        "reverse_caar_cooldown_trigger_count",
        "reverse_caar_cooldown_action_count",
        "reverse_caar_cooldown_followup_action_count",
        "max_reverse_caar_cooldown_remaining",
        "probe_call_count",
        "final_none_action_count",
        "max_concurrent_nominal_ao",
        "max_concurrent_ao_executed",
    )
    missing = [key for key in required if key not in stats]
    if missing:
        raise RuntimeError(
            "CAAR-LS diagnostics are incomplete: " + ", ".join(missing)
        )

    violations = []
    cooldown_steps = stats["reverse_caar_cooldown_steps"]
    reverse_override_enabled = stats["reverse_caar_override_enabled"]
    if not isinstance(reverse_override_enabled, bool):
        violations.append("reverse CAAR override flag is invalid")
        reverse_override_enabled = True
    if not isinstance(cooldown_steps, int) or cooldown_steps < 0:
        violations.append("reverse CAAR cooldown steps are invalid")
        cooldown_steps = 0
    if cooldown_steps > 0 and not reverse_override_enabled:
        violations.append("reverse cooldown is enabled without its override")
    if stats["hybrid_mode"] != _ls_mode(
        cooldown_steps,
        reverse_override_enabled,
    ):
        violations.append("hybrid mode does not match the CAAR-LS contract")
    total = stats["total_action_count"]
    if total != stats["total_actions"]:
        violations.append("total action aliases disagree")
    if stats["nominal_caar_count"] + stats["nominal_ao_count"] != total:
        violations.append("nominal branches are not fully classified")
    if stats["executed_caar_count"] + stats["executed_ao_count"] != total:
        violations.append("executed sources are not fully classified")
    if stats["forced_caar_count"] != (
        stats["nominal_ao_count"] - stats["executed_ao_count"]
    ):
        violations.append("nominal AO fallbacks are not fully classified")
    if stats["comparison_cadence"] != "every_step_per_agent":
        violations.append("absolute returns are not compared every step")
    if stats["switch_constraint"] != _ls_switch_constraint(cooldown_steps):
        violations.append("switch constraint does not match cooldown config")
    if stats["reverse_caar_cooldown_includes_trigger_step"] is not True:
        violations.append("reverse cooldown trigger-step semantics are invalid")
    if stats["probe_call_count"] != 0:
        violations.append("Probe was called")
    if (
        reverse_override_enabled
        and stats["reverse_ao_executed_count"] != 0
    ):
        violations.append("a reverse raw Plan proposal executed as AO")
    if (
        not reverse_override_enabled
        and stats["reverse_override_count"] != 0
    ):
        violations.append("predictor-only mode recorded a reverse override")
    if stats["final_none_action_count"] != 0:
        violations.append("a final environment action was None")

    bounded_by_total = (
        "value_comparison_count",
        "branch_switch_count",
        "nominal_caar_count",
        "nominal_ao_count",
        "executed_caar_count",
        "executed_ao_count",
        "reverse_count",
        "reverse_override_count",
        "reverse_ao_executed_count",
        "none_count",
        "none_override_count",
        "invalid_plan_count",
        "invalid_override_count",
        "nonfinite_value_count",
        "forced_caar_count",
        "reverse_caar_cooldown_trigger_count",
        "reverse_caar_cooldown_action_count",
        "reverse_caar_cooldown_followup_action_count",
    )
    for key in bounded_by_total:
        if not 0 <= stats[key] <= total:
            violations.append(f"{key} is outside total-action bounds")
    if stats["branch_switch_count"] > stats["value_comparison_count"]:
        violations.append("branch switches exceed value comparisons")
    if stats["nonfinite_value_count"] > stats["value_comparison_count"]:
        violations.append("non-finite comparisons exceed value comparisons")
    if stats["value_comparison_count"] != total:
        violations.append("absolute-return values were not compared every step")
    for key in (
        "nonfinite_caar_value_count",
        "nonfinite_ao_value_count",
    ):
        if not 0 <= stats[key] <= stats["nonfinite_value_count"]:
            violations.append(f"{key} is inconsistent")
    if stats["reverse_override_count"] > stats["reverse_count"]:
        violations.append("reverse overrides exceed reverse proposals")
    if stats["reverse_ao_executed_count"] > stats["reverse_count"]:
        violations.append("executed reverse AO exceeds reverse proposals")
    if stats["reverse_ao_executed_count"] > stats["executed_ao_count"]:
        violations.append("executed reverse AO exceeds all AO executions")
    if stats["none_override_count"] > stats["none_count"]:
        violations.append("None overrides exceed None proposals")
    if stats["invalid_override_count"] > stats["invalid_plan_count"]:
        violations.append("invalid overrides exceed invalid proposals")
    override_counts = (
        stats["reverse_override_count"],
        stats["none_override_count"],
        stats["invalid_override_count"],
    )
    if any(value > stats["forced_caar_count"] for value in override_counts):
        violations.append("an override class exceeds all forced CAAR actions")
    if stats["forced_caar_count"] > sum(override_counts):
        violations.append("a forced CAAR action has no safety-override reason")
    cooldown_triggers = stats["reverse_caar_cooldown_trigger_count"]
    cooldown_actions = stats["reverse_caar_cooldown_action_count"]
    cooldown_followups = stats[
        "reverse_caar_cooldown_followup_action_count"
    ]
    if cooldown_actions != cooldown_triggers + cooldown_followups:
        violations.append("reverse cooldown action accounting is inconsistent")
    if cooldown_actions > stats["executed_caar_count"]:
        violations.append("reverse cooldown actions exceed CAAR executions")
    if cooldown_steps == 0:
        if cooldown_triggers or cooldown_actions or cooldown_followups:
            violations.append("disabled reverse cooldown recorded actions")
    else:
        if cooldown_triggers != stats["reverse_override_count"]:
            violations.append("reverse cooldown triggers miss reverse overrides")
        if cooldown_actions > cooldown_triggers * cooldown_steps:
            violations.append("reverse cooldown action count exceeds its bound")
        if cooldown_followups > cooldown_triggers * (cooldown_steps - 1):
            violations.append("reverse cooldown followups exceed their bound")
    max_remaining = stats["max_reverse_caar_cooldown_remaining"]
    if not 0 <= max_remaining <= max(cooldown_steps - 1, 0):
        violations.append("maximum reverse cooldown remaining is invalid")
    if not (
        0
        <= stats["max_concurrent_nominal_ao"]
        <= stats["nominal_ao_count"]
    ):
        violations.append("maximum concurrent nominal AO count is inconsistent")
    if not (
        0
        <= stats["max_concurrent_ao_executed"]
        <= stats["executed_ao_count"]
    ):
        violations.append("maximum concurrent executed AO count is inconsistent")
    road_keys = {
        "road_like",
        "road_agent_density",
        "road_caar_only_density_threshold",
        "density_gate_active",
        "road_density_gate_forced_nominal_count",
        "road_topology_provenance",
    }
    if road_keys & set(stats):
        missing_road = sorted(road_keys - set(stats))
        if missing_road:
            violations.append(
                "road diagnostics are incomplete: "
                + ", ".join(missing_road)
            )
        else:
            road_like = stats["road_like"]
            gate_active = stats["density_gate_active"]
            if not isinstance(road_like, bool):
                violations.append("road topology decision is invalid")
            if not isinstance(gate_active, bool):
                violations.append("road density gate flag is invalid")
            gate_forced = stats["road_density_gate_forced_nominal_count"]
            if not isinstance(gate_forced, int) or not 0 <= gate_forced <= total:
                violations.append("road density-gate accounting is invalid")
            provenance = stats["road_topology_provenance"]
            if not isinstance(provenance, dict) or provenance.get(
                "uses_map_name"
            ) is not False:
                violations.append("road topology provenance is invalid")
            if gate_active:
                threshold = stats["road_caar_only_density_threshold"]
                density = stats["road_agent_density"]
                if road_like is not True:
                    violations.append("density gate activated on non-road topology")
                if threshold is None or density is None or density < threshold:
                    violations.append("density gate activated below its threshold")
                if stats["nominal_ao_count"] or stats["executed_ao_count"]:
                    violations.append("density gate allowed an AO branch")
    if violations:
        raise RuntimeError(
            "CAAR-LS safety contract failed: " + "; ".join(violations)
        )


class _MoveFailureTracker:
    """Count submitted moves that the environment did not execute."""

    METRIC_VERSION = "submitted_nonwait_no_position_change_v1"

    def __init__(self, moves):
        self.moves = tuple(
            tuple(int(value) for value in move) for move in moves
        )
        self.environment_step_count = 0
        self.active_agent_step_count = 0
        self.wait_action_count = 0
        self.move_attempt_count = 0
        self.successful_move_count = 0
        self.conflict_count = 0
        self.agent_conflict_count = 0
        self.other_or_unattributed_conflict_count = 0
        self.conflict_step_count = 0

    @staticmethod
    def _point(observation):
        value = observation["xy"]
        return int(value[0]), int(value[1])

    def capture(self, actions, observations, dones, infos):
        positions = [self._point(observation) for observation in observations]
        active = [
            not bool(dones[index])
            and bool(infos[index].get("is_active", True))
            for index in range(len(observations))
        ]
        attempts = []
        waits = 0
        for index, is_active in enumerate(active):
            if not is_active:
                continue
            try:
                action = int(actions[index])
                delta = self.moves[action]
            except (IndexError, TypeError, ValueError):
                waits += 1
                continue
            if delta == (0, 0):
                waits += 1
                continue
            position = positions[index]
            target = position[0] + delta[0], position[1] + delta[1]
            attempts.append((index, position, target))
        return {
            "positions": positions,
            "active_count": sum(active),
            "wait_count": waits,
            "attempts": attempts,
        }

    def commit(self, pending, observations):
        after_positions = [
            self._point(observation) for observation in observations
        ]
        occupied = {}
        for index, position in enumerate(pending["positions"]):
            occupied.setdefault(position, set()).add(index)
        target_counts = {}
        for _, _, target in pending["attempts"]:
            target_counts[target] = target_counts.get(target, 0) + 1

        failed_this_step = 0
        agent_conflicts = 0
        successes = 0
        for index, before, target in pending["attempts"]:
            if after_positions[index] != before:
                successes += 1
                continue
            failed_this_step += 1
            occupied_by_other = any(
                other != index for other in occupied.get(target, ())
            )
            if occupied_by_other or target_counts[target] > 1:
                agent_conflicts += 1

        self.environment_step_count += 1
        self.active_agent_step_count += pending["active_count"]
        self.wait_action_count += pending["wait_count"]
        self.move_attempt_count += len(pending["attempts"])
        self.successful_move_count += successes
        self.conflict_count += failed_this_step
        self.agent_conflict_count += agent_conflicts
        self.other_or_unattributed_conflict_count += (
            failed_this_step - agent_conflicts
        )
        if failed_this_step:
            self.conflict_step_count += 1

    @staticmethod
    def _rate(numerator, denominator):
        return float(numerator / denominator) if denominator else 0.0

    def metrics(self):
        return {
            "congestion_metric_version": self.METRIC_VERSION,
            "environment_step_count_observed": self.environment_step_count,
            "active_agent_step_count": self.active_agent_step_count,
            "wait_action_count": self.wait_action_count,
            "move_attempt_count": self.move_attempt_count,
            "successful_move_count": self.successful_move_count,
            "conflict_count": self.conflict_count,
            "move_failure_count": self.conflict_count,
            "agent_conflict_count": self.agent_conflict_count,
            "other_or_unattributed_conflict_count": (
                self.other_or_unattributed_conflict_count
            ),
            "conflict_step_count": self.conflict_step_count,
            "congestion_rate": self._rate(
                self.conflict_count,
                self.move_attempt_count,
            ),
            "agent_conflict_rate": self._rate(
                self.agent_conflict_count,
                self.move_attempt_count,
            ),
            "conflict_agent_step_rate": self._rate(
                self.conflict_count,
                self.active_agent_step_count,
            ),
            "conflict_step_rate": self._rate(
                self.conflict_step_count,
                self.environment_step_count,
            ),
        }


def run_algorithm(
    algo,
    *,
    map_name,
    max_episode_steps,
    seed,
    num_agents,
    obs_radius,
    animate,
    on_target,
    collision_system,
    map_text,
):
    """Run one episode and attach environment-level congestion metrics."""

    gc_kwargs = {
        "max_episode_steps": max_episode_steps,
        "seed": seed,
        "num_agents": num_agents,
        "on_target": on_target,
    }
    if obs_radius is not None:
        gc_kwargs["obs_radius"] = obs_radius
    if collision_system is not None:
        gc_kwargs["collision_system"] = collision_system
    if map_text is not None:
        gc_kwargs["map"] = map_text
        gc_kwargs["map_name"] = None
    else:
        gc_kwargs["map_name"] = map_name

    grid_config = POMAPFConfig(**gc_kwargs)
    env = make_pomapf(grid_config=grid_config, with_animations=False)
    if animate:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        map_short = map_name.split("-")[-1] if "-" in map_name else map_name
        directory = Path("renders") / (
            f"{map_short}_{type(algo).__name__}_{num_agents}agents_{timestamp}"
        )
        env = AnimationMonitor(
            env,
            AnimationConfig(directory=str(directory)),
        )

    tracker = _MoveFailureTracker(grid_config.MOVES)
    try:
        observations, _ = env.reset()
        algo.after_reset()
        if hasattr(algo, "set_grid_config"):
            algo.set_grid_config(env.grid_config)
        if hasattr(algo, "set_env"):
            algo.set_env(env)
        results_holder = ResultsHolder()
        dones = [False for _ in observations]
        infos = [{"is_active": True} for _ in observations]
        rewards = [0 for _ in observations]
        with torch.no_grad():
            while True:
                actions = algo.act(observations, rewards, dones, infos)
                pending = tracker.capture(
                    actions,
                    observations,
                    dones,
                    infos,
                )
                observations, rewards, terminated, truncated, infos = env.step(
                    actions
                )
                tracker.commit(pending, observations)
                dones = [
                    terminated_value or truncated_value
                    for terminated_value, truncated_value in zip(
                        terminated,
                        truncated,
                    )
                ]
                results_holder.after_step(infos)
                algo.after_step(dones)
                if all(dones):
                    break
        results = results_holder.get_final()
        results.update(tracker.metrics())
        results["algorithm"] = type(algo).__name__
        return results
    finally:
        env.close()



def run_single_experiment(task):

    quiet_model_logs()

    algo_name = canonical_algorithm_name(task["algorithm"]) or task["algorithm"]

    main_dir = task["main_dir"]

    seed = task["seed"]

    random.seed(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)

    # CAAR-LS estimators are trained against an episode-fresh CAAR
    # observation normalizer. Reusing the actor across maps would make its
    # online normalization state depend on task order and silently change the
    # two policies whose returns are being compared.
    use_cache = bool(task.get("cache_algorithms", False)) and (
        algo_name not in ("CAAR-LS", "CAAR-RS")
    )

    cache_key = (

        algo_name,

        str(Path(main_dir).resolve()),

        seed,

        task.get("caar_weights_path"),

        task.get("caar_ra_weights_path"),

        task.get("caar_ls_caar_estimator_checkpoint"),

        task.get("caar_ls_ao_estimator_checkpoint"),

        task.get("caar_ls_value_margin", 0.0),

        task.get("caar_ls_reverse_caar_override_enabled", True),

        task.get("caar_ls_reverse_caar_cooldown_steps", 0),

        task.get(
            "caar_ls_road_topology_adaptive_cooldown_enabled",
            False,
        ),

        task.get("caar_ls_road_open4_threshold", 0.68),

        task.get("caar_ls_road_dense_obstacle_threshold", 0.70),

        task.get("caar_ls_road_reverse_caar_cooldown_steps", 8),

        task.get("caar_ls_road_caar_only_density_threshold"),

        task.get("notau_weights_path"),

        task.get("dhc_weights_path"),

        task.get("dcc_weights_path"),

        task.get("scrimp_weights_path"),

        task.get("epom_weights_path"),

    )


    try:

        if use_cache:

            if cache_key not in _worker_algo_cache:

                _worker_algo_cache[cache_key] = build_algorithm(

                    algo_name,

                    main_dir,

                    seed,

                    caar_weights_path=task.get("caar_weights_path"),

                    caar_ra_weights_path=task.get("caar_ra_weights_path"),

                    caar_ls_caar_estimator_checkpoint=task.get(
                        "caar_ls_caar_estimator_checkpoint"
                    ),

                    caar_ls_ao_estimator_checkpoint=task.get(
                        "caar_ls_ao_estimator_checkpoint"
                    ),

                    caar_ls_value_margin=task.get(
                        "caar_ls_value_margin", 0.0
                    ),

                    caar_ls_reverse_caar_override_enabled=task.get(
                        "caar_ls_reverse_caar_override_enabled", True
                    ),

                    caar_ls_reverse_caar_cooldown_steps=task.get(
                        "caar_ls_reverse_caar_cooldown_steps", 0
                    ),

                    caar_ls_road_topology_adaptive_cooldown_enabled=task.get(
                        "caar_ls_road_topology_adaptive_cooldown_enabled",
                        False,
                    ),

                    caar_ls_road_open4_threshold=task.get(
                        "caar_ls_road_open4_threshold", 0.68
                    ),

                    caar_ls_road_dense_obstacle_threshold=task.get(
                        "caar_ls_road_dense_obstacle_threshold", 0.70
                    ),

                    caar_ls_road_reverse_caar_cooldown_steps=task.get(
                        "caar_ls_road_reverse_caar_cooldown_steps", 8
                    ),

                    caar_ls_road_caar_only_density_threshold=task.get(
                        "caar_ls_road_caar_only_density_threshold"
                    ),

                    notau_weights_path=task.get("notau_weights_path"),

                    dhc_weights_path=task.get("dhc_weights_path"),

                    dcc_weights_path=task.get("dcc_weights_path"),

                    scrimp_weights_path=task.get("scrimp_weights_path"),

                    epom_weights_path=task.get("epom_weights_path"),

                )

            algo = _worker_algo_cache[cache_key]

        else:

            algo = build_algorithm(

                algo_name,

                main_dir,

                seed,

                caar_weights_path=task.get("caar_weights_path"),

                caar_ra_weights_path=task.get("caar_ra_weights_path"),

                caar_ls_caar_estimator_checkpoint=task.get(
                    "caar_ls_caar_estimator_checkpoint"
                ),

                caar_ls_ao_estimator_checkpoint=task.get(
                    "caar_ls_ao_estimator_checkpoint"
                ),

                caar_ls_value_margin=task.get(
                    "caar_ls_value_margin", 0.0
                ),

                caar_ls_reverse_caar_override_enabled=task.get(
                    "caar_ls_reverse_caar_override_enabled", True
                ),

                caar_ls_reverse_caar_cooldown_steps=task.get(
                    "caar_ls_reverse_caar_cooldown_steps", 0
                ),

                caar_ls_road_topology_adaptive_cooldown_enabled=task.get(
                    "caar_ls_road_topology_adaptive_cooldown_enabled",
                    False,
                ),

                caar_ls_road_open4_threshold=task.get(
                    "caar_ls_road_open4_threshold", 0.68
                ),

                caar_ls_road_dense_obstacle_threshold=task.get(
                    "caar_ls_road_dense_obstacle_threshold", 0.70
                ),

                caar_ls_road_reverse_caar_cooldown_steps=task.get(
                    "caar_ls_road_reverse_caar_cooldown_steps", 8
                ),

                caar_ls_road_caar_only_density_threshold=task.get(
                    "caar_ls_road_caar_only_density_threshold"
                ),

                notau_weights_path=task.get("notau_weights_path"),

                dhc_weights_path=task.get("dhc_weights_path"),

                dcc_weights_path=task.get("dcc_weights_path"),

                scrimp_weights_path=task.get("scrimp_weights_path"),

                epom_weights_path=task.get("epom_weights_path"),

            )


        start = time.time()

        result = run_algorithm(

            algo,

            map_name=task["map_name"],

            max_episode_steps=task["max_steps"],

            seed=seed,

            num_agents=task["num_agents"],

            obs_radius=task.get("obs_radius"),

            animate=task["animate"],

            on_target=task.get("on_target", "restart"),

            collision_system=task.get("collision_system"),

            map_text=task.get("map_text"),

        )

        run_time = time.time() - start



        on_target = task.get("on_target", "restart")
        is_restart = on_target == "restart"
        is_replan = algo_name in ("Replan", "AO-RePlan")

        if hasattr(algo, "get_hybrid_stats"):
            hybrid_stats = algo.get_hybrid_stats()
        elif hasattr(algo, "get_switch_stats"):
            hybrid_stats = algo.get_switch_stats()
        else:
            hybrid_stats = {}
        if algo_name == "CAAR-Yield":
            validate_caar_yield_stats(hybrid_stats)
        elif algo_name == "CAAR-PB":
            validate_caar_pb_stats(hybrid_stats)
        elif algo_name == "CAAR-RA":
            validate_caar_ra_stats(hybrid_stats)
        elif algo_name == "CAAR-RS":
            validate_caar_rs_stats(hybrid_stats)
        elif algo_name == "CAAR-LS":
            validate_caar_ls_stats(hybrid_stats)
        correction_stats = (
            algo.get_action_correction_stats()
            if hasattr(algo, "get_action_correction_stats")
            else {}
        )
        result_record = {
            "algorithm": algo_name,
            "map_name": task["map_name"],
            "num_agents": task["num_agents"],
            "max_steps": task["max_steps"],
            "seed": seed,
            "on_target": on_target,
            "run_time_seconds": run_time,
        }
        result_record.update(
            {
                key: value
                for key, value in result.items()
                if key != "algorithm"
            }
        )
        result_record.update(correction_stats)

        result_record.update(hybrid_stats)

        if is_restart:
            if is_replan:
                result_record["reverse_action_rate"] = getattr(algo, "reverse_action_rate", None)


        return result_record

    except Exception as exc:

        import traceback


        return {

            "algorithm": algo_name,

            "map_name": task["map_name"],

            "num_agents": task["num_agents"],

            "max_steps": task["max_steps"],

            "seed": seed,

            "on_target": task.get("on_target", "restart"),

            "avg_throughput": None,

            "run_time_seconds": 0.0,

            "error": str(exc),

            "traceback": traceback.format_exc(),

        }



def parse_algorithms(value):

    if value.strip().lower() == "all":
        return list(DEFAULT_ALGORITHMS)


    raw_algorithms = [item.strip() for item in value.split(",") if item.strip()]

    algorithms = []

    seen = set()

    unknown = []

    for item in raw_algorithms:

        canonical = canonical_algorithm_name(item)

        if canonical is None:

            unknown.append(item)

        else:

            if canonical not in seen:

                algorithms.append(canonical)

                seen.add(canonical)

    if unknown:

        choices = ", ".join(SUPPORTED_ALGORITHMS)

        raise argparse.ArgumentTypeError(f"Unknown algorithm(s): {unknown}. Choices: {choices}")

    if not algorithms:

        raise argparse.ArgumentTypeError("No algorithms selected")

    return algorithms



def parse_agent_counts(args):

    if args.agents:

        counts = [int(item.strip()) for item in args.agents.split(",") if item.strip()]

    else:

        if args.agent_step <= 0:

            raise ValueError("--agent-step must be positive")

        counts = list(range(args.agent_start, args.agent_stop + 1, args.agent_step))


    counts = sorted(set(counts))

    if not counts:

        raise ValueError("No agent counts selected")

    if any(count <= 0 for count in counts):

        raise ValueError("Agent counts must all be positive")

    return counts



def parse_seeds(args):

    if args.seeds:

        seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]

        seeds = sorted(set(seeds))

        if not seeds:

            raise ValueError("No seeds selected")

        return seeds

    return [args.seed]



def parse_maps(map_types_value, map_overrides):

    if map_types_value.strip().lower() == "custom":

        maps = {}

        for item in map_overrides:

            map_name = item.strip()

            if not map_name:

                continue

            maps[map_name] = map_name

        if not maps:

            raise ValueError("When --map-types=custom, --map must provide at least one map name")

        return maps


    maps = dict(DEFAULT_MAPS)

    for item in map_overrides:

        if "=" not in item:

            raise ValueError("--map must use the format map_type=map_name")

        map_type, map_name = item.split("=", 1)

        map_type = map_type.strip()

        map_name = map_name.strip()

        if not map_type or not map_name:

            raise ValueError("--map must use non-empty map_type=map_name values")

        maps[map_type] = map_name


    if map_types_value.strip().lower() == "all":

        selected_types = list(DEFAULT_MAPS.keys())

    else:

        selected_types = [item.strip() for item in map_types_value.split(",") if item.strip()]


    unknown = [item for item in selected_types if item not in maps]

    if unknown:

        raise ValueError(f"Unknown map type(s): {unknown}. Available: {sorted(maps)}")

    if not selected_types:

        raise ValueError("No map types selected")


    return {map_type: maps[map_type] for map_type in selected_types}



def _looks_like_movingai_map(lines):

    if not lines:

        return False

    head = [line.strip().lower() for line in lines[:4]]

    return "map" in head and any(line.startswith("type ") for line in head)



def _translate_map_rows(rows):

    trans = {".": ".", "G": ".", "S": ".", "W": "#", "T": "#", "@": "#", "O": "#"}

    return ["".join(trans.get(ch, "#") for ch in row.rstrip()) for row in rows if row.strip()]



def load_map_text(path_or_url, trim_border=False):

    path_or_url = str(path_or_url)

    if path_or_url.startswith(("http://", "https://")):

        raw_text = urllib.request.urlopen(path_or_url, timeout=30).read().decode("utf-8")

        source = path_or_url

    else:

        raw_text = Path(path_or_url).read_text()

        source = str(Path(path_or_url).resolve())


    lines = raw_text.splitlines()

    if _looks_like_movingai_map(lines):

        map_start = next(i for i, line in enumerate(lines) if line.strip().lower() == "map") + 1

        rows = _translate_map_rows(lines[map_start:])

    else:

        rows = [line.rstrip() for line in lines if line.strip()]


    if trim_border and len(rows) >= 3 and len(rows[0]) >= 3:

        rows = [row[1:-1] for row in rows[1:-1]]


    if not rows:

        raise ValueError(f"Map source produced no rows: {path_or_url}")

    width = len(rows[0])

    if width == 0 or any(len(row) != width for row in rows):

        raise ValueError(f"Map source must contain a non-empty rectangular grid: {path_or_url}")


    label = Path(path_or_url).name if not path_or_url.startswith(("http://", "https://")) else Path(path_or_url.split("?")[0]).name

    return {

        "map_name": label or "custom-map",

        "map_text": "\n".join(rows),

        "map_source": source,

        "map_size": [len(rows[0]), len(rows)],

    }



def load_map_list_snapshot(path, registry_path=None):

    """Snapshot selected names and the exact grids sent to workers."""

    import yaml


    path = Path(path).resolve()

    payload = path.read_bytes()

    try:

        data = yaml.safe_load(payload.decode("utf-8"))

    except (UnicodeDecodeError, yaml.YAMLError) as error:

        raise ValueError(f"Could not parse map list {path}: {error}") from error

    if not isinstance(data, dict) or not data:

        raise ValueError("--map-list must contain a non-empty YAML mapping")

    if any(not isinstance(name, str) or not name for name in data):

        raise ValueError("--map-list keys must be non-empty strings")

    registry_payload = payload

    registry = data

    if any(not isinstance(value, str) or not value.strip() for value in data.values()):

        if registry_path is None:

            raise ValueError(
                "--map-list entries without grid text require a registry"
            )

        registry_path = Path(registry_path).resolve()

        registry_payload = registry_path.read_bytes()

        try:

            registry = yaml.safe_load(registry_payload.decode("utf-8"))

        except (UnicodeDecodeError, yaml.YAMLError) as error:

            raise ValueError(
                f"Could not parse map registry {registry_path}: {error}"
            ) from error

        if not isinstance(registry, dict) or not registry:

            raise ValueError(
                f"Map registry must be a non-empty mapping: {registry_path}"
            )

    map_texts = {}

    for name, selected_value in data.items():

        value = (

            selected_value

            if isinstance(selected_value, str) and selected_value.strip()

            else registry.get(name)

        )

        if not isinstance(value, str) or not value.strip():

            raise ValueError(f"No grid text found for map {name!r}")

        rows = value.splitlines()

        if (

            not rows

            or len({len(row) for row in rows}) != 1

            or not rows[0]

            or any(set(row) - {".", "#"} for row in rows)

        ):

            raise ValueError(
                f"Map {name!r} must be a non-empty rectangular .# grid"
            )

        map_texts[name] = value

    return (

        {name: name for name in data},

        map_texts,

        hashlib.sha256(payload).hexdigest(),

        hashlib.sha256(registry_payload).hexdigest(),

        path,

    )



def build_tasks(
    algorithms,
    maps,
    agent_counts,
    seeds,
    args,
    custom_map=None,
    map_texts=None,
):

    map_items = (

        [("custom", custom_map["map_name"], custom_map["map_text"], custom_map["map_source"])]

        if custom_map is not None

        else [
            (
                map_type,
                map_name,
                map_texts.get(map_name) if map_texts is not None else None,
                None,
            )
            for map_type, map_name in maps.items()
        ]

    )

    return [

        {

            "algorithm": algorithm,

            "map_type": map_type,

            "map_name": map_name,

            "map_text": map_text,

            "map_source": map_source,

            "num_agents": num_agents,

            "obs_radius": args.obs_radius,

            "max_steps": args.max_steps,

            "seed": seed,

            "animate": args.animate,

            "main_dir": args.main_dir,

            "on_target": args.on_target,

            "collision_system": args.collision_system,

            "caar_weights_path": args.caar_weights_path,

            "caar_ra_weights_path": args.caar_ra_weights_path,

            "caar_ls_caar_estimator_checkpoint": (
                args.caar_ls_caar_estimator_checkpoint
            ),

            "caar_ls_ao_estimator_checkpoint": (
                args.caar_ls_ao_estimator_checkpoint
            ),

            "caar_ls_value_margin": args.caar_ls_value_margin,

            "caar_ls_reverse_caar_override_enabled": (
                args.caar_ls_reverse_caar_override_enabled
            ),

            "caar_ls_reverse_caar_cooldown_steps": (
                args.caar_ls_reverse_caar_cooldown_steps
            ),

            "caar_ls_road_topology_adaptive_cooldown_enabled": getattr(
                args,
                "caar_ls_road_topology_adaptive_cooldown_enabled",
                False,
            ),

            "caar_ls_road_open4_threshold": getattr(
                args, "caar_ls_road_open4_threshold", 0.68
            ),

            "caar_ls_road_dense_obstacle_threshold": getattr(
                args, "caar_ls_road_dense_obstacle_threshold", 0.70
            ),

            "caar_ls_road_reverse_caar_cooldown_steps": getattr(
                args, "caar_ls_road_reverse_caar_cooldown_steps", 8
            ),

            "caar_ls_road_caar_only_density_threshold": getattr(
                args, "caar_ls_road_caar_only_density_threshold", None
            ),

            "notau_weights_path": args.notau_weights_path,

            "dhc_weights_path": args.dhc_weights_path,

            "dcc_weights_path": args.dcc_weights_path,

            "scrimp_weights_path": args.scrimp_weights_path,

            "epom_weights_path": args.epom_weights_path,

            "cache_algorithms": (
                args.cache_algorithms and algorithm != "CAAR-LS"
            ),

        }

        for algorithm in algorithms

        for map_type, map_name, map_text, map_source in map_items

        for num_agents in agent_counts

        for seed in seeds

    ]



def format_duration(seconds):

    if seconds < 60:

        return f"{seconds:.1f}s"

    minutes, rem = divmod(seconds, 60)

    if minutes < 60:

        return f"{int(minutes)}m{int(rem):02d}s"

    hours, minutes = divmod(minutes, 60)

    return f"{int(hours)}h{int(minutes):02d}m"



def run_experiments(tasks, workers):

    results = []

    total = len(tasks)

    start_time = time.time()


    print(f"Starting experiments: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"Total: {total} | Workers: {workers}")

    print("-" * 110, flush=True)


    with ProcessPoolExecutor(max_workers=workers) as executor:

        futures = [executor.submit(run_single_experiment, task) for task in tasks]

        for index, future in enumerate(as_completed(futures), start=1):

            result = future.result()

            elapsed = time.time() - start_time

            eta = elapsed / index * (total - index) if index else 0.0

            result["completed_index"] = index

            result["total_experiments"] = total

            result["elapsed_since_start_seconds"] = elapsed

            result["eta_after_result_seconds"] = eta

            result["finished_at"] = datetime.now().isoformat(timespec="seconds")

            results.append(result)


            if result.get("error"):

                status = f"ERROR: {result['error']}"

            else:

                on_target = result.get("on_target", "restart")

                gate_str = ""

                if result.get("learning_ratio") is not None:

                    switch_label = "epom" if result["algorithm"] == "AS" else "caar"

                    gate_str += f" {switch_label}={result['learning_ratio']:.1%}"

                if result.get("planner_ratio") is not None:

                    gate_str += f" planner={result['planner_ratio']:.1%}"

                for key, label in (
                    ("caar_action_ratio", "caar_actions"),
                    ("guided_agent_step_ratio", "guided"),
                ):

                    if result.get(key) is not None:

                        gate_str += f" {label}={result[key]:.1%}"

                diag_str = ""

                if result.get("congestion_rate") is not None:

                    diag_str += (

                        f" congestion={result['congestion_rate']:.1%}"

                    )

                if on_target != "restart":

                    if result.get("ep_length") is not None:

                        diag_str += f" ep_len={result['ep_length']:.1f}"

                    isr = result.get("ISR")

                    csr = result.get("CSR")

                    status = (

                        f"isr={(0.0 if isr is None else isr):.1%} "

                        f"csr={(0.0 if csr is None else csr):.1%}{gate_str}{diag_str} "

                        f"run={format_duration(result['run_time_seconds'])}"

                    )

                else:

                    if result.get("reverse_action_rate") is not None:

                        diag_str += f" rev={result['reverse_action_rate']:.1%}"

                    status = (

                        f"throughput={result['avg_throughput']:.4f}{gate_str}{diag_str} "

                        f"run={format_duration(result['run_time_seconds'])}"

                    )


            print(

                f"[{index:>3}/{total:<3}] {result['algorithm']:<{ALGORITHM_COLUMN_WIDTH}} | "

                f"{result['map_name']:<22} | {result['num_agents']:>3} agents | "

                f"{status:<34}",

                flush=True,

            )


    return results, time.time() - start_time



def save_results(results, metadata, output_dir, filename=None):

    os.makedirs(output_dir, exist_ok=True)

    if filename:

        output_path = Path(output_dir) / filename

    else:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_path = Path(output_dir) / f"experiments_{timestamp}.json"

    payload = {

        "metadata": metadata,

        "results": results,

    }

    temporary_path = output_path.with_name(f".{output_path.name}.tmp")

    with temporary_path.open("w", encoding="utf-8") as f:

        json.dump(payload, f, indent=2)

        f.flush()

        os.fsync(f.fileno())

    temporary_path.replace(output_path)

    print(f"\nResults saved: {output_path}")

    return output_path



def parse_args():

    parser = argparse.ArgumentParser(description="Unified Lifelong MAPF experiment runner")

    parser.add_argument(

        "--algorithms",

        type=parse_algorithms,

        default=list(DEFAULT_ALGORITHMS),

        help=(
            "Comma-separated algorithms, or 'all'. "
            f"Choices: {', '.join(SUPPORTED_ALGORITHMS)}"
        ),

    )

    parser.add_argument("--agents", type=str, default=None, help="Comma-separated agent counts, e.g. 50,100,200")

    parser.add_argument("--agent-start", type=int, default=50, help="First agent count when --agents is not set")

    parser.add_argument("--agent-stop", type=int, default=500, help="Last inclusive agent count when --agents is not set")

    parser.add_argument("--agent-step", type=int, default=50, help="Agent count step when --agents is not set")

    parser.add_argument("--workers", "--works", dest="workers", type=int, default=8, help="Parallel workers")

    parser.add_argument(

        "--obs-radius",

        type=int,

        default=None,

        help="Override the local observation radius (default: environment configuration)",

    )

    parser.add_argument(

        "--cache-algorithms",

        action="store_true",

        help="Reuse algorithm objects inside each worker. Faster, but less isolated between experiment tasks.",

    )

    parser.add_argument("--animate", action="store_true", help="Generate SVG animations")

    parser.add_argument("--max-steps", type=int, default=512, help="Episode length")

    parser.add_argument("--seed", type=int, default=0, help="Random seed")

    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seeds, e.g. 0,1,2")

    parser.add_argument("--main-dir", type=str, default="./", help="Project root directory")

    parser.add_argument("--map-types", type=str, default="all", help="Comma-separated map types, or 'all'")

    parser.add_argument(

        "--map",

        action="append",

        default=[],

        help="Override one representative map with map_type=map_name. Can be repeated.",

    )

    parser.add_argument("--map-url", type=str, default=None, help="Custom map URL. Supports MovingAI .map files.")

    parser.add_argument("--map-file", type=str, default=None, help="Custom local map file path.")

    parser.add_argument("--map-list", type=str, default=None, help="YAML file whose top-level keys are map names (e.g. maps/eval.yaml)")

    parser.add_argument("--trim-border", dest="trim_border", action="store_true", help="Trim one-cell border from custom map")

    parser.add_argument("--no-trim-border", dest="trim_border", action="store_false", help="Do not trim border from custom map")

    parser.set_defaults(trim_border=None)

    parser.add_argument(

        "--on-target",

        choices=("restart", "finish", "nothing"),

        default=None,

        help="Override Pogema on_target mode",

    )

    parser.add_argument(

        "--collision-system",

        choices=("soft", "block_both"),

        default=None,

        help="Override collision system",

    )

    parser.add_argument("--dhc-weights-path", type=str, default=None, help="Override DHC model .pth path")

    parser.add_argument("--dcc-weights-path", type=str, default=None, help="Override DCC model .pth path")

    parser.add_argument("--scrimp-weights-path", type=str, default=None, help="Override SCRIMP checkpoint .pkl path")

    parser.add_argument(

        "--epom-weights-path",

        type=str,

        default=None,

        help="Override EPOM weights directory",

    )

    parser.add_argument(

        "--caar-weights-path",

        dest="caar_weights_path",

        type=str,

        default=None,

        help="Override CAAR weights directory",

    )

    parser.add_argument(

        "--caar-ra-weights-path",

        dest="caar_ra_weights_path",

        type=str,

        default=None,

        help="Override CAAR-RA binary gate weights directory",

    )

    parser.add_argument(

        "--caar-ls-caar-estimator-checkpoint",

        "--caar-ls-caar-checkpoint",

        dest="caar_ls_caar_estimator_checkpoint",

        type=str,

        default=None,

        help="Override the CAAR absolute-return estimator checkpoint",

    )

    parser.add_argument(

        "--caar-ls-ao-estimator-checkpoint",

        "--caar-ls-ao-checkpoint",

        dest="caar_ls_ao_estimator_checkpoint",

        type=str,

        default=None,

        help="Override the AO-safe absolute-return estimator checkpoint",

    )

    parser.add_argument(

        "--caar-ls-value-margin",

        dest="caar_ls_value_margin",

        type=float,

        default=0.0,

        help="Require V_AO > V_CAAR + margin before selecting AO",

    )

    parser.add_argument(

        "--caar-ls-reverse-caar-override",

        dest="caar_ls_reverse_caar_override_enabled",

        action="store_true",

        default=True,

        help="Replace predictor-selected reverse AO proposals with CAAR",

    )

    parser.add_argument(

        "--no-caar-ls-reverse-caar-override",

        dest="caar_ls_reverse_caar_override_enabled",

        action="store_false",

        help="Let the return predictors execute reverse AO proposals",

    )

    parser.add_argument(

        "--caar-ls-reverse-caar-cooldown-steps",

        dest="caar_ls_reverse_caar_cooldown_steps",

        type=int,

        default=None,

        help=(
            "Keep each reverse-fallback agent on CAAR for this many steps, "
            "including the triggering step"
        ),

    )

    parser.add_argument(

        "--caar-ls-road-topology-adaptive-cooldown",

        dest="caar_ls_road_topology_adaptive_cooldown_enabled",

        action="store_true",

        default=False,

        help=(
            "Use the road-specific reverse cooldown when map-only topology "
            "features pass both thresholds"
        ),

    )

    parser.add_argument(

        "--caar-ls-road-open4-threshold",

        dest="caar_ls_road_open4_threshold",

        type=float,

        default=0.68,

        help="Minimum open-four-neighbour free-cell ratio for road topology",

    )

    parser.add_argument(

        "--caar-ls-road-dense-obstacle-threshold",

        dest="caar_ls_road_dense_obstacle_threshold",

        type=float,

        default=0.70,

        help="Minimum dense-obstacle ratio for road topology",

    )

    parser.add_argument(

        "--caar-ls-road-reverse-caar-cooldown-steps",

        dest="caar_ls_road_reverse_caar_cooldown_steps",

        type=int,

        default=8,

        help="Reverse-triggered CAAR dwell used on detected road topology",

    )

    parser.add_argument(

        "--caar-ls-road-caar-only-density-threshold",

        dest="caar_ls_road_caar_only_density_threshold",

        type=float,

        default=None,

        help=(
            "On detected road topology, stay with CAAR when "
            "num_agents/free_cells reaches this threshold"
        ),

    )

    parser.add_argument(

        "--notau-weights-path",

        dest="notau_weights_path",

        type=str,

        default=None,

        help="Override NoTau weights directory",

    )

    parser.add_argument("--output-dir", type=str, default="exp_result", help="Directory for JSON results")

    parser.add_argument("--output", type=str, default=None, help="Output filename (default: experiments_TIMESTAMP.json)")

    parser.add_argument("--save", dest="save", action="store_true", default=True, help="Save JSON results")

    parser.add_argument("--no-save", dest="save", action="store_false", help="Do not save JSON results")

    args = parser.parse_args()
    if args.caar_ls_reverse_caar_cooldown_steps is None:
        args.caar_ls_reverse_caar_cooldown_steps = (
            4 if args.caar_ls_reverse_caar_override_enabled else 0
        )
    return args



def main():

    multiprocessing.set_start_method("spawn", force=True)

    quiet_model_logs()

    args = parse_args()

    if args.workers < 1:

        raise ValueError("--workers must be at least 1")

    if args.max_steps < 1:

        raise ValueError("--max-steps must be at least 1")

    if args.obs_radius is not None and args.obs_radius < 1:

        raise ValueError("--obs-radius must be at least 1")

    if not np.isfinite(args.caar_ls_value_margin):

        raise ValueError("--caar-ls-value-margin must be finite")

    if args.caar_ls_reverse_caar_cooldown_steps < 0:
        raise ValueError(
            "--caar-ls-reverse-caar-cooldown-steps must be non-negative"
        )
    if (
        not args.caar_ls_reverse_caar_override_enabled
        and args.caar_ls_reverse_caar_cooldown_steps > 0
    ):
        raise ValueError(
            "reverse CAAR cooldown requires "
            "--caar-ls-reverse-caar-override"
        )
    for option, value in (
        (
            "--caar-ls-road-open4-threshold",
            args.caar_ls_road_open4_threshold,
        ),
        (
            "--caar-ls-road-dense-obstacle-threshold",
            args.caar_ls_road_dense_obstacle_threshold,
        ),
    ):
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{option} must be finite and within [0, 1]")
    if args.caar_ls_road_reverse_caar_cooldown_steps < 0:
        raise ValueError(
            "--caar-ls-road-reverse-caar-cooldown-steps must be "
            "non-negative"
        )
    if (
        args.caar_ls_road_topology_adaptive_cooldown_enabled
        and args.caar_ls_road_reverse_caar_cooldown_steps > 0
        and not args.caar_ls_reverse_caar_override_enabled
    ):
        raise ValueError(
            "road reverse CAAR cooldown requires "
            "--caar-ls-reverse-caar-override"
        )
    if (
        args.caar_ls_road_caar_only_density_threshold is not None
        and (
            not np.isfinite(
                args.caar_ls_road_caar_only_density_threshold
            )
            or args.caar_ls_road_caar_only_density_threshold < 0.0
        )
    ):
        raise ValueError(
            "--caar-ls-road-caar-only-density-threshold must be a "
            "finite non-negative value"
        )

    map_sources = sum(

        bool(value)

        for value in (args.map_url, args.map_file, args.map_list)

    )

    if map_sources > 1:

        raise ValueError("--map-url, --map-file, and --map-list are mutually exclusive")


    if args.on_target is None:

        args.on_target = "restart"

    if args.collision_system is None:

        args.collision_system = "soft"

    if args.trim_border is None:

        args.trim_border = False


    algorithms = args.algorithms
    waypoint_contract = hybrid_contract_metadata(algorithms)
    active_yield_contract = yield_contract_metadata(algorithms)
    probe_block_contract = pb_contract_metadata(algorithms)
    relative_advantage_contract = ra_contract_metadata(algorithms)
    rule_only_contract = rs_contract_metadata(algorithms)
    absolute_return_contract = ls_contract_metadata(
        algorithms,
        value_margin=args.caar_ls_value_margin,
        reverse_caar_override_enabled=(
            args.caar_ls_reverse_caar_override_enabled
        ),
        reverse_caar_cooldown_steps=(
            args.caar_ls_reverse_caar_cooldown_steps
        ),
        road_topology_adaptive_cooldown_enabled=(
            args.caar_ls_road_topology_adaptive_cooldown_enabled
        ),
        road_open4_threshold=args.caar_ls_road_open4_threshold,
        road_dense_obstacle_threshold=(
            args.caar_ls_road_dense_obstacle_threshold
        ),
        road_reverse_caar_cooldown_steps=(
            args.caar_ls_road_reverse_caar_cooldown_steps
        ),
        road_caar_only_density_threshold=(
            args.caar_ls_road_caar_only_density_threshold
        ),
    )
    active_contracts = [
        contract
        for contract in (
            waypoint_contract,
            active_yield_contract,
            probe_block_contract,
            relative_advantage_contract,
            rule_only_contract,
            absolute_return_contract,
        )
        if contract is not None
    ]
    if len(active_contracts) > 1:
        raise ValueError(
            "CAAR-RG, CAAR-Yield, CAAR-PB, CAAR-RA, CAAR-RS, and "
            "CAAR-LS hybrids "
            "must be evaluated "
            "in separate runs so "
            "each result file has one unambiguous hybrid contract."
        )
    hybrid_contract = active_contracts[0] if active_contracts else None

    agent_counts = parse_agent_counts(args)

    seeds = parse_seeds(args)


    custom_map = None

    map_list_sha256 = None

    map_registry_sha256 = None

    map_list_path = None

    map_texts = None

    if args.map_url or args.map_file:

        custom_map = load_map_text(args.map_url or args.map_file, trim_border=args.trim_border)

        maps = {"custom": custom_map["map_name"]}

    elif args.map_list:

        (

            maps,

            map_texts,

            map_list_sha256,

            map_registry_sha256,

            map_list_path,

        ) = load_map_list_snapshot(

            _project_path(args.main_dir, args.map_list),

            registry_path=_project_path(

                args.main_dir,

                "maps/eval.yaml",

            ),

        )

    else:

        maps = parse_maps(args.map_types, args.map)


    tasks = build_tasks(

        algorithms,

        maps,

        agent_counts,

        seeds,

        args,

        custom_map=custom_map,

        map_texts=map_texts,

    )

    if absolute_return_contract is not None:
        integrity_metadata = ls_integrity_metadata(
            args,
            map_list_sha256=map_list_sha256,
            map_registry_sha256=map_registry_sha256,
        )
    elif rule_only_contract is not None:
        integrity_metadata = rs_integrity_metadata(
            args,
            map_list_sha256=map_list_sha256,
            map_registry_sha256=map_registry_sha256,
        )
    elif relative_advantage_contract is not None:
        integrity_metadata = ra_integrity_metadata(
            args,
            map_list_sha256=map_list_sha256,
            map_registry_sha256=map_registry_sha256,
        )
    elif active_yield_contract is not None:
        integrity_metadata = yield_integrity_metadata(
            args,
            map_list_sha256=map_list_sha256,
            map_registry_sha256=map_registry_sha256,
        )
    elif probe_block_contract is not None:
        integrity_metadata = pb_integrity_metadata(
            args,
            map_list_sha256=map_list_sha256,
            map_registry_sha256=map_registry_sha256,
        )
    elif waypoint_contract is not None:
        integrity_metadata = hybrid_integrity_metadata(
            args,
            map_list_sha256=map_list_sha256,
            map_registry_sha256=map_registry_sha256,
        )
    else:
        integrity_metadata = None

    algorithm_cache = cache_algorithm_metadata(
        algorithms,
        args.cache_algorithms,
    )

    metadata = {

        "started_at": datetime.now().isoformat(timespec="seconds"),

        "runtime_provenance": runtime_provenance(),

        "congestion_metric": {

            "version": _MoveFailureTracker.METRIC_VERSION,

            "conflict_definition": (

                "an active agent submitted a non-wait movement action but "

                "its xy position was unchanged after env.step"

            ),

            "congestion_rate_denominator": (

                "submitted non-wait movement actions"

            ),

            "agent_conflict_classification": (

                "the failed move targeted a cell occupied by another agent "

                "before the step, or multiple agents targeted the same cell"

            ),

        },

        "algorithms": algorithms,

        "agent_counts": agent_counts,

        "seeds": seeds,

        "maps": maps,

        "workers": args.workers,

        "obs_radius": args.obs_radius,

        "animate": args.animate,

        "max_steps": args.max_steps,

        "on_target": args.on_target,

        "collision_system": args.collision_system,

        "custom_map": custom_map,

        "main_dir": args.main_dir,

        "caar_weights_path": args.caar_weights_path,

        "caar_ra_weights_path": args.caar_ra_weights_path,

        "caar_ls_caar_estimator_checkpoint": (
            args.caar_ls_caar_estimator_checkpoint
        ),

        "caar_ls_ao_estimator_checkpoint": (
            args.caar_ls_ao_estimator_checkpoint
        ),

        "caar_ls_value_margin": args.caar_ls_value_margin,

        "caar_ls_reverse_caar_override_enabled": (
            args.caar_ls_reverse_caar_override_enabled
        ),

        "caar_ls_reverse_caar_cooldown_steps": (
            args.caar_ls_reverse_caar_cooldown_steps
        ),

        "caar_ls_road_topology_adaptive_cooldown_enabled": (
            args.caar_ls_road_topology_adaptive_cooldown_enabled
        ),

        "caar_ls_road_open4_threshold": (
            args.caar_ls_road_open4_threshold
        ),

        "caar_ls_road_dense_obstacle_threshold": (
            args.caar_ls_road_dense_obstacle_threshold
        ),

        "caar_ls_road_reverse_caar_cooldown_steps": (
            args.caar_ls_road_reverse_caar_cooldown_steps
        ),

        "caar_ls_road_caar_only_density_threshold": (
            args.caar_ls_road_caar_only_density_threshold
        ),

        "notau_weights_path": args.notau_weights_path,

        "dhc_weights_path": args.dhc_weights_path,

        "dcc_weights_path": args.dcc_weights_path,

        "scrimp_weights_path": args.scrimp_weights_path,

        "epom_weights_path": args.epom_weights_path,

        "hybrid_mode": (
            hybrid_contract["hybrid_mode"]
            if hybrid_contract is not None
            else None
        ),

        "hybrid_components": (
            hybrid_contract["hybrid_components"]
            if hybrid_contract is not None
            else None
        ),

        "hybrid_action_policy": (
            hybrid_contract["action_policy"]
            if hybrid_contract is not None
            else None
        ),

        "hybrid_guide_algorithm": (
            hybrid_contract["guide_algorithm"]
            if hybrid_contract is not None
            else None
        ),

        "hybrid_contract": hybrid_contract,

        "integrity": integrity_metadata,

        "cache_algorithms_requested": algorithm_cache["requested"],

        "cache_algorithms_effective_by_algorithm": (
            algorithm_cache["effective_by_algorithm"]
        ),

        "cache_algorithms_exceptions": algorithm_cache["exceptions"],

        "map_list": str(map_list_path) if map_list_path is not None else None,

        "trim_border": args.trim_border,

    }

    print("Configuration")

    print(f"  algorithms: {', '.join(algorithms)}")

    print(f"  agent_counts: {agent_counts[0]}..{agent_counts[-1]} ({len(agent_counts)} values)")

    print(f"  maps: {', '.join(maps.values())}")

    print(f"  obs_radius: {args.obs_radius if args.obs_radius is not None else 'default'}")

    print(f"  max_steps: {args.max_steps} | seeds: {', '.join(str(seed) for seed in seeds)} | animate: {args.animate}")

    print(f"  on_target: {args.on_target} | collision: {args.collision_system}")

    results, elapsed = run_experiments(tasks, args.workers)

    metadata["finished_at"] = datetime.now().isoformat(timespec="seconds")

    metadata["total_elapsed_seconds"] = elapsed


    print(f"\nTotal elapsed: {format_duration(elapsed)}")


    if args.save:

        save_results(results, metadata, args.output_dir, args.output)

    failed = [result for result in results if result.get("error")]

    if failed:

        raise RuntimeError(

            f"{len(failed)} of {len(results)} experiments failed; "

            f"first error: {failed[0]['error']}"

        )



if __name__ == "__main__":

    main()
