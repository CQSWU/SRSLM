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


# Methods that require explicit artifacts or distinct evaluation protocols
# remain opt-in so the default/"all" batch stays unambiguous.
DEFAULT_ALGORITHMS = ("RePlan", "AORePlan")
ALGORITHM_ALIASES = {
    "replan": "RePlan",
    "aoreplan": "AORePlan",
    "noreweight": "NoReweight",
    "no-reweight": "NoReweight",
    "direct": "Direct",
    "caar": "CAAR",
    "srslm": "SRSLM",
    "srslm-nowaitdetect": "SRSLM-NoWaitDetect",
    "srslm-no-wait-detect": "SRSLM-NoWaitDetect",
    "srslm-waitdetectonly": "SRSLM-WaitDetectOnly",
    "srslm-wait-detect-only": "SRSLM-WaitDetectOnly",
    "epom-lifelong-ft": "EPOM-Lifelong-FT",
    "epom_lifelong_ft": "EPOM-Lifelong-FT",
}


ALGORITHM_COLUMN_WIDTH = max(13, *(len(algorithm) for algorithm in SUPPORTED_ALGORITHMS))


_worker_algo_cache = {}



def canonical_algorithm_name(value):

    canonical = ALGORITHM_ALIASES.get(value.strip().lower())
    return canonical if canonical in SUPPORTED_ALGORITHMS else None






def epom_lifelong_result_manifest(results, algorithms):
    """Pin the EPOM-L artifact used by the public lifelong baseline."""
    selected = (
        ["EPOM-Lifelong-FT"]
        if "EPOM-Lifelong-FT" in algorithms
        else []
    )
    if not selected:
        return None
    error_rows = [
        row
        for row in results
        if row.get("algorithm") in selected and row.get("error")
    ]
    if error_rows:
        return {
            "validated": False,
            "algorithms": selected,
            "error_rows": len(error_rows),
            "shared_artifact": None,
        }

    manifests = []
    rows_by_algorithm = {algorithm: 0 for algorithm in selected}
    for row in results:
        algorithm = row.get("algorithm")
        if algorithm not in rows_by_algorithm or row.get("error"):
            continue
        rows_by_algorithm[algorithm] += 1
        provenance = row.get("model_provenance") or {}
        epom = provenance
        if epom.get("artifact_profile") != "lifelong_finetuned":
            raise RuntimeError(
                f"{algorithm} row is missing lifelong_finetuned EPOM provenance"
            )
        manifests.append(
            {
                key: epom.get(key)
                for key in (
                    "artifact_profile",
                    "weights_path",
                    "config_path",
                    "config_sha256",
                    "checkpoint_path",
                    "checkpoint_name",
                    "checkpoint_size",
                    "checkpoint_sha256",
                    "selection_rule",
                    "training_protocol",
                    "source_encoder_custom",
                    "inference_encoder_custom",
                )
            }
        )

    missing = [
        algorithm for algorithm, count in rows_by_algorithm.items() if count == 0
    ]
    if missing:
        raise RuntimeError(
            "No successful rows were available for EPOM-L manifest: "
            + ", ".join(missing)
        )
    unique = {json.dumps(item, sort_keys=True) for item in manifests}
    if len(unique) != 1:
        raise RuntimeError(
            "EPOM-L rows did not use one identical artifact."
        )
    return {
        "validated": True,
        "algorithms": selected,
        "shared_artifact": json.loads(next(iter(unique))),
        "rows_by_algorithm": rows_by_algorithm,
    }




def srslm_contract_metadata(algorithms):
    """Describe the fixed AORePlan-wait bypass and learned Switcher."""
    if "SRSLM" not in algorithms:
        return None
    return {
        "strategy_kind": "hybrid_switching",
        "hybrid_mode": "aoreplan_wait_bypass_switcher_v3",
        "branch_algorithms": ["CAAR", "AORePlan"],
        "hybrid_components": {
            "learning_branch": "CAAR",
            "planning_branch": "AORePlan",
            "selector": "Switcher",
        },
        "action_policy": "CAAR-or-AORePlan",
        "guide_algorithm": "AORePlan",
        "deployment": {
            "wait_rule": "aoreplan_wait_directly_uses_caar",
            "switcher_scope": "aoreplan_nonwait_only",
            "switcher_output": "two_branch_categorical_logits",
            "selection": "softmax_sampling",
            "joint_conflict_prediction_enabled": False,
            "simulator_collision_system": "block_both",
        },
    }




def srslm_ablation_contract_metadata(algorithms):
    """Describe one current-V3 wait-detector ablation."""
    selected = [
        name
        for name in ("SRSLM-NoWaitDetect", "SRSLM-WaitDetectOnly")
        if name in algorithms
    ]
    if not selected:
        return None
    if len(selected) != 1 or "SRSLM" in algorithms:
        raise ValueError(
            "Run each SRSLM wait-detector ablation in a separate invocation "
            "so its artifact contract stays unambiguous."
        )
    algorithm = selected[0]
    if algorithm == "SRSLM-NoWaitDetect":
        return {
            "strategy_kind": "hybrid_switching_ablation",
            "algorithm": algorithm,
            "hybrid_mode": "all_state_switcher_v3",
            "branch_algorithms": ["CAAR", "AORePlan"],
            "hybrid_components": {
                "learning_branch": "CAAR",
                "planning_branch": "AORePlan",
                "selector": "AllStateSwitcher",
            },
            "action_policy": "CAAR-or-AORePlan",
            "guide_algorithm": "AORePlan",
            "deployment": {
                "wait_rule": "disabled",
                "switcher_scope": "all_states",
                "switcher_output": "two_branch_categorical_logits",
                "selection": "softmax_sampling",
                "joint_conflict_prediction_enabled": False,
                "simulator_collision_system": "block_both",
            },
        }
    return {
        "strategy_kind": "hybrid_switching_ablation",
        "algorithm": algorithm,
        "hybrid_mode": "aoreplan_wait_detect_only_v3",
        "branch_algorithms": ["CAAR", "AORePlan"],
        "hybrid_components": {
            "learning_branch": "CAAR",
            "planning_branch": "AORePlan",
            "selector": "deterministic_wait_detector",
        },
        "action_policy": "CAAR-on-wait-otherwise-AORePlan",
        "guide_algorithm": "AORePlan",
        "deployment": {
            "wait_rule": "aoreplan_wait_directly_uses_caar",
            "switcher_scope": "none",
            "selection": "deterministic",
            "learned_switcher_called": False,
            "joint_conflict_prediction_enabled": False,
            "simulator_collision_system": "block_both",
        },
    }


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
    current = _find_weight_run_dir(
        root / "weights" / "EPOM-TracePaperConvDirectCorrection-R5-500m"
    )
    if current is None:
        raise FileNotFoundError(
            "No current CAAR checkpoint was found under "
            f"{root / 'weights' / 'EPOM-TracePaperConvDirectCorrection-R5-500m'}. "
            "Train CAAR or pass --caar-weights-path explicitly."
        )
    return str(current)


def _find_no_reweight_weights(main_dir):
    root = Path(main_dir).resolve()
    candidate = _find_weight_run_dir(root / "weights" / "NoReweight-block-1b")
    if candidate is not None:
        return str(candidate)
    return str(
        root
        / "weights"
        / "NoReweight-block-1b"
        / "NoReweight-Block-R5-1B"
    )


def _find_switcher_weights(main_dir):
    root = Path(main_dir).resolve()
    candidate = _find_weight_run_dir(
        root / "weights" / "SRSLM-switcher-wait-aware-caar-100m"
    )
    if candidate is None:
        raise FileNotFoundError(
            "No current wait-aware SRSLM Switcher checkpoint was found under "
            f"{root / 'weights' / 'SRSLM-switcher-wait-aware-caar-100m'}. "
            "Train the Switcher or pass --switcher-weights-path explicitly."
        )
    return str(candidate)


def _find_no_wait_detect_switcher_weights(main_dir):
    root = Path(main_dir).resolve()
    candidate = _find_weight_run_dir(
        root / "weights" / "SRSLM-switcher-caar-nowait-100m"
    )
    if candidate is None:
        raise FileNotFoundError(
            "No independently trained SRSLM-NoWaitDetect checkpoint was "
            "found under "
            f"{root / 'weights' / 'SRSLM-switcher-caar-nowait-100m'}. "
            "Train the all-state policy or pass "
            "--no-wait-detect-switcher-weights-path explicitly."
        )
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


def static_astar_metric_metadata():
    """Describe AORePlan's static-A* query statistic."""

    return {
        "version": "aoreplan_static_astar_query_v3",
        "scope": "standalone AORePlan in lifelong evaluation",
        "definition": (
            "the reported rate counts static-map A* checks triggered by "
            "reverse dynamic movement proposals"
        ),
        "static_astar_query_rate_numerator": (
            "static-map A* queries attached to raw dynamic RePlan "
            "non-wait movement proposals"
        ),
        "static_astar_query_rate_denominator": (
            "raw dynamic RePlan non-wait movement proposals before AORePlan "
            "substitution"
        ),
        "no_path_fallback_count": (
            "dynamic A*/BestMove failures handled by the original 50% wait, "
            "50% obstacle-screened random-direction fallback"
        ),
    }


def runtime_metric_metadata():
    """Describe the wall-clock fields emitted by the experiment runner."""

    return {
        "version": "end_to_end_episode_wall_v1",
        "run_time_seconds": (
            "per-run worker wall time around run_algorithm; includes environment "
            "construction, reset, policy actions, environment steps, and metric "
            "collection, but excludes algorithm construction and checkpoint loading"
        ),
        "total_elapsed_seconds": (
            "wall time for the complete ProcessPool batch at the configured "
            "worker count"
        ),
        "elapsed_since_start_seconds": (
            "cumulative batch time when a result completed; this is not an "
            "individual run duration"
        ),
    }


def contention_metric_metadata():
    """Describe how many active agents participate in traffic contention."""

    return {
        "version": "agent_contention_participation_v1",
        "definition": (
            "an active agent participates when it is one of multiple movement "
            "proposals for the same destination, is part of an edge swap, or "
            "proposes or occupies a destination that is not vacated during "
            "the environment step"
        ),
        "contention_rate_numerator": (
            "active agent-steps participating in at least one contention event"
        ),
        "contention_rate_denominator": "all active agent-steps",
        "counting_rule": (
            "each active agent is counted at most once per environment step"
        ),
    }


def vertex_flow_metric_metadata():
    """Describe the one-step vertex flow cost used for evaluation."""

    return {
        "version": "submitted_one_step_vertex_flow_pairs_v1",
        "definition": (
            "before collision resolution, valid non-wait movement proposals "
            "are grouped by destination; a destination with n incoming "
            "proposals contributes n*(n-1)/2 pairwise vertex-flow cost"
        ),
        "vertex_flow_pair_count": (
            "sum of pairwise same-destination movement proposals over steps"
        ),
        "vertex_flow_move_denominator": (
            "submitted non-wait proposals whose destination is an in-bounds "
            "free cell"
        ),
        "vertex_flow_pair_cost_per_move": (
            "vertex_flow_pair_count divided by vertex_flow_move_denominator"
        ),
        "capture_point": "submitted actions before environment collision resolution",
    }


def _project_path(main_dir, value):
    path = Path(value)
    if not path.is_absolute():
        path = Path(main_dir) / path
    return path.resolve()




_EPISODE_FRESH_ALGORITHMS = frozenset(
    (
        "CAAR",
        "NoReweight",
        "Direct",
        "SRSLM-NoWaitDetect",
        "SRSLM-WaitDetectOnly",
        "SRSLM",
    )
)


def should_cache_algorithm(algorithm, requested):
    """Return whether an inference instance is safe to reuse across episodes."""
    canonical = canonical_algorithm_name(algorithm) or algorithm
    return bool(requested) and canonical not in _EPISODE_FRESH_ALGORITHMS


def cache_algorithm_metadata(algorithms, requested):
    """Describe requested and effective per-algorithm instance caching."""
    requested = bool(requested)
    effective_by_algorithm = {
        algorithm: should_cache_algorithm(algorithm, requested)
        for algorithm in algorithms
    }
    exceptions = {
        algorithm: "disabled_to_preserve_episode_fresh_policy_state"
        for algorithm, effective in effective_by_algorithm.items()
        if requested and not effective
    }
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


def srslm_integrity_metadata(
    args,
    map_list_sha256=None,
    map_registry_sha256=None,
):
    """Hash the two frozen branches, Switcher, and current routing code."""

    root = Path(args.main_dir).resolve()
    code_root = Path(__file__).resolve().parent
    caar_weights = _project_path(
        root,
        args.caar_weights_path or _find_caar_weights(root),
    )
    switcher_weights = _project_path(
        root,
        args.switcher_weights_path or _find_switcher_weights(root),
    )
    caar_checkpoint = _latest_caar_checkpoint(caar_weights)
    switcher_checkpoint = _latest_caar_checkpoint(switcher_weights)
    files = {
        "caar_config": caar_weights / "config.json",
        "caar_checkpoint": caar_checkpoint,
        "switcher_config": switcher_weights / "config.json",
        "switcher_checkpoint": switcher_checkpoint,
        "run_experiments.py": code_root / "run_experiments.py",
        "agents/caar.py": code_root / "agents/caar.py",
        "agents/srslm.py": code_root / "agents/srslm.py",
        "agents/switcher.py": code_root / "agents/switcher.py",
        "agents/switcher_core.py": code_root / "agents/switcher_core.py",
        "agents/reverse_metrics.py": code_root / "agents/reverse_metrics.py",
        "learning/encoder.py": code_root / "learning/encoder.py",
        "learning/switcher_actor_critic.py": (
            code_root / "learning/switcher_actor_critic.py"
        ),
        "planning/ao_replan_algo.py": code_root / "planning/ao_replan_algo.py",
        "planning/aoreplan_branch.py": (
            code_root / "planning/aoreplan_branch.py"
        ),
        "pomapf_env/switcher_env.py": code_root / "pomapf_env/switcher_env.py",
    }
    if getattr(args, "map_list", None):
        files["map_list"] = _project_path(root, args.map_list)
        files["map_registry"] = _project_path(root, "maps/eval.yaml")
    missing = [f"{label}={path}" for label, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot create SRSLM integrity metadata; missing " + ", ".join(missing)
        )
    hashes = {label: _sha256_file(path) for label, path in files.items()}
    aggregate = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "strategy_kind": "hybrid_switching",
        "hybrid_mode": "aoreplan_wait_bypass_switcher_v3",
        "caar_weights_path": str(caar_weights),
        "switcher_weights_path": str(switcher_weights),
        "caar_checkpoint_sha256": hashes["caar_checkpoint"],
        "switcher_checkpoint_sha256": hashes["switcher_checkpoint"],
        "artifact_sha256": hashes,
        "aggregate_sha256": aggregate,
        "map_list_sha256": (
            map_list_sha256
            if map_list_sha256 is not None
            else hashes.get("map_list")
        ),
        "map_registry_sha256": map_registry_sha256,
    }




def srslm_ablation_integrity_metadata(
    args,
    algorithm,
    map_list_sha256=None,
    map_registry_sha256=None,
):
    """Bind a V3 wait-detector ablation to its exact code and artifacts."""
    root = Path(args.main_dir).resolve()
    code_root = Path(__file__).resolve().parent
    caar_weights = _project_path(
        root,
        args.caar_weights_path or _find_caar_weights(root),
    )
    caar_checkpoint = _latest_caar_checkpoint(caar_weights)
    files = {
        "caar_config": caar_weights / "config.json",
        "caar_checkpoint": caar_checkpoint,
        "run_experiments.py": code_root / "run_experiments.py",
        "agents/caar.py": code_root / "agents/caar.py",
        "agents/srslm_ablation.py": code_root / "agents/srslm_ablation.py",
        "agents/switcher_core.py": code_root / "agents/switcher_core.py",
        "agents/reverse_metrics.py": code_root / "agents/reverse_metrics.py",
        "planning/ao_replan_algo.py": code_root / "planning/ao_replan_algo.py",
        "planning/aoreplan_branch.py": code_root / "planning/aoreplan_branch.py",
    }
    switcher_weights = None
    switcher_checkpoint = None
    if algorithm == "SRSLM-NoWaitDetect":
        switcher_weights = _project_path(
            root,
            args.no_wait_detect_switcher_weights_path
            or _find_no_wait_detect_switcher_weights(root),
        )
        switcher_checkpoint = _latest_caar_checkpoint(switcher_weights)
        files.update(
            {
                "switcher_config": switcher_weights / "config.json",
                "switcher_checkpoint": switcher_checkpoint,
                "agents/switcher.py": code_root / "agents/switcher.py",
                "learning/encoder.py": code_root / "learning/encoder.py",
                "learning/switcher_actor_critic.py": (
                    code_root / "learning/switcher_actor_critic.py"
                ),
            }
        )
    if getattr(args, "map_list", None):
        files["map_list"] = _project_path(root, args.map_list)
    missing = [
        f"{label}={path}"
        for label, path in files.items()
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Cannot create SRSLM ablation integrity metadata; missing "
            + ", ".join(missing)
        )
    hashes = {label: _sha256_file(path) for label, path in files.items()}
    aggregate = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    mode = (
        "all_state_switcher_v3"
        if algorithm == "SRSLM-NoWaitDetect"
        else "aoreplan_wait_detect_only_v3"
    )
    return {
        "strategy_kind": "hybrid_switching_ablation",
        "algorithm": algorithm,
        "hybrid_mode": mode,
        "caar_weights_path": str(caar_weights),
        "caar_checkpoint_sha256": hashes["caar_checkpoint"],
        "switcher_weights_path": (
            str(switcher_weights) if switcher_weights is not None else None
        ),
        "switcher_checkpoint_sha256": (
            hashes.get("switcher_checkpoint")
        ),
        "artifact_sha256": hashes,
        "aggregate_sha256": aggregate,
        "map_list_sha256": (
            map_list_sha256
            if map_list_sha256 is not None
            else hashes.get("map_list")
        ),
        "map_registry_sha256": map_registry_sha256,
    }

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

        max_planning_steps=max_planning_steps,

        seed=seed,

    )


def build_algorithm(
    algo_name,
    main_dir,
    seed,
    caar_weights_path=None,
    switcher_weights_path=None,
    no_wait_detect_switcher_weights_path=None,
    no_reweight_weights_path=None,
    epom_weights_path=None,
):
    algo_name = canonical_algorithm_name(algo_name) or algo_name

    if algo_name == "RePlan":
        from agents.replan import RePlan

        return RePlan(_replan_cfg(seed))

    if algo_name == "AORePlan":
        from agents.ao_replan import AORePlan

        return AORePlan(_ao_replan_cfg(seed))

    if algo_name in ("SRSLM-NoWaitDetect", "SRSLM-WaitDetectOnly"):
        from agents.caar import CAARConfig
        from agents.srslm_ablation import (
            SRSLMNoWaitDetect,
            SRSLMNoWaitDetectConfig,
            SRSLMWaitDetectOnly,
            SRSLMWaitDetectOnlyConfig,
        )

        caar_config = CAARConfig(
            path_to_weights=str(
                _project_path(
                    main_dir,
                    caar_weights_path or _find_caar_weights(main_dir),
                )
            ),
            checkpoint_kind="latest",
            device="auto",
        )
        if algo_name == "SRSLM-WaitDetectOnly":
            return SRSLMWaitDetectOnly(
                SRSLMWaitDetectOnlyConfig(
                    caar=caar_config,
                    seed=seed,
                )
            )

        from agents.switcher import AllStateSwitcherConfig

        return SRSLMNoWaitDetect(
            SRSLMNoWaitDetectConfig(
                caar=caar_config,
                switcher=AllStateSwitcherConfig(
                    path_to_weights=str(
                        _project_path(
                            main_dir,
                            no_wait_detect_switcher_weights_path
                            or _find_no_wait_detect_switcher_weights(main_dir),
                        )
                    ),
                    checkpoint_kind="auto",
                    device="auto",
                    deterministic=False,
                ),
                seed=seed,
            )
        )

    if algo_name == "SRSLM":
        from agents.srslm import SRSLM, SRSLMConfig
        from agents.switcher import SwitcherConfig

        return SRSLM(
            SRSLMConfig(
                switcher=SwitcherConfig(
                    path_to_weights=str(
                        _project_path(
                            main_dir,
                            switcher_weights_path
                            or _find_switcher_weights(main_dir),
                        )
                    ),
                    checkpoint_kind="auto",
                    device="auto",
                    deterministic=False,
                ),
                seed=seed,
            ),
            project_root=Path(main_dir).resolve(),
        )

    if algo_name == "EPOM-Lifelong-FT":
        from agents.epom import EPOM, EPOMConfig

        if not epom_weights_path:
            raise ValueError(
                "EPOM-Lifelong-FT requires an explicit --epom-weights-path."
            )
        return EPOM(
            EPOMConfig(
                path_to_weights=epom_weights_path,
                seed=seed,
                device="auto",
                artifact_profile="lifelong_finetuned",
            )
        )

    if algo_name == "NoReweight":
        from agents.caar import NoReweight, NoReweightConfig

        weights_path = (
            no_reweight_weights_path or _find_no_reweight_weights(main_dir)
        )
        return NoReweight(
            NoReweightConfig(
                path_to_weights=weights_path,
                seed=seed,
                checkpoint_kind="latest",
                device="auto",
            )
        )

    if algo_name == "Direct":
        from agents.direct import Direct, DirectConfig

        weights_path = (
            no_reweight_weights_path or _find_no_reweight_weights(main_dir)
        )
        return Direct(
            DirectConfig(
                path_to_weights=weights_path,
                name="Direct",
                pressure_scale=1.0,
                seed=seed,
                checkpoint_kind="latest",
                device="auto",
            )
        )

    if algo_name == "CAAR":
        from agents.caar import CAAR

        caar_path = caar_weights_path or _find_caar_weights(main_dir)
        return CAAR(_caar_cfg(caar_path, seed))

    raise ValueError(f"Unsupported algorithm: {algo_name}")


def validate_srslm_stats(stats):
    """Reject an SRSLM episode that violates the fixed routing contract."""

    required = (
        "hybrid_mode",
        "switch_pair",
        "switcher_training",
        "value_predictor_loaded",
        "switcher_feature_schema",
        "selector_kind",
        "switcher_decision_scope",
        "joint_conflict_prediction_enabled",
        "total_action_count",
        "switcher_choice_count",
        "switcher_model_choice_count",
        "selected_ao_count",
        "switcher_model_selected_ao_count",
        "executed_ao_count",
        "executed_caar_count",
        "aoreplan_wait_bypass_count",
        "branch_action_agreement_count",
        "static_astar_query_count",
        "aoreplan_commit_count",
        "switcher_checkpoint_sha256",
        "switcher_stochastic",
    )
    missing = [key for key in required if key not in stats]
    if missing:
        raise RuntimeError(
            "SRSLM diagnostics are incomplete: " + ", ".join(missing)
        )

    total = int(stats["total_action_count"])
    choices = int(stats["switcher_choice_count"])
    bypasses = int(stats["aoreplan_wait_bypass_count"])
    selected_ao = int(stats["selected_ao_count"])
    executed_ao = int(stats["executed_ao_count"])
    executed_caar = int(stats["executed_caar_count"])
    violations = []
    if stats["hybrid_mode"] != "aoreplan_wait_bypass_switcher_v3":
        violations.append("hybrid mode differs from the fixed SRSLM policy")
    if stats["switch_pair"] != ["CAAR", "AORePlan"]:
        violations.append("branch names are not CAAR/AORePlan")
    if stats["switcher_training"] != "PPO":
        violations.append("Switcher training method is not PPO")
    if stats["value_predictor_loaded"] is not False:
        violations.append("a retired value predictor was loaded")
    if stats["switcher_feature_schema"] != "srslm_switcher_state_v3":
        violations.append("Switcher state schema differs")
    if stats["selector_kind"] != "ppo_two_branch_categorical":
        violations.append("Switcher is not a two-branch categorical policy")
    if stats["switcher_decision_scope"] != "aoreplan_nonwait_only":
        violations.append("Switcher received states outside AORePlan moves")
    if stats["joint_conflict_prediction_enabled"] is not False:
        violations.append("retired joint-conflict prediction is active")
    if choices + bypasses != total:
        violations.append("Switcher choices and AORePlan-wait bypasses do not sum")
    if executed_ao + executed_caar != total:
        violations.append("executed branch counts do not sum")
    if selected_ao != executed_ao or selected_ao > choices:
        violations.append("selected and executed AORePlan counts disagree")
    if int(stats["switcher_model_choice_count"]) != choices:
        violations.append("Switcher model and router choice counts disagree")
    if int(stats["switcher_model_selected_ao_count"]) != selected_ao:
        violations.append("Switcher model and router AO counts disagree")
    if int(stats["aoreplan_commit_count"]) > total:
        violations.append("AORePlan commit count exceeds total actions")
    if int(stats["branch_action_agreement_count"]) > total:
        violations.append("branch agreement count exceeds total actions")
    digest = str(stats["switcher_checkpoint_sha256"])
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        violations.append("Switcher checkpoint SHA256 is invalid")
    if stats["switcher_stochastic"] is not True:
        violations.append("Switcher evaluation is not stochastic")
    for key in (
        "switcher_choice_rate",
        "selected_ao_rate",
        "executed_ao_rate",
        "aoreplan_wait_bypass_rate",
        "branch_action_agreement_rate",
        "switcher_sampled_ao_rate",
        "switcher_ao_probability_mean",
        "switcher_ao_probability_p05",
        "switcher_ao_probability_p95",
    ):
        value = stats.get(key)
        if value is None or not np.isfinite(value) or not 0.0 <= value <= 1.0:
            violations.append(f"{key} is not a probability")
    if violations:
        raise RuntimeError("SRSLM contract failed: " + "; ".join(violations))








def validate_srslm_ablation_stats(algorithm, stats):
    """Reject rows that do not implement the named V3 wait ablation."""
    common = (
        "hybrid_mode",
        "ablation_name",
        "switch_pair",
        "switcher_training",
        "value_predictor_loaded",
        "switcher_feature_schema",
        "selector_kind",
        "switcher_decision_scope",
        "wait_detection_enabled",
        "learned_switcher_called",
        "joint_conflict_prediction_enabled",
        "total_action_count",
        "switcher_choice_count",
        "switcher_model_choice_count",
        "selected_ao_count",
        "switcher_model_selected_ao_count",
        "executed_ao_count",
        "executed_caar_count",
        "aoreplan_wait_bypass_count",
        "branch_action_agreement_count",
        "static_astar_query_count",
        "aoreplan_commit_count",
        "switcher_stochastic",
    )
    missing = [key for key in common if key not in stats]
    if missing:
        raise RuntimeError(
            f"{algorithm} diagnostics are incomplete: " + ", ".join(missing)
        )
    total = int(stats["total_action_count"])
    choices = int(stats["switcher_choice_count"])
    selected_ao = int(stats["selected_ao_count"])
    executed_ao = int(stats["executed_ao_count"])
    executed_caar = int(stats["executed_caar_count"])
    bypasses = int(stats["aoreplan_wait_bypass_count"])
    violations = []
    if stats["ablation_name"] != algorithm:
        violations.append("ablation label differs")
    if stats["switch_pair"] != ["CAAR", "AORePlan"]:
        violations.append("branch names differ")
    if stats["value_predictor_loaded"] is not False:
        violations.append("a retired value predictor was loaded")
    if stats["switcher_feature_schema"] != "srslm_switcher_state_v3":
        violations.append("feature schema differs")
    if stats["joint_conflict_prediction_enabled"] is not False:
        violations.append("retired joint-conflict prediction is active")
    if executed_ao + executed_caar != total:
        violations.append("executed branch counts do not sum")
    if int(stats["aoreplan_commit_count"]) > total:
        violations.append("AORePlan commit count exceeds total actions")
    if int(stats["branch_action_agreement_count"]) > total:
        violations.append("branch agreement count exceeds total actions")

    if algorithm == "SRSLM-NoWaitDetect":
        if stats["hybrid_mode"] != "all_state_switcher_v3":
            violations.append("hybrid mode differs")
        if stats["switcher_training"] != "PPO":
            violations.append("Switcher training is not PPO")
        if stats["selector_kind"] != "ppo_two_branch_categorical":
            violations.append("selector is not categorical PPO")
        if stats["switcher_decision_scope"] != "all_states":
            violations.append("Switcher does not cover all states")
        if stats["wait_detection_enabled"] is not False:
            violations.append("wait detector is still enabled")
        if stats["learned_switcher_called"] is not True:
            violations.append("learned Switcher was not called")
        if choices != total or bypasses != 0:
            violations.append("all-state choice accounting differs")
        if selected_ao != executed_ao:
            violations.append("selected and executed AORePlan counts differ")
        if int(stats["switcher_model_choice_count"]) != total:
            violations.append("model did not receive every state")
        if int(stats["switcher_model_selected_ao_count"]) != selected_ao:
            violations.append("model/router AO counts differ")
        if stats["switcher_stochastic"] is not True:
            violations.append("all-state Switcher is not stochastic")
        digest = str(stats.get("switcher_checkpoint_sha256", ""))
        if len(digest) != 64 or any(
            char not in "0123456789abcdef" for char in digest
        ):
            violations.append("all-state checkpoint SHA256 is invalid")
    elif algorithm == "SRSLM-WaitDetectOnly":
        if stats["hybrid_mode"] != "aoreplan_wait_detect_only_v3":
            violations.append("hybrid mode differs")
        if stats["switcher_training"] != "none":
            violations.append("a learned Switcher was declared")
        if stats["selector_kind"] != "deterministic_wait_detect_only":
            violations.append("selector is not the wait detector")
        if stats["switcher_decision_scope"] != "none":
            violations.append("a learned decision scope was declared")
        if stats["wait_detection_enabled"] is not True:
            violations.append("wait detector is disabled")
        if stats["learned_switcher_called"] is not False:
            violations.append("learned Switcher was called")
        if choices or selected_ao:
            violations.append("Switcher choice counters are nonzero")
        if int(stats["switcher_model_choice_count"]) or int(
            stats["switcher_model_selected_ao_count"]
        ):
            violations.append("Switcher model counters are nonzero")
        if executed_caar != bypasses or executed_ao + bypasses != total:
            violations.append("deterministic wait routing differs")
        if stats["switcher_stochastic"] is not False:
            violations.append("deterministic ablation is marked stochastic")
    else:
        raise ValueError(f"Unknown SRSLM ablation {algorithm!r}.")

    for key in (
        "switcher_choice_rate",
        "selected_ao_rate",
        "executed_ao_rate",
        "aoreplan_wait_bypass_rate",
        "branch_action_agreement_rate",
    ):
        value = stats.get(key)
        if value is None or not np.isfinite(value) or not 0.0 <= value <= 1.0:
            violations.append(f"{key} is not a probability")
    if violations:
        raise RuntimeError(
            f"{algorithm} contract failed: " + "; ".join(violations)
        )





class _MoveFailureTracker:
    """Track movement outcomes and agent contention without changing actions."""

    METRIC_VERSION = "submitted_nonwait_no_position_change_v1"
    CONTENTION_METRIC_VERSION = "agent_contention_participation_v1"
    VERTEX_FLOW_METRIC_VERSION = "submitted_one_step_vertex_flow_pairs_v1"

    def __init__(self, moves, obstacle_mask=None):
        self.moves = tuple(
            tuple(int(value) for value in move) for move in moves
        )
        self.obstacle_mask = (
            None
            if obstacle_mask is None
            else np.asarray(obstacle_mask, dtype=bool).copy()
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
        self.contention_participant_count = 0
        self.contention_step_count = 0
        self.vertex_flow_pair_count = 0
        self.vertex_flow_move_denominator = 0
        self.vertex_flow_contested_destination_count = 0
        self.vertex_flow_step_count = 0
        self.vertex_flow_max_inflow = 0

    @staticmethod
    def _point(observation):
        value = observation["xy"]
        return int(value[0]), int(value[1])

    def _is_valid_flow_target(self, target):
        if self.obstacle_mask is None:
            return True
        x, y = target
        height, width = self.obstacle_mask.shape
        return (
            0 <= x < height
            and 0 <= y < width
            and not bool(self.obstacle_mask[x, y])
        )

    def capture(
        self,
        actions,
        observations,
        dones,
        infos,
        global_positions=None,
    ):
        positions = (
            [tuple(int(value) for value in position) for position in global_positions]
            if global_positions is not None
            else [self._point(observation) for observation in observations]
        )
        active = [
            not bool(dones[index])
            and bool(infos[index].get("is_active", True))
            for index in range(len(observations))
        ]
        attempts = []
        flow_attempts = []
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
            if self._is_valid_flow_target(target):
                flow_attempts.append((index, position, target))
        return {
            "positions": positions,
            "active": active,
            "active_count": sum(active),
            "wait_count": waits,
            "attempts": attempts,
            "flow_attempts": flow_attempts,
        }

    def commit(self, pending, observations, global_positions=None):
        after_positions = (
            [tuple(int(value) for value in position) for position in global_positions]
            if global_positions is not None
            else [self._point(observation) for observation in observations]
        )
        occupied = {}
        for index, position in enumerate(pending["positions"]):
            occupied.setdefault(position, set()).add(index)
        target_counts = {}
        for _, _, target in pending["attempts"]:
            target_counts[target] = target_counts.get(target, 0) + 1

        attempts_by_agent = {
            index: (before, target)
            for index, before, target in pending["attempts"]
        }
        contention_participants = set()

        target_groups = {}
        for index, _, target in pending["attempts"]:
            target_groups.setdefault(target, []).append(index)
        for group in target_groups.values():
            if len(group) > 1:
                contention_participants.update(group)

        flow_target_counts = {}
        for _, _, target in pending["flow_attempts"]:
            flow_target_counts[target] = flow_target_counts.get(target, 0) + 1
        vertex_flow_pairs = sum(
            count * (count - 1) // 2
            for count in flow_target_counts.values()
        )
        contested_flow_destinations = sum(
            count > 1 for count in flow_target_counts.values()
        )
        max_inflow = max(flow_target_counts.values(), default=0)

        position_to_agents = {}
        for index, position in enumerate(pending["positions"]):
            position_to_agents.setdefault(position, []).append(index)

        for index, before, target in pending["attempts"]:
            for other in position_to_agents.get(target, ()):
                if other == index:
                    continue
                other_attempt = attempts_by_agent.get(other)
                if other_attempt is not None and other_attempt[1] == before:
                    contention_participants.add(index)
                    if pending["active"][other]:
                        contention_participants.add(other)
                if after_positions[other] == pending["positions"][other]:
                    contention_participants.add(index)
                    if pending["active"][other]:
                        contention_participants.add(other)

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
        self.contention_participant_count += len(contention_participants)
        if contention_participants:
            self.contention_step_count += 1
        self.vertex_flow_pair_count += vertex_flow_pairs
        self.vertex_flow_move_denominator += len(pending["flow_attempts"])
        self.vertex_flow_contested_destination_count += (
            contested_flow_destinations
        )
        if vertex_flow_pairs:
            self.vertex_flow_step_count += 1
        self.vertex_flow_max_inflow = max(
            self.vertex_flow_max_inflow,
            max_inflow,
        )

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
            "contention_metric_version": self.CONTENTION_METRIC_VERSION,
            "contention_participant_count": self.contention_participant_count,
            "contention_step_count": self.contention_step_count,
            "contention_participation_rate": self._rate(
                self.contention_participant_count,
                self.active_agent_step_count,
            ),
            "contention_step_rate": self._rate(
                self.contention_step_count,
                self.environment_step_count,
            ),
            "vertex_flow_metric_version": self.VERTEX_FLOW_METRIC_VERSION,
            "vertex_flow_pair_count": self.vertex_flow_pair_count,
            "vertex_flow_move_denominator": self.vertex_flow_move_denominator,
            "vertex_flow_pair_cost_per_move": self._rate(
                self.vertex_flow_pair_count,
                self.vertex_flow_move_denominator,
            ),
            "vertex_flow_contested_destination_count": (
                self.vertex_flow_contested_destination_count
            ),
            "vertex_flow_step_count": self.vertex_flow_step_count,
            "vertex_flow_max_inflow": self.vertex_flow_max_inflow,
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
    agents_xy=None,
    targets_xy=None,
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

    if (agents_xy is None) != (targets_xy is None):
        raise ValueError(
            "Explicit placements require both agents_xy and targets_xy"
        )
    if agents_xy is not None:
        if len(agents_xy) != num_agents or len(targets_xy) != num_agents:
            raise ValueError(
                "Explicit placement count must equal num_agents"
            )
        gc_kwargs["agents_xy"] = [list(position) for position in agents_xy]
        gc_kwargs["targets_xy"] = [list(position) for position in targets_xy]

    grid_config = POMAPFConfig(**gc_kwargs)
    # A single-episode evaluator must observe the true final transition.
    # Auto-resetting inside env.step replaces final positions before the
    # movement tracker can commit step 512 and silently drops those failures.
    env = make_pomapf(
        grid_config=grid_config,
        with_animations=False,
        auto_reset=False,
    )
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

    def environment_grid():
        current = env
        while current is not None:
            grid = getattr(current, "grid", None)
            if grid is not None:
                return grid
            current = getattr(current, "env", None)
        raise RuntimeError("Could not locate the environment grid")

    def global_positions():
        grid = environment_grid()
        positions = getattr(grid, "positions_xy", None)
        if positions is None and hasattr(grid, "get_agents_xy"):
            positions = grid.get_agents_xy()
        if positions is not None:
            return np.asarray(positions, dtype=np.int64).copy()
        raise RuntimeError(
            "Contention participation requires global agent positions from "
            "the environment grid."
        )

    def global_obstacles():
        grid = environment_grid()
        obstacles = getattr(grid, "obstacles", None)
        if obstacles is not None:
            return np.asarray(obstacles, dtype=bool).copy()
        raise RuntimeError(
            "Vertex flow evaluation requires the environment obstacle grid."
        )

    try:
        observations, _ = env.reset()
        if agents_xy is not None:
            grid = environment_grid()
            expected_agents = np.asarray(agents_xy, dtype=np.int64)
            if (
                targets_xy
                and isinstance(targets_xy[0], (list, tuple))
                and targets_xy[0]
                and isinstance(targets_xy[0][0], (list, tuple))
            ):
                expected_targets = np.asarray(
                    [sequence[0] for sequence in targets_xy],
                    dtype=np.int64,
                )
            else:
                expected_targets = np.asarray(targets_xy, dtype=np.int64)
            actual_agents = np.asarray(
                grid.get_agents_xy(ignore_borders=True),
                dtype=np.int64,
            )
            actual_targets = np.asarray(
                grid.get_targets_xy(ignore_borders=True),
                dtype=np.int64,
            )
            expected_obstacles = np.asarray(
                [
                    [character == "#" for character in row]
                    for row in map_text.splitlines()
                ],
                dtype=bool,
            )
            actual_obstacles = np.asarray(
                grid.get_obstacles(ignore_borders=True),
                dtype=bool,
            )
            if not np.array_equal(actual_agents, expected_agents):
                raise RuntimeError(
                    "Environment changed the explicit start coordinates"
                )
            if not np.array_equal(actual_targets, expected_targets):
                raise RuntimeError(
                    "Environment changed the explicit goal coordinates"
                )
            if not np.array_equal(actual_obstacles, expected_obstacles):
                raise RuntimeError(
                    "Environment changed the explicit obstacle grid"
                )
            padding = int(grid_config.obs_radius)
            if not np.array_equal(
                np.asarray(grid.positions_xy, dtype=np.int64),
                expected_agents + padding,
            ):
                raise RuntimeError("Unexpected padded start coordinates")
            if not np.array_equal(
                np.asarray(grid.finishes_xy, dtype=np.int64),
                expected_targets + padding,
            ):
                raise RuntimeError("Unexpected padded goal coordinates")
        tracker = _MoveFailureTracker(
            grid_config.MOVES,
            obstacle_mask=global_obstacles(),
        )
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
                    global_positions=global_positions(),
                )
                observations, rewards, terminated, truncated, infos = env.step(
                    actions
                )
                tracker.commit(
                    pending,
                    observations,
                    global_positions=global_positions(),
                )
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

    # Recurrent and trace state must never cross episode boundaries.
    use_cache = should_cache_algorithm(
        algo_name,
        task.get("cache_algorithms", False),
    )

    algorithm_kwargs = {
        "caar_weights_path": task.get("caar_weights_path"),
        "switcher_weights_path": task.get("switcher_weights_path"),
        "no_wait_detect_switcher_weights_path": task.get(
            "no_wait_detect_switcher_weights_path"
        ),
        "no_reweight_weights_path": task.get("no_reweight_weights_path"),
        "epom_weights_path": task.get("epom_weights_path"),
    }
    cache_key = (
        algo_name,
        str(Path(main_dir).resolve()),
        seed,
        *algorithm_kwargs.values(),
    )


    try:

        if use_cache:

            if cache_key not in _worker_algo_cache:

                _worker_algo_cache[cache_key] = build_algorithm(

                    algo_name,

                    main_dir,

                    seed,
                    **algorithm_kwargs,
                )

            algo = _worker_algo_cache[cache_key]

        else:

            algo = build_algorithm(

                algo_name,

                main_dir,

                seed,
                **algorithm_kwargs,
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

            agents_xy=task.get("agents_xy"),

            targets_xy=task.get("targets_xy"),

        )

        run_time = time.time() - start



        on_target = task.get("on_target", "restart")
        is_restart = on_target == "restart"
        is_replan = algo_name in ("RePlan", "AORePlan")

        if hasattr(algo, "get_hybrid_stats"):
            hybrid_stats = algo.get_hybrid_stats()
        elif hasattr(algo, "get_switch_stats"):
            hybrid_stats = algo.get_switch_stats()
        else:
            hybrid_stats = {}
        if algo_name == "SRSLM":
            validate_srslm_stats(hybrid_stats)
        elif algo_name in (
            "SRSLM-NoWaitDetect",
            "SRSLM-WaitDetectOnly",
        ):
            validate_srslm_ablation_stats(algo_name, hybrid_stats)
        correction_stats = (
            algo.get_action_correction_stats()
            if hasattr(algo, "get_action_correction_stats")
            else {}
        )
        model_provenance = (
            algo.get_model_provenance()
            if hasattr(algo, "get_model_provenance")
            else None
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
        if task.get("task_id") is not None:
            result_record.update(
                {
                    "task_id": task["task_id"],
                    "family_id": task["family_id"],
                    "density_percent": task["density_percent"],
                    "placement_sha256": task["placement_sha256"],
                    "target_sequences_sha256": task.get(
                        "target_sequences_sha256"
                    ),
                }
            )
        result_record.update(
            {
                key: value
                for key, value in result.items()
                if key != "algorithm"
            }
        )
        result_record.update(correction_stats)

        if model_provenance is not None:
            result_record["model_provenance"] = model_provenance

        result_record.update(hybrid_stats)

        if is_restart:
            if is_replan:
                result_record["reverse_action_rate"] = getattr(algo, "reverse_action_rate", None)
                result_record["reverse_action_count"] = getattr(
                    algo,
                    "reverse_action_count",
                    None,
                )
                result_record["reverse_action_denominator"] = getattr(
                    algo,
                    "reverse_action_denominator",
                    None,
                )
                result_record["reverse_metric_version"] = getattr(
                    algo,
                    "reverse_metric_version",
                    None,
                )
                if algo_name == "AORePlan":
                    result_record["static_astar_query_count"] = getattr(
                        algo,
                        "static_astar_query_count",
                        None,
                    )
                    result_record["static_astar_query_denominator"] = getattr(
                        algo,
                        "static_astar_query_denominator",
                        None,
                    )
                    result_record["static_astar_query_rate"] = getattr(
                        algo,
                        "static_astar_query_rate",
                        None,
                    )
                    result_record["no_path_fallback_count"] = getattr(
                        algo,
                        "no_path_fallback_count",
                        None,
                    )


        return result_record

    except Exception as exc:

        import traceback


        error_record = {

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
        if task.get("task_id") is not None:
            error_record.update(
                {
                    "task_id": task["task_id"],
                    "family_id": task["family_id"],
                    "density_percent": task["density_percent"],
                    "placement_sha256": task["placement_sha256"],
                    "target_sequences_sha256": task.get(
                        "target_sequences_sha256"
                    ),
                }
            )
        return error_record



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


def _canonical_json_sha256(value):

    return hashlib.sha256(

        json.dumps(

            value,

            sort_keys=True,

            separators=(",", ":"),

        ).encode("utf-8")

    ).hexdigest()


def load_explicit_task_manifest_snapshot(
    path,
    maps,
    map_texts,
    expected_map_list_sha256=None,
):

    """Load exact per-map starts/goals without changing legacy task grids."""

    path = Path(path).resolve()

    payload_bytes = path.read_bytes()

    try:

        payload = json.loads(payload_bytes.decode("utf-8"))

    except (UnicodeDecodeError, json.JSONDecodeError) as error:

        raise ValueError(f"Could not parse task manifest {path}: {error}") from error

    if payload.get("artifact_kind") != "paired_density_task_manifest":

        raise ValueError(

            "--task-manifest must be a paired_density_task_manifest"

        )

    protocol_id = payload.get("protocol_id")

    if not isinstance(protocol_id, str) or not protocol_id:

        raise ValueError("--task-manifest is missing protocol_id")

    if (

        expected_map_list_sha256 is not None

        and payload.get("map_list_sha256") != expected_map_list_sha256

    ):

        raise ValueError(

            "--task-manifest map_list_sha256 does not match --map-list"

        )

    family_rows = payload.get("families")

    task_rows = payload.get("tasks")

    if not isinstance(family_rows, list) or not family_rows:

        raise ValueError("--task-manifest families must be a non-empty list")

    if not isinstance(task_rows, list) or not task_rows:

        raise ValueError("--task-manifest tasks must be a non-empty list")


    def normalize_positions(value, label):

        if not isinstance(value, list) or not value:

            raise ValueError(f"{label} must be a non-empty list")

        normalized = []

        for position in value:

            if (

                not isinstance(position, list)

                or len(position) != 2

                or any(

                    not isinstance(coordinate, int)

                    or isinstance(coordinate, bool)

                    for coordinate in position

                )

            ):

                raise ValueError(f"{label} contains an invalid coordinate")

            normalized.append([int(position[0]), int(position[1])])

        return normalized

    def normalize_target_sequences(value, label, initial_targets):
        if value is None:
            return None, None
        if not isinstance(value, list) or len(value) != len(initial_targets):
            raise ValueError(
                f"{label} must contain one sequence per initial target"
            )
        sequences = []
        for agent_index, sequence in enumerate(value):
            if not isinstance(sequence, list) or len(sequence) < 2:
                raise ValueError(
                    f"{label}[{agent_index}] must contain at least two goals"
                )
            normalized = normalize_positions(
                sequence,
                f"{label}[{agent_index}]",
            )
            if normalized[0] != initial_targets[agent_index]:
                raise ValueError(
                    f"{label}[{agent_index}] does not start at the initial goal"
                )
            sequences.append(normalized)
        return sequences, _canonical_json_sha256(sequences)


    families = {}

    for family in family_rows:

        if not isinstance(family, dict):

            raise ValueError("Every task-manifest family must be a mapping")

        family_id = family.get("family_id")

        if not isinstance(family_id, str) or not family_id:

            raise ValueError("Every family must have a non-empty family_id")

        if family_id in families:

            raise ValueError(f"Duplicate family_id in task manifest: {family_id}")

        agents_xy = normalize_positions(

            family.get("agents_xy"),

            f"{family_id}.agents_xy",

        )

        targets_xy = normalize_positions(

            family.get("targets_xy"),

            f"{family_id}.targets_xy",

        )

        target_sequences_xy, target_sequences_sha256 = (
            normalize_target_sequences(
                family.get("target_sequences_xy"),
                f"{family_id}.target_sequences_xy",
                targets_xy,
            )
        )
        if target_sequences_xy is not None and family.get(
            "target_sequences_sha256"
        ) != target_sequences_sha256:
            raise ValueError(f"{family_id}: target-sequence hash mismatch")

        if len(agents_xy) != len(targets_xy):

            raise ValueError(f"{family_id}: start/goal counts differ")

        starts = {tuple(position) for position in agents_xy}

        targets = {tuple(position) for position in targets_xy}

        if len(starts) != len(agents_xy) or len(targets) != len(targets_xy):

            raise ValueError(f"{family_id}: starts and goals must each be unique")

        if starts & targets:

            raise ValueError(f"{family_id}: starts and goals must be disjoint")

        placement_sha256 = _canonical_json_sha256(

            {"agents_xy": agents_xy, "targets_xy": targets_xy}

        )

        if family.get("placement_sha256") != placement_sha256:

            raise ValueError(f"{family_id}: placement hash mismatch")

        families[family_id] = {

            "agents_xy": agents_xy,

            "targets_xy": targets_xy,

            "target_sequences_xy": target_sequences_xy,

            "target_sequences_sha256": target_sequences_sha256,

            "placement_sha256": placement_sha256,

            "episode_seed": int(family["episode_seed"]),

        }


    task_specs = []

    seen_task_ids = set()

    seen_maps = set()

    for row in task_rows:

        if not isinstance(row, dict):

            raise ValueError("Every task-manifest task must be a mapping")

        task_id = row.get("task_id")

        family_id = row.get("family_id")

        map_name = row.get("map_name")

        if not isinstance(task_id, str) or not task_id:

            raise ValueError("Every explicit task must have a task_id")

        if task_id in seen_task_ids:

            raise ValueError(f"Duplicate task_id: {task_id}")

        if family_id not in families:

            raise ValueError(f"{task_id}: unknown family {family_id!r}")

        if map_name not in maps or map_name not in map_texts:

            raise ValueError(f"{task_id}: unknown map {map_name!r}")

        if map_name in seen_maps:

            raise ValueError(f"Explicit map appears in multiple tasks: {map_name}")

        family = families[family_id]

        num_agents = int(row["num_agents"])

        episode_seed = int(row["episode_seed"])

        density_percent = int(row["density_percent"])

        if num_agents != len(family["agents_xy"]):

            raise ValueError(f"{task_id}: num_agents does not match placements")

        if episode_seed != family["episode_seed"]:

            raise ValueError(f"{task_id}: episode seed differs from its family")

        if row.get("placement_sha256") != family["placement_sha256"]:

            raise ValueError(f"{task_id}: placement hash differs from its family")

        if family["target_sequences_xy"] is not None and row.get(
            "target_sequences_sha256"
        ) != family["target_sequences_sha256"]:

            raise ValueError(
                f"{task_id}: target-sequence hash differs from its family"
            )

        grid_rows = map_texts[map_name].splitlines()

        height = len(grid_rows)

        width = len(grid_rows[0])

        checked_positions = family["agents_xy"] + family["targets_xy"]
        if family["target_sequences_xy"] is not None:
            checked_positions += [
                coordinate
                for sequence in family["target_sequences_xy"]
                for coordinate in sequence
            ]
        for coordinate in checked_positions:

            first, second = coordinate

            if (

                first < 0

                or first >= height

                or second < 0

                or second >= width

                or grid_rows[first][second] != "."

            ):

                raise ValueError(

                    f"{task_id}: placement {coordinate} is not a free map cell"

                )

        task_specs.append(

            {

                "task_id": task_id,

                "family_id": family_id,

                "density_percent": density_percent,

                "map_name": map_name,

                "num_agents": num_agents,

                "seed": episode_seed,

                "agents_xy": family["agents_xy"],

                "targets_xy": (
                    family["target_sequences_xy"]
                    if family["target_sequences_xy"] is not None
                    else family["targets_xy"]
                ),

                "initial_targets_xy": family["targets_xy"],

                "target_sequences_sha256": family[
                    "target_sequences_sha256"
                ],

                "placement_sha256": family["placement_sha256"],

            }

        )

        seen_task_ids.add(task_id)

        seen_maps.add(map_name)

    if seen_maps != set(maps):

        missing = sorted(set(maps) - seen_maps)

        extra = sorted(seen_maps - set(maps))

        raise ValueError(

            "Task manifest and map list differ: "

            f"missing={missing[:3]}, extra={extra[:3]}"

        )

    if payload.get("task_count") != len(task_specs):

        raise ValueError("task_count does not match the explicit task list")

    return (

        task_specs,

        hashlib.sha256(payload_bytes).hexdigest(),

        protocol_id,

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
    explicit_task_specs=None,
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

    if explicit_task_specs is None:

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

            "switcher_weights_path": args.switcher_weights_path,

            "no_wait_detect_switcher_weights_path": (
                args.no_wait_detect_switcher_weights_path
            ),

            "no_reweight_weights_path": args.no_reweight_weights_path,

            "epom_weights_path": args.epom_weights_path,

            "cache_algorithms": should_cache_algorithm(
                algorithm,
                args.cache_algorithms,
            ),

        }

        for algorithm in algorithms

        for map_type, map_name, map_text, map_source in map_items

        for num_agents in agent_counts

        for seed in seeds

    ]

    if custom_map is not None or map_texts is None:

        raise ValueError(

            "Explicit task manifests require a YAML --map-list snapshot"

        )

    tasks = []

    for algorithm in algorithms:

        for spec in explicit_task_specs:

            map_name = spec["map_name"]

            tasks.append(

                {

                    "algorithm": algorithm,

                    "map_type": "paired_density",

                    "map_name": map_name,

                    "map_text": map_texts[map_name],

                    "map_source": None,

                    "num_agents": spec["num_agents"],

                    "obs_radius": args.obs_radius,

                    "max_steps": args.max_steps,

                    "seed": spec["seed"],

                    "animate": args.animate,

                    "main_dir": args.main_dir,

                    "on_target": args.on_target,

                    "collision_system": args.collision_system,

                    "caar_weights_path": args.caar_weights_path,

                    "switcher_weights_path": args.switcher_weights_path,

                    "no_wait_detect_switcher_weights_path": (
                        args.no_wait_detect_switcher_weights_path
                    ),

                    "no_reweight_weights_path": (
                        args.no_reweight_weights_path
                    ),

                    "epom_weights_path": args.epom_weights_path,

                    "cache_algorithms": should_cache_algorithm(
                        algorithm,
                        args.cache_algorithms,
                    ),

                    "task_id": spec["task_id"],

                    "family_id": spec["family_id"],

                    "density_percent": spec["density_percent"],

                    "agents_xy": spec["agents_xy"],

                    "targets_xy": spec["targets_xy"],

                    "initial_targets_xy": spec.get("initial_targets_xy"),

                    "target_sequences_sha256": spec.get(
                        "target_sequences_sha256"
                    ),

                    "placement_sha256": spec["placement_sha256"],

                }

            )

    return tasks



def format_duration(seconds):

    if seconds < 60:

        return f"{seconds:.1f}s"

    minutes, rem = divmod(seconds, 60)

    if minutes < 60:

        return f"{int(minutes)}m{int(rem):02d}s"

    hours, minutes = divmod(minutes, 60)

    return f"{int(hours)}h{int(minutes):02d}m"



def _journal_task_key(value):

    """Return the stable identity of one experiment task/result row."""

    try:

        return (

            str(value["algorithm"]),

            str(value["map_name"]),

            int(value["num_agents"]),

            int(value["seed"]),

            str(value.get("task_id") or ""),

        )

    except (KeyError, TypeError, ValueError) as error:

        raise ValueError("Experiment journal entry has an invalid task key") from error


def _append_journal_record(path, record):

    encoded = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode(

        "utf-8"

    )

    with Path(path).open("ab", buffering=0) as stream:

        stream.write(encoded)

        os.fsync(stream.fileno())


def _initialize_result_journal(path, contract, total):

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    record = {

        "record_type": "header",

        "schema": "experiment_result_journal_v1",

        "contract_sha256": contract,

        "expected_tasks": int(total),

    }

    encoded = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode(

        "utf-8"

    )

    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)

    try:

        os.write(descriptor, encoded)

        os.fsync(descriptor)

    finally:

        os.close(descriptor)


def _load_result_journal(path, contract, tasks, *, repair_final_record=False):

    path = Path(path)

    raw = path.read_bytes()

    if raw and not raw.endswith(b"\n"):

        last_newline = raw.rfind(b"\n")

        if not repair_final_record or last_newline < 0:

            raise ValueError("Result journal has a truncated final record")

        raw = raw[: last_newline + 1]

        temporary = path.with_name(f".{path.name}.{os.getpid()}.repair")

        with temporary.open("wb") as stream:

            stream.write(raw)

            stream.flush()

            os.fsync(stream.fileno())

        os.replace(temporary, path)

    lines = raw.splitlines()

    if not lines:

        raise ValueError("Result journal is empty")

    try:

        header = json.loads(lines[0])

    except json.JSONDecodeError as error:

        raise ValueError("Result journal header is malformed") from error

    if header != {

        "record_type": "header",

        "schema": "experiment_result_journal_v1",

        "contract_sha256": contract,

        "expected_tasks": len(tasks),

    }:

        raise ValueError("Result journal contract/header differs from this run")

    expected_keys = {_journal_task_key(task) for task in tasks}

    if len(expected_keys) != len(tasks):

        raise ValueError("Experiment tasks are not uniquely journalable")

    successful = {}

    ordered = []

    for line_number, line in enumerate(lines[1:], 2):

        try:

            record = json.loads(line)

        except json.JSONDecodeError as error:

            raise ValueError(

                f"Result journal record {line_number} is malformed"

            ) from error

        if record.get("record_type") != "result":

            raise ValueError(f"Unexpected result journal record {line_number}")

        result = record.get("result")

        if not isinstance(result, dict):

            raise ValueError(f"Result journal record {line_number} has no result")

        key = _journal_task_key(result)

        if record.get("task_key") != list(key) or key not in expected_keys:

            raise ValueError(f"Result journal record {line_number} has a foreign task")

        if result.get("error"):

            # Failed attempts remain as audit events but are safe to retry.

            continue

        if key in successful:

            raise ValueError(f"Result journal repeats successful task {key}")

        successful[key] = result

        ordered.append(result)

    return ordered


def run_experiments(

    tasks,

    workers,

    *,

    result_journal=None,

    journal_contract=None,

    resume_result_journal=False,

):

    results = []

    total = len(tasks)

    start_time = time.time()

    elapsed_offset = 0.0

    journal_path = Path(result_journal) if result_journal is not None else None

    if journal_path is not None:

        if not isinstance(journal_contract, str) or len(journal_contract) != 64 or any(

            char not in "0123456789abcdef" for char in journal_contract

        ):

            raise ValueError("A lowercase SHA256 --result-journal-contract is required")

        if journal_path.exists():

            if not resume_result_journal:

                raise FileExistsError(

                    f"Result journal already exists without resume permission: {journal_path}"

                )

            results = _load_result_journal(

                journal_path,

                journal_contract,

                tasks,

                repair_final_record=True,

            )

            elapsed_offset = max(

                (float(row.get("elapsed_since_start_seconds", 0.0)) for row in results),

                default=0.0,

            )

        else:

            _initialize_result_journal(journal_path, journal_contract, total)

        completed_keys = {_journal_task_key(row) for row in results}

        tasks_to_run = [

            task for task in tasks if _journal_task_key(task) not in completed_keys

        ]

    else:

        if resume_result_journal:

            raise ValueError("--resume-result-journal requires --result-journal")

        tasks_to_run = list(tasks)


    print(f"Starting experiments: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(

        f"Total: {total} | Workers: {workers} | "

        f"Recovered: {len(results)} | Remaining: {len(tasks_to_run)}"

    )

    print("-" * 110, flush=True)


    if not tasks_to_run:

        return results, elapsed_offset

    with ProcessPoolExecutor(max_workers=workers) as executor:

        futures = [executor.submit(run_single_experiment, task) for task in tasks_to_run]

        for index, future in enumerate(

            as_completed(futures), start=len(results) + 1

        ):

            result = future.result()

            elapsed = elapsed_offset + time.time() - start_time

            eta = elapsed / index * (total - index) if index else 0.0

            result["completed_index"] = index

            result["total_experiments"] = total

            result["elapsed_since_start_seconds"] = elapsed

            result["eta_after_result_seconds"] = eta

            result["finished_at"] = datetime.now().isoformat(timespec="seconds")

            results.append(result)

            if journal_path is not None:

                _append_journal_record(

                    journal_path,

                    {

                        "record_type": "result",

                        "task_key": list(_journal_task_key(result)),

                        "result": result,

                    },

                )


            if result.get("error"):

                status = f"ERROR: {result['error']}"

            else:

                on_target = result.get("on_target", "restart")

                gate_str = ""

                if result.get("learning_ratio") is not None:
                    gate_str += f" caar={result['learning_ratio']:.1%}"

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


    return results, elapsed_offset + time.time() - start_time



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
    parser = argparse.ArgumentParser(
        description="Unified Lifelong MAPF experiment runner"
    )
    parser.add_argument(
        "--algorithms",
        type=parse_algorithms,
        default=list(DEFAULT_ALGORITHMS),
        help=(
            "Comma-separated algorithms, or 'all'. "
            f"Choices: {', '.join(SUPPORTED_ALGORITHMS)}"
        ),
    )
    parser.add_argument(
        "--agents",
        type=str,
        default=None,
        help="Comma-separated agent counts, e.g. 50,100,200",
    )
    parser.add_argument(
        "--agent-start",
        type=int,
        default=50,
        help="First agent count when --agents is not set",
    )
    parser.add_argument(
        "--agent-stop",
        type=int,
        default=500,
        help="Last inclusive agent count when --agents is not set",
    )
    parser.add_argument(
        "--agent-step",
        type=int,
        default=50,
        help="Agent count step when --agents is not set",
    )
    parser.add_argument(
        "--workers",
        "--works",
        dest="workers",
        type=int,
        default=8,
        help="Parallel workers",
    )
    parser.add_argument(
        "--obs-radius",
        type=int,
        default=None,
        help="Override the local observation radius",
    )
    parser.add_argument(
        "--cache-algorithms",
        action="store_true",
        help="Reuse algorithm objects only for methods with verified reset state",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Generate SVG animations",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=512,
        help="Episode length",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated seeds, e.g. 0,1,2",
    )
    parser.add_argument(
        "--main-dir",
        type=str,
        default="./",
        help="Project root directory",
    )
    parser.add_argument(
        "--map-types",
        type=str,
        default="all",
        help="Comma-separated map types, or 'all'",
    )
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        help="Override one representative map with map_type=map_name",
    )
    parser.add_argument(
        "--map-url",
        type=str,
        default=None,
        help="Custom map URL; MovingAI .map files are supported",
    )
    parser.add_argument(
        "--map-file",
        type=str,
        default=None,
        help="Custom local map file path",
    )
    parser.add_argument(
        "--map-list",
        type=str,
        default=None,
        help="YAML file whose top-level keys are map names",
    )
    parser.add_argument(
        "--task-manifest",
        type=str,
        default=None,
        help=(
            "JSON manifest defining one explicit seed/start/goal assignment "
            "per selected map; requires --map-list"
        ),
    )
    parser.add_argument(
        "--trim-border",
        dest="trim_border",
        action="store_true",
        help="Trim one-cell border from a custom map",
    )
    parser.add_argument(
        "--no-trim-border",
        dest="trim_border",
        action="store_false",
        help="Do not trim a custom map border",
    )
    parser.set_defaults(trim_border=None)
    parser.add_argument(
        "--on-target",
        choices=("restart", "finish", "nothing"),
        default=None,
        help="Override the POGEMA on-target mode",
    )
    parser.add_argument(
        "--collision-system",
        choices=("soft", "block_both", "priority"),
        default="block_both",
        help="Override the collision system",
    )
    parser.add_argument(
        "--epom-weights-path",
        type=str,
        default=None,
        help="EPOM-L weights directory",
    )
    parser.add_argument(
        "--caar-weights-path",
        dest="caar_weights_path",
        type=str,
        default=None,
        help="CAAR weights directory",
    )
    parser.add_argument(
        "--caar-candidate-manifest",
        type=str,
        default=None,
        help="JSON declaration pinning the CAAR and frozen EPOM-L artifacts",
    )
    parser.add_argument(
        "--switcher-weights-path",
        type=str,
        default=None,
        help="Wait-aware Switcher weights directory",
    )
    parser.add_argument(
        "--no-wait-detect-switcher-weights-path",
        type=str,
        default=None,
        help="All-state Switcher weights for SRSLM-NoWaitDetect",
    )
    parser.add_argument(
        "--no-reweight-weights-path",
        dest="no_reweight_weights_path",
        type=str,
        default=None,
        help="NoReweight base-policy weights directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exp_result",
        help="Directory for JSON results",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename",
    )
    parser.add_argument(
        "--result-journal",
        type=str,
        default=None,
        help="Append each completed tuple to an fsync-backed JSONL journal",
    )
    parser.add_argument(
        "--result-journal-contract",
        type=str,
        default=None,
        help="Lowercase SHA256 binding a journal to one frozen protocol",
    )
    parser.add_argument(
        "--resume-result-journal",
        action="store_true",
        help="Reuse successful tuples from a matching result journal",
    )
    parser.add_argument(
        "--save",
        dest="save",
        action="store_true",
        default=True,
        help="Save JSON results",
    )
    parser.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="Do not save JSON results",
    )
    return parser.parse_args()



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

    map_sources = sum(

        bool(value)

        for value in (args.map_url, args.map_file, args.map_list)

    )

    if map_sources > 1:

        raise ValueError("--map-url, --map-file, and --map-list are mutually exclusive")

    if args.task_manifest and not args.map_list:

        raise ValueError("--task-manifest requires --map-list")

    if args.task_manifest and (args.agents is not None or args.seeds is not None):

        raise ValueError(

            "--task-manifest cannot be combined with --agents or --seeds"

        )


    if args.on_target is None:

        args.on_target = "restart"

    if args.collision_system is None:

        args.collision_system = "block_both"

    if args.trim_border is None:

        args.trim_border = False


    algorithms = args.algorithms
    srslm_contract = srslm_contract_metadata(algorithms)
    srslm_ablation_contract = srslm_ablation_contract_metadata(algorithms)
    hybrid_contract = srslm_contract or srslm_ablation_contract

    agent_counts = parse_agent_counts(args)

    seeds = parse_seeds(args)


    custom_map = None

    map_list_sha256 = None

    map_registry_sha256 = None

    map_list_path = None

    map_texts = None

    explicit_task_specs = None

    task_manifest_sha256 = None

    task_manifest_protocol_id = None

    task_manifest_path = None

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

        if args.task_manifest:

            (

                explicit_task_specs,

                task_manifest_sha256,

                task_manifest_protocol_id,

                task_manifest_path,

            ) = load_explicit_task_manifest_snapshot(

                _project_path(args.main_dir, args.task_manifest),

                maps,

                map_texts,

                expected_map_list_sha256=map_list_sha256,

            )

            agent_counts = sorted(

                {spec["num_agents"] for spec in explicit_task_specs}

            )

            seeds = sorted({spec["seed"] for spec in explicit_task_specs})

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

        explicit_task_specs=explicit_task_specs,

    )

    if srslm_contract is not None:
        integrity_metadata = srslm_integrity_metadata(
            args,
            map_list_sha256=map_list_sha256,
            map_registry_sha256=map_registry_sha256,
        )
    elif srslm_ablation_contract is not None:
        integrity_metadata = srslm_ablation_integrity_metadata(
            args,
            srslm_ablation_contract["algorithm"],
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

        "contention_metric": contention_metric_metadata(),

        "vertex_flow_metric": vertex_flow_metric_metadata(),

        "reverse_metric": {

            "version": "previous_timestep_position_target_segment_v3",

            "definition": (

                "a submitted movement proposes the position occupied at the "

                "immediately previous timestep in the current target segment"

            ),

            "reverse_rate_denominator": "submitted non-wait movement actions",

            "history_update": (

                "the observed position is recorded every timestep, including "

                "waits and blocked moves; a target change resets it"

            ),

        },

        "static_astar_metric": static_astar_metric_metadata(),

        "runtime_metric": runtime_metric_metadata(),

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

        "caar_candidate_manifest": getattr(
            args, "caar_candidate_manifest", None
        ),

        "switcher_weights_path": args.switcher_weights_path,

        "no_wait_detect_switcher_weights_path": (
            args.no_wait_detect_switcher_weights_path
        ),

        "no_reweight_weights_path": args.no_reweight_weights_path,

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

        "map_list_sha256": map_list_sha256,

        "map_registry_sha256": map_registry_sha256,

        "task_manifest": (

            str(task_manifest_path)

            if task_manifest_path is not None

            else None

        ),

        "task_manifest_sha256": task_manifest_sha256,

        "task_manifest_protocol_id": task_manifest_protocol_id,

        "explicit_task_count": (

            len(explicit_task_specs)

            if explicit_task_specs is not None

            else None

        ),

        "trim_border": args.trim_border,

        "result_journal": args.result_journal,

        "result_journal_contract": args.result_journal_contract,

        "resume_result_journal": args.resume_result_journal,

    }

    print("Configuration")

    print(f"  algorithms: {', '.join(algorithms)}")

    print(f"  agent_counts: {agent_counts[0]}..{agent_counts[-1]} ({len(agent_counts)} values)")

    print(f"  maps: {', '.join(maps.values())}")

    print(f"  obs_radius: {args.obs_radius if args.obs_radius is not None else 'default'}")

    print(f"  max_steps: {args.max_steps} | seeds: {', '.join(str(seed) for seed in seeds)} | animate: {args.animate}")

    print(f"  on_target: {args.on_target} | collision: {args.collision_system}")

    results, elapsed = run_experiments(

        tasks,

        args.workers,

        result_journal=args.result_journal,

        journal_contract=args.result_journal_contract,

        resume_result_journal=args.resume_result_journal,

    )

    metadata["epom_lifelong_manifest"] = epom_lifelong_result_manifest(
        results,
        algorithms,
    )

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
