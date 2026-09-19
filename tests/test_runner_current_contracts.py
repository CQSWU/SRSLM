"""Public runner binds the paper artifacts, not the retired ARPE backbone."""
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import run_experiments as runner
from agents.ao_replan import AORePlan, AORePlanConfig
from agents.ao_replan_soft_ablation import AORePlanSoftNoCheck, _SoftNoCheckWrapper
from planning.ao_replan_algo import AORePlanWrapper


def test_public_names_are_explicit_and_retired_names_are_rejected():
    assert runner.SUPPORTED_ALGORITHMS == (
        'RePlan', 'AORePlan', 'AORePlan-SoftNoCheck', 'EPOM-Lifelong-FT',
        'Direct', 'ARPE', 'SRSLM-NoWait', 'SRSLM-OnlyWait', 'SRSLM',
    )
    assert set(runner.ALGORITHM_ALIASES.values()) == set(runner.SUPPORTED_ALGORITHMS)
    for name in ('DCC', 'DHC', 'Follower', 'NoReweight', 'SRSLM-NoWaitDetect', 'SRSLM-WaitDetectOnly', 'v8b'):
        with pytest.raises(Exception):
            runner.parse_algorithms(name)


def test_public_runner_has_no_unshipped_adapter_imports():
    source = inspect.getsource(runner)
    for module in ('agents.dcc', 'agents.primal2', 'agents.follower',
                   'agents.chs', 'agents.assistant_switcher', 'agents.srslm_ablation'):
        assert module not in source


def test_learned_policies_are_always_episode_fresh():
    for name in ('Direct', 'ARPE', 'SRSLM', 'SRSLM-NoWait', 'SRSLM-OnlyWait'):
        assert not runner.should_cache_algorithm(name, True)
    assert runner.should_cache_algorithm('AORePlan', True)


@pytest.mark.parametrize('collision', ['block_both', 'soft', 'priority'])
def test_srslm_contract_records_actual_execution_rule(collision):
    contract = runner.srslm_contract_metadata(['SRSLM'], collision)
    assert contract['deployment']['simulator_collision_system'] == collision
    assert contract['deployment']['wait_rule'] == 'aoreplan_wait_directly_uses_caar'
    assert not contract['deployment']['joint_conflict_prediction_enabled']
    assert runner.srslm_contract_metadata(['RePlan'], collision) is None


def test_main_supplies_selected_rule_to_srslm_contract():
    import ast
    module = ast.parse(inspect.getsource(runner))
    calls = [node for node in ast.walk(module)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == 'srslm_contract_metadata']
    assert len(calls) == 1
    assert ast.unparse(calls[0].args[1]) == 'args.collision_system'


def test_direct_uses_epom_l_and_only_the_selected_top2_rule(tmp_path):
    with patch('agents.epom_direct_reweight.EPOMDirectReweight') as policy:
        runner.build_algorithm('Direct', '.', 42, epom_weights_path='weights/base')
    cfg = policy.call_args.args[0]
    assert cfg.artifact_profile == 'lifelong_finetuned'
    assert Path(cfg.path_to_weights).is_absolute()
    assert cfg.reweight_bonus == 1.0
    weights = tmp_path / 'weights/EPOM-lifelong-finetune-r5/EPOM-Lifelong-Finetune-R5'
    (weights / 'checkpoint_p0').mkdir(parents=True)
    (weights / 'config.json').write_text('{}')
    (weights / 'checkpoint_p0/checkpoint.pth').touch()
    with patch('agents.epom_direct_reweight.EPOMDirectReweight') as auto_policy:
        runner.build_algorithm('Direct', tmp_path, 42)
    auto_cfg = auto_policy.call_args.args[0]
    assert auto_cfg.path_to_weights.endswith('EPOM-Lifelong-Finetune-R5')


def test_retired_direct_cli_variants_are_rejected():
    with patch.object(sys, 'argv', [
        'run_experiments.py', '--algorithms', 'Direct',
        '--direct-transform', 'clipped_relu',
    ]), pytest.raises(SystemExit):
        args = runner.parse_args()


def _artifact():
    return SimpleNamespace(weights_relative='weights/caar', checkpoint_relative='weights/caar/checkpoint_p0/model.pth',
        base_weights_relative='weights/base', base_checkpoint_relative='weights/base/checkpoint_p0/base.pth',
        weights_path=Path('weights/caar').resolve(),
        checkpoint_path=Path('weights/caar/checkpoint_p0/model.pth').resolve())


def test_arpe_loads_exact_frozen_candidate():
    artifact = _artifact()
    with patch.object(runner, '_load_arpe_candidate_artifact', return_value=artifact), \
         patch('agents.arpe.ARPE.load') as load:
        runner.build_algorithm('ARPE', '.', 42, arpe_candidate_manifest='manifest.json')
    assert load.call_args.args[0] is artifact
    assert load.call_args.kwargs['seed'] == 42
    assert load.call_args.kwargs['action_sampling'] == 'direct_numpy'


def test_wait_ablations_use_identical_candidate_and_distinct_switcher_contracts():
    with patch.object(runner, '_load_arpe_candidate_artifact', return_value=_artifact()), \
         patch('agents.srslm_arpe_ablation.SRSLMOnlyWait') as only, \
         patch('agents.srslm_arpe_ablation.SRSLMNoWait') as nowait:
        runner.build_algorithm('SRSLM-OnlyWait', '.', 42, arpe_candidate_manifest='manifest.json')
        runner.build_algorithm('SRSLM-NoWait', '.', 42, arpe_candidate_manifest='manifest.json',
                               switcher_weights_path='weights/independent-nowait')
        with pytest.raises(ValueError, match='independently trained'):
            runner.build_algorithm('SRSLM-NoWait', '.', 42, arpe_candidate_manifest='manifest.json')
        with pytest.raises(ValueError, match='must not load'):
            runner.build_algorithm('SRSLM-OnlyWait', '.', 42, arpe_candidate_manifest='manifest.json',
                                   switcher_weights_path='weights/full')
    only_cfg = only.call_args.args[0]
    nowait_cfg = nowait.call_args.args[0]
    assert only_cfg.candidate == nowait_cfg.candidate
    assert not hasattr(only_cfg, 'switcher')
    assert 'independent-nowait' in nowait_cfg.switcher.path_to_weights


@pytest.mark.parametrize('collision', ['block_both', 'soft'])
def test_default_aoreplan_keeps_static_occupancy_check(collision):
    agent = AORePlan(AORePlanConfig())
    agent.set_grid_config(SimpleNamespace(collision_system=collision))
    assert agent.WRAPPER_CLASS is AORePlanWrapper
    wrapper = object.__new__(agent.WRAPPER_CLASS)
    wrapper.static_astar = SimpleNamespace(
        get_action=lambda _index, _observation: 4
    )
    wrapper.last_static_astar_invoked_mask = [False]
    wrapper.moves = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
    agents = np.zeros((11, 11), dtype=int)
    agents[5, 6] = 1
    obs = {'obstacles': np.zeros((11, 11)), 'agents': agents}
    assert wrapper._static_astar_action(0, obs) == 0


def test_search_soft_bypass_is_explicit_and_not_available_in_block_both():
    policy = runner.build_algorithm('AORePlan-SoftNoCheck', '.', 42)
    assert isinstance(policy, AORePlanSoftNoCheck)
    policy.set_grid_config(SimpleNamespace(collision_system='soft'))
    with pytest.raises(ValueError, match='standalone soft'):
        policy.set_grid_config(SimpleNamespace(collision_system='block_both'))
    wrapper = object.__new__(_SoftNoCheckWrapper)
    wrapper.static_astar = SimpleNamespace(
        get_action=lambda _index, _observation: 4
    )
    wrapper.last_static_astar_invoked_mask = [False]
    assert wrapper._static_astar_action(0, {}) == 4
    assert wrapper.last_static_astar_invoked_mask == [True]


def test_public_manifest_uses_portable_paths_without_required_hashes():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / 'configs/arpe_final_candidate.json').read_text())
    from agents.arpe import ArpeCandidateArtifact
    artifact = ArpeCandidateArtifact.from_mapping(data, root)
    assert artifact.weights_relative == data['weights_path']
    assert artifact.checkpoint_relative == data['checkpoint_path']
    for key in ('checkpoint_sha256', 'base_checkpoint_sha256'):
        assert key not in data
