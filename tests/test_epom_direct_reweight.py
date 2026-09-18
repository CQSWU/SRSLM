from types import SimpleNamespace

import numpy as np

from agents.epom_direct_reweight import (
    EPOMDirectReweight,
    EPOMDirectReweightConfig,
    PRIMAL3_ENTROPY_THRESHOLD,
)
from pomapf_env.stigmergic import AcoState


def _algorithm(seed=42):
    algorithm = object.__new__(EPOMDirectReweight)
    algorithm.algo_cfg = SimpleNamespace(seed=seed, tau_rho=0.1)
    algorithm.reweight_bonus = 1.0
    algorithm.entropy_threshold = PRIMAL3_ENTROPY_THRESHOLD
    algorithm.aco = AcoState(rho=0.1)
    algorithm._bonus_rng = np.random.default_rng(
        np.random.SeedSequence([seed, 130913, 1])
    )
    return algorithm


def test_defaults_are_the_single_selected_paper_rule():
    config = EPOMDirectReweightConfig()
    assert config.reweight_bonus == 1.0
    assert config.entropy_threshold == PRIMAL3_ENTROPY_THRESHOLD
    assert not hasattr(config, "gate")
    assert not hasattr(config, "centering_scope")
    assert not hasattr(config, "pressure_transform")


def test_trace_uses_full_11x11_free_cell_mean():
    algorithm = _algorithm()
    obstacles = np.zeros((11, 11), dtype=bool)
    obstacles[4, 5] = True
    algorithm.aco.configure_from_obstacle_mask(obstacles, clear=True)
    algorithm.aco.tau.fill(1.0)
    algorithm.aco.tau[obstacles] = 0.0
    algorithm.aco.tau[5, 5] = 10.0
    algorithm.aco.tau[6, 5] = 4.0
    algorithm.aco.tau[5, 4] = 2.0
    algorithm.aco.tau[5, 6] = 1.0
    algorithm.aco.tau[0, 0] -= 13.0

    pressure = algorithm._candidate_trace(np.array([[5, 5]], dtype=np.int64))

    assert np.allclose(pressure, [[9.0, 0.0, 3.0, 1.0, 0.0]])


def test_bonus_goes_only_to_lower_pressure_of_top_two_legal_movements():
    algorithm = _algorithm()
    pressure = np.array([[0.0, 4.0, 1.0, -2.0, 3.0]], dtype=np.float32)
    logits = np.array([[8.0, 2.0, 1.5, 1.0, 0.0]], dtype=np.float32)
    legal = np.array([[False, True, True, True, True]])

    bonus = algorithm._top2_bonus(
        pressure, legal, logits, np.array([True])
    )

    assert np.array_equal(bonus, [[0.0, 0.0, 1.0, 0.0, 0.0]])


def test_gate_or_fewer_than_two_legal_movements_disables_correction():
    algorithm = _algorithm()
    pressure = np.zeros((2, 5), dtype=np.float32)
    logits = np.ones((2, 5), dtype=np.float32)
    legal = np.array([
        [False, True, True, False, False],
        [False, True, False, False, False],
    ])

    bonus = algorithm._top2_bonus(
        pressure, legal, logits, np.array([False, True])
    )

    assert np.count_nonzero(bonus) == 0


def test_wait_never_receives_a_bonus():
    algorithm = _algorithm()
    pressure = np.array([[-10.0, 2.0, 1.0, 4.0, 5.0]], dtype=np.float32)
    logits = np.array([[100.0, 3.0, 2.0, 1.0, 0.0]], dtype=np.float32)
    legal = np.array([[False, True, True, True, True]])

    bonus = algorithm._top2_bonus(
        pressure, legal, logits, np.array([True])
    )

    assert bonus[0, 0] == 0.0
    assert bonus.sum() == 1.0


def test_after_reset_clears_trace_and_restores_both_rngs():
    algorithm = _algorithm()
    clears = []
    algorithm.grid_memory = SimpleNamespace(clear=lambda: clears.append(True))
    algorithm.aco.configure_from_obstacle_mask(
        np.zeros((3, 3), dtype=bool), clear=True
    )
    algorithm.after_reset()
    sample = algorithm._bonus_rng.random(20)
    algorithm.aco.tau.fill(5)
    algorithm._stats = [{"entropy_mean": 1.0}]
    algorithm.after_reset()
    assert np.array_equal(sample, algorithm._bonus_rng.random(20))
    assert np.count_nonzero(algorithm.aco.tau) == 0
    assert algorithm._stats == []
    assert algorithm.rnn_states is None
    assert len(clears) == 2
