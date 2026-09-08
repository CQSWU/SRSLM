import numpy as np
from pydantic import ValidationError
import pytest

from agents.epom_direct_reweight import (
    EPOMDirectReweight,
    EPOMDirectReweightConfig,
)
from pomapf_env.stigmergic import AcoState


def test_signed_pressure_transform_is_identity():
    centered = np.array([[-3.0, -0.5, 0.0, 0.5, 3.0]], dtype=np.float32)
    transformed = EPOMDirectReweight.transform_pressure(centered, "signed")
    assert np.array_equal(transformed, centered)


def test_clipped_relu_pressure_only_penalizes_above_mean_candidates():
    centered = np.array([[-3.0, -0.5, 0.0, 0.5, 3.0]], dtype=np.float32)
    transformed = EPOMDirectReweight.transform_pressure(
        centered, "clipped_relu", cap=2.0
    )
    assert np.array_equal(
        transformed,
        np.array([[0.0, 0.0, 0.0, 0.5, 2.0]], dtype=np.float32),
    )


def test_candidate_trace_centers_only_static_legal_actions():
    algorithm = object.__new__(EPOMDirectReweight)
    algorithm.centering_scope = "candidate"
    algorithm.aco = AcoState(rho=0.1)
    obstacles = np.zeros((3, 3), dtype=bool)
    obstacles[0, 1] = True  # up is statically illegal
    algorithm.aco.configure_from_obstacle_mask(obstacles, clear=True)
    algorithm.aco.tau[:] = np.array(
        [[0.0, 0.0, 0.0], [3.0, 5.0, 1.0], [0.0, 2.0, 0.0]],
        dtype=np.float32,
    )

    centered = algorithm._candidate_trace(np.array([[1, 1]], dtype=np.int64))

    # POGEMA action order is wait, up, down, left, right.  The legal-action
    # mean is (5 + 2 + 3 + 1) / 4 = 2.75; illegal up is left untouched.
    assert np.allclose(centered, [[2.25, 0.0, -0.75, 0.25, -1.75]])


def test_direct_api_defaults_match_paper_cli():
    config = EPOMDirectReweightConfig()
    assert config.centering_scope == "crop"
    assert config.gate == "always"
    assert config.pressure_transform == "signed"


def test_historical_candidate_centering_requires_explicit_selection():

    algorithm = object.__new__(EPOMDirectReweight)
    algorithm.centering_scope = "candidate"
    algorithm.aco = AcoState(rho=0.1)
    algorithm.aco.configure_from_obstacle_mask(
        np.zeros((11, 11), dtype=bool), clear=True
    )
    algorithm.aco.tau[5, 5] = 5.0
    algorithm.aco.tau[4, 5] = 1.0
    algorithm.aco.tau[6, 5] = 2.0
    algorithm.aco.tau[5, 4] = 3.0
    algorithm.aco.tau[5, 6] = 4.0

    # The historical five-cell rule remains explicitly labelled, never default.
    centered = algorithm._candidate_trace(np.array([[5, 5]], dtype=np.int64))
    assert np.array_equal(centered, [[2.0, -2.0, -1.0, 0.0, 1.0]])


def test_crop_scope_uses_full_11x11_free_cell_mean_without_recentering_candidates():
    algorithm = object.__new__(EPOMDirectReweight)
    algorithm.centering_scope = "crop"
    algorithm.aco = AcoState(rho=0.1)
    obstacles = np.zeros((11, 11), dtype=bool)
    obstacles[4, 5] = True  # up candidate must remain exactly zero
    algorithm.aco.configure_from_obstacle_mask(obstacles, clear=True)

    # 120 free cells: total trace is 120, hence the full-crop free-cell mean is
    # exactly 1.  Candidate values deliberately do not have mean 1, so a second
    # five-candidate centring would be observable.
    algorithm.aco.tau.fill(1.0)
    algorithm.aco.tau[obstacles] = 0.0
    algorithm.aco.tau[5, 5] = 10.0
    algorithm.aco.tau[6, 5] = 4.0
    algorithm.aco.tau[5, 4] = 2.0
    algorithm.aco.tau[5, 6] = 1.0
    # Restore the total to 120 without touching any sampled candidate.
    algorithm.aco.tau[0, 0] -= 13.0

    pressure = algorithm._candidate_trace(
        np.array([[5, 5]], dtype=np.int64)
    )

    assert np.allclose(pressure, [[9.0, 0.0, 3.0, 1.0, 0.0]])
    assert pressure[0].mean() == pytest.approx(2.6)  # no candidate re-centring


def test_centering_scope_rejects_unknown_value():
    with pytest.raises(ValidationError):
        EPOMDirectReweightConfig(centering_scope="whole_map")


def test_after_reset_clears_trace_memory_and_restores_sampler():
    from types import SimpleNamespace
    algorithm = object.__new__(EPOMDirectReweight)
    algorithm.algo_cfg = SimpleNamespace(seed=42)
    clears = []
    algorithm.grid_memory = SimpleNamespace(clear=lambda: clears.append(True))
    algorithm.aco = AcoState(rho=0.1)
    algorithm.aco.configure_from_obstacle_mask(np.zeros((3, 3), dtype=bool), clear=True)
    algorithm.after_reset()
    first = algorithm._rng.random(20)
    algorithm.aco.tau.fill(5)
    algorithm._stats = [{'entropy_mean': 1.0}]
    algorithm.after_reset()
    assert np.array_equal(first, algorithm._rng.random(20))
    assert np.count_nonzero(algorithm.aco.tau) == 0
    assert algorithm._stats == []
    assert algorithm.rnn_states is None
    assert len(clears) == 2
