import numpy as np
import pytest

from agents.controller import build_switcher_state
from pomapf_env.target_projection import get_square_target
from pomapf_env.wrappers import MatrixObservationWrapper


@pytest.mark.parametrize(
    "xy,target,radius,expected",
    [
        ((0, 0), (0, 0), 0, (0, 0)),
        ((0, 0), (0, 0), 5, (5, 5)),
        ((0, 0), (1, -1), 1, (2, 0)),
        ((3, 4), (-100, 100), 5, (0, 10)),
        ((0, 0), (100, -100), 7, (14, 0)),
        ((0.5, 1.5), (0, 0), 5, (5, 3)),
        ((-0.5, -1.5), (0, 0), 5, (5, 7)),
        ((2.5, -2.5), (0, 0), 5, (3, 7)),
    ],
)
def test_target_projection_keeps_rounding_clipping_and_dtype(
    xy, target, radius, expected
):
    projected = get_square_target(*xy, *target, radius)
    assert projected.shape == (2 * radius + 1, 2 * radius + 1)
    assert projected.dtype == np.float32
    assert projected.sum() == 1.0
    assert projected[expected] == 1.0


def _observation(xy, target, radius):
    size = 2 * radius + 1
    return {
        "obstacles": np.zeros((size, size), dtype=np.float32),
        "agents": np.zeros((size, size), dtype=np.float32),
        "xy": xy,
        "target_xy": target,
    }


@pytest.mark.parametrize("radius", [1, 5, 7])
def test_matrix_observation_uses_its_own_radius(radius):
    observation = _observation((0, 0), (100, -100), radius)
    matrix = MatrixObservationWrapper.to_matrix([observation])[0]
    expected = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
    expected[-1, 0] = 1.0
    assert matrix["obs"].dtype == np.float32
    np.testing.assert_array_equal(matrix["obs"][2], expected)


@pytest.mark.parametrize(
    "xy,target,expected",
    [
        ((0, 0), (1, -1), (6, 4)),
        ((3, 4), (-100, 100), (0, 10)),
        ((0.5, -1.5), (0, 0), (5, 7)),
    ],
)
def test_switcher_and_matrix_project_exact_float32_coordinates_identically(
    xy, target, expected
):
    observation = _observation(xy, target, 5)
    matrix = MatrixObservationWrapper.to_matrix([observation])[0]["obs"][2]
    switcher = build_switcher_state([observation], [1], [4])["obs"][0, 2]
    np.testing.assert_array_equal(switcher, matrix)
    assert switcher.dtype == np.float32
    assert switcher.sum() == 1.0
    assert switcher[expected] == 1.0


def test_each_observation_path_preserves_its_coordinate_conversion_order():
    observation = _observation((0.50000001, 0), (0, 0), 5)
    matrix = MatrixObservationWrapper.to_matrix([observation])[0]["obs"][2]
    switcher = build_switcher_state([observation], [1], [4])["obs"][0, 2]
    assert matrix[4, 5] == 1.0
    assert switcher[5, 5] == 1.0
    assert matrix.sum() == switcher.sum() == 1.0
