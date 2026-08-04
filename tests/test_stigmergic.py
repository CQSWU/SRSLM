import unittest

import numpy as np

from pomapf_env.stigmergic import AcoState


class TestOccupancyTrace(unittest.TestCase):
    def setUp(self):
        self.state = AcoState(rho=0.1)
        obstacles = np.zeros((5, 5), dtype=bool)
        obstacles[0, 0] = True
        self.state.configure_from_obstacle_mask(obstacles, clear=True)

    def test_continuous_occupancy_is_decaying_aco_trace(self):
        for step in range(1, 101):
            self.state._update_trace([(2, 2)], evaporate=step > 1)

        expected = (1.0 - 0.9**100) / 0.1
        self.assertAlmostEqual(float(self.state.tau[2, 2]), expected, delta=1e-5)

    def test_one_step_evaporation_is_exactly_point_nine(self):
        self.state.tau[2, 2] = 2.0
        self.state._update_trace([], evaporate=True)
        self.assertAlmostEqual(float(self.state.tau[2, 2]), 1.8, places=7)

    def test_duplicate_positions_form_one_occupancy_event(self):
        self.state._update_trace([(2, 2), (2, 2)], evaporate=False)
        self.assertAlmostEqual(float(self.state.tau[2, 2]), 1.0, places=7)

    def test_reset_clears_previous_episode_trace(self):
        observations = [
            {
                "obs": np.zeros((3, 3, 3), dtype=np.float32),
                "xy": np.array([2, 2], dtype=np.float32),
            }
        ]
        self.state.tau[1, 1] = 7.0
        self.state.reset_episode(observations, positions=[(2, 2)])
        self.assertEqual(float(self.state.tau[1, 1]), 0.0)
        self.assertEqual(float(self.state.tau[2, 2]), 1.0)
        self.assertEqual(tuple(observations[0]["tau"].shape), (1, 3, 3))

    def test_raw_pogema_observation_can_receive_tau(self):
        observations = [
            {
                "obstacles": np.zeros((3, 3), dtype=np.float32),
                "xy": np.array([2, 2], dtype=np.float32),
            }
        ]

        self.state.reset_episode(observations, positions=[(2, 2)])

        self.assertEqual(tuple(observations[0]["tau"].shape), (1, 3, 3))
        self.assertGreater(float(observations[0]["tau"][0, 1, 1]), 0.0)

    def test_explicit_tau_radius_is_independent_of_context_radius(self):
        observations = [
            {
                "obs": np.zeros((3, 11, 11), dtype=np.float32),
                "xy": np.array([2, 2], dtype=np.float32),
            }
        ]

        self.state.reset_episode(
            observations,
            positions=[(2, 2)],
            radius=1,
        )

        self.assertEqual(tuple(observations[0]["tau"].shape), (1, 3, 3))
        self.assertGreater(float(observations[0]["tau"][0, 1, 1]), 0.0)

    def test_raw_tau_observation_is_not_mean_centered(self):
        observations = [
            {
                "obstacles": np.zeros((3, 3), dtype=np.float32),
                "xy": np.array([2, 2], dtype=np.float32),
            }
        ]

        self.state.reset_episode(
            observations,
            positions=[(2, 2)],
            raw_tau=True,
        )

        tau = observations[0]["tau"][0]
        self.assertEqual(float(tau[1, 1]), 1.0)
        self.assertEqual(float(tau[0, 0]), 0.0)

    def test_local_channel_is_mean_centered_and_masks_obstacles(self):
        self.state.tau[2, 2] = 4.0
        self.state.tau[1, 2] = 1.0
        self.state.tau[0, 0] = 0.9
        local = self.state.extract_local_tau(1, 1, radius=1)
        free_values = np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0])
        mean = float(free_values.mean())
        self.assertAlmostEqual(float(local[1, 2]), 1.0 - mean, places=7)
        self.assertAlmostEqual(float(local[2, 2]), 4.0 - mean, places=7)
        self.assertAlmostEqual(float(local[0, 1]), -mean, places=7)
        self.assertEqual(float(local[0, 0]), 0.0)

        free_mask = np.ones((3, 3), dtype=bool)
        free_mask[0, 0] = False
        self.assertAlmostEqual(float(local[free_mask].mean()), 0.0, places=7)

    def test_zero_and_uniform_traces_have_zero_pressure(self):
        local = self.state.extract_local_tau(2, 2, radius=1)
        np.testing.assert_array_equal(local, np.zeros((3, 3), dtype=np.float32))

        self.state.tau.fill(3.0)
        local = self.state.extract_local_tau(2, 2, radius=1)
        np.testing.assert_allclose(local, np.zeros((3, 3), dtype=np.float32), atol=1e-7)

    def test_map_edge_padding_does_not_enter_local_mean(self):
        self.state.tau[0, 1] = 2.0
        self.state.tau[1, 0] = 4.0
        self.state.tau[1, 1] = 0.0
        local = self.state.extract_local_tau(0, 0, radius=1)

        mean = 2.0
        self.assertEqual(float(local[0, 0]), 0.0)
        self.assertEqual(float(local[0, 1]), 0.0)
        self.assertEqual(float(local[1, 0]), 0.0)
        self.assertEqual(float(local[1, 1]), 0.0)
        self.assertAlmostEqual(float(local[1, 2]), 2.0 - mean, places=7)
        self.assertAlmostEqual(float(local[2, 1]), 4.0 - mean, places=7)
        self.assertAlmostEqual(float(local[2, 2]), -mean, places=7)

if __name__ == "__main__":
    unittest.main()
