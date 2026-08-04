import unittest

import numpy as np

from agents.caar_lswitcher import CAARLSConfig, select_ao_by_absolute_return


class TraceSwitcherContractTests(unittest.TestCase):
    def test_strict_value_margin_and_nonfinite_values_choose_caar(self):
        choices, nonfinite = select_ao_by_absolute_return(
            [1.0, 0.0, np.nan, 2.0, 1.0],
            [1.0, 1.1, 100.0, np.inf, 2.0],
            margin=1.0,
        )
        np.testing.assert_array_equal(choices, [0, 1, 0, 0, 0])
        np.testing.assert_array_equal(nonfinite, [False, False, True, True, False])

    def test_default_reverse_fallback_has_four_step_caar_cooldown(self):
        config = CAARLSConfig()
        self.assertTrue(config.reverse_caar_override_enabled)
        self.assertEqual(config.reverse_caar_cooldown_steps, 4)

    def test_predictor_only_mode_cannot_retain_a_reverse_cooldown(self):
        with self.assertRaisesRegex(ValueError, "reverse CAAR cooldown"):
            CAARLSConfig(
                reverse_caar_override_enabled=False,
                reverse_caar_cooldown_steps=4,
            )


if __name__ == "__main__":
    unittest.main()
