import unittest
from types import SimpleNamespace

import numpy as np

from agents.srlsm import SRSLM, SRSLMConfig, select_ao_by_absolute_return


class _PolicyStub:
    device = "cpu"
    ppo = SimpleNamespace(action_space=SimpleNamespace(n=5))

    def after_reset(self):
        pass

    def set_grid_config(self, _grid_config):
        pass


class _PlannerStub:
    def reset(self):
        pass


class _EstimatorStub:
    def eval(self):
        return self

    def parameters(self):
        return ()


class TraceSwitcherContractTests(unittest.TestCase):
    def test_strict_value_margin_and_nonfinite_values_choose_caar(self):
        choices, nonfinite = select_ao_by_absolute_return(
            [1.0, 0.0, np.nan, 2.0, 1.0],
            [1.0, 1.1, 100.0, np.inf, 2.0],
            margin=1.0,
        )
        np.testing.assert_array_equal(choices, [0, 1, 0, 0, 0])
        np.testing.assert_array_equal(nonfinite, [False, False, True, True, False])

    def test_reverse_fallback_has_no_persistent_state(self):
        config = SRSLMConfig()
        self.assertTrue(config.reverse_caar_override_enabled)
        self.assertEqual(config.value_margin, 0.0)

    def test_loading_does_not_require_source_or_training_hashes(self):
        switcher = SRSLM(
            SRSLMConfig(),
            caar_factory=lambda _config: _PolicyStub(),
            planner_factory=lambda **_kwargs: _PlannerStub(),
            estimator_factory=lambda **_kwargs: _EstimatorStub(),
        )
        self.assertEqual(switcher.action_count, 5)


if __name__ == "__main__":
    unittest.main()
