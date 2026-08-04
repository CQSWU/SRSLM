import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from policy_estimation.model import (
    CheckpointSchemaError,
    PolicyEstimationModel,
    PolicyEstimationModelConfig,
    PolicyReturnEstimator,
    make_checkpoint,
    save_policy_return_checkpoint,
    validate_checkpoint_payload,
)


def _config(*, coordinate_encoding="absolute_v1"):
    return PolicyEstimationModelConfig(
        obs_shape=(4, 5, 5),
        encoder_num_filters=4,
        encoder_num_res_blocks=0,
        coordinate_encoding=coordinate_encoding,
    )


def _observations(count=3):
    result = []
    for index in range(count):
        obs = np.zeros((4, 5, 5), dtype=np.float32)
        obs[0, index % 5, (index + 1) % 5] = 1.0
        result.append(
            {
                "obs": obs,
                "xy": np.asarray([index, -index], dtype=np.float32),
                "target_xy": np.asarray([4, 4], dtype=np.float32),
            }
        )
    return result


class PolicyEstimationModelTests(unittest.TestCase):
    def test_absolute_v1_keeps_original_coordinate_preprocessing_exactly(self):
        model = PolicyEstimationModel(_config())
        batch = {
            key: torch.from_numpy(
                np.stack([sample[key] for sample in _observations()])
            ).float()
            for key in ("obs", "xy", "target_xy")
        }
        coordinates = torch.cat([batch["xy"], batch["target_xy"]], dim=-1)
        expected = coordinates / torch.maximum(
            torch.abs(coordinates),
            torch.tensor(64.0, dtype=coordinates.dtype),
        )

        actual = model.encoder._coordinate_features(batch)

        self.assertTrue(torch.equal(actual, expected))

    def test_legacy_model_config_defaults_coordinate_encoding(self):
        payload = make_checkpoint(PolicyEstimationModel(_config()), branch="caar")
        del payload["model_config"]["coordinate_encoding"]
        branch, config = validate_checkpoint_payload(payload, expected_branch="caar")
        self.assertEqual(branch, "caar")
        self.assertEqual(config.coordinate_encoding, "absolute_v1")

    def test_nonrecurrent_scalar_model_and_independent_instances(self):
        first = PolicyEstimationModel(_config())
        second = PolicyEstimationModel(_config())
        batch = {
            key: torch.from_numpy(
                np.stack([sample[key] for sample in _observations()])
            ).float()
            for key in ("obs", "xy", "target_xy")
        }

        self.assertEqual(tuple(first(batch).shape), (3,))
        self.assertFalse(any("rnn" in name.lower() for name, _ in first.named_modules()))
        first_parameters = dict(first.named_parameters())
        second_parameters = dict(second.named_parameters())
        self.assertEqual(set(first_parameters), set(second_parameters))
        for name in first_parameters:
            self.assertNotEqual(
                first_parameters[name].data_ptr(),
                second_parameters[name].data_ptr(),
            )
        linear_layers = [
            module for module in first.value_head if isinstance(module, torch.nn.Linear)
        ]
        self.assertEqual(
            [(layer.in_features, layer.out_features) for layer in linear_layers],
            [(512, 512), (512, 512), (512, 1)],
        )

    def test_strict_schema_rejects_wrong_branch_and_extra_fields(self):
        model = PolicyEstimationModel(_config())
        payload = make_checkpoint(model, branch="caar")
        with self.assertRaisesRegex(CheckpointSchemaError, "Expected"):
            validate_checkpoint_payload(payload, expected_branch="ao_safe")

        malformed = dict(payload)
        malformed["action_head"] = {}
        with self.assertRaisesRegex(CheckpointSchemaError, "unexpected"):
            validate_checkpoint_payload(malformed, expected_branch="caar")

        with self.assertRaisesRegex(ValueError, "coordinate_encoding"):
            PolicyEstimationModelConfig(coordinate_encoding="unknown")

        mismatched_encoding = make_checkpoint(
            PolicyEstimationModel(_config()),
            branch="caar",
            training_metadata={"coordinate_encoding": "removed_v2"},
        )
        with self.assertRaisesRegex(
            CheckpointSchemaError,
            "coordinate_encoding disagrees",
        ):
            validate_checkpoint_payload(mismatched_encoding)

    def test_inference_load_and_predict_are_deterministic_and_rng_neutral(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "caar_estimator.pth"
            torch.manual_seed(7)
            model = PolicyEstimationModel(_config())
            save_policy_return_checkpoint(
                path,
                model,
                branch="caar",
                training_metadata={"objective": "mse_on_raw_mc_return"},
            )

            torch.manual_seed(12345)
            before_load = torch.random.get_rng_state().clone()
            estimator = PolicyReturnEstimator(
                path,
                device="cpu",
                expected_branch="caar",
            )
            self.assertTrue(torch.equal(before_load, torch.random.get_rng_state()))
            before_predict = torch.random.get_rng_state().clone()
            first = estimator.predict(_observations())
            second = estimator.predict(_observations())

            np.testing.assert_array_equal(first, second)
            self.assertEqual(first.shape, (3,))
            self.assertTrue(torch.equal(before_predict, torch.random.get_rng_state()))
            self.assertFalse(estimator.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in estimator.parameters()))


if __name__ == "__main__":
    unittest.main()
