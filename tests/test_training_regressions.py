import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import yaml
from pydantic import ValidationError
from sample_factory.algo.learning.learner import Learner

from learning.grid_memory import GridMemory
from train import _sync_resume_cli_overrides, validate_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TrainingRegressionTests(unittest.TestCase):
    def test_resume_preserves_explicit_environment_step_override(self):
        with tempfile.TemporaryDirectory() as directory:
            train_dir = Path(directory)
            run_dir = train_dir / "trace-run"
            run_dir.mkdir()
            resume_config = run_dir / "config.json"
            resume_config.write_text(
                json.dumps(
                    {
                        "train_for_env_steps": 50_000_000,
                        "full_config": {
                            "experiment_settings": {
                                "train_for_env_steps": 50_000_000,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            flat_config = SimpleNamespace(
                train_dir=str(train_dir),
                experiment="trace-run",
                train_for_env_steps=1_000_000_000,
                cli_args={},
            )

            _sync_resume_cli_overrides(
                flat_config,
                {"train_for_env_steps"},
            )

            updated = json.loads(resume_config.read_text(encoding="utf-8"))
            self.assertEqual(updated["train_for_env_steps"], 1_000_000_000)
            self.assertEqual(
                updated["full_config"]["experiment_settings"][
                    "train_for_env_steps"
                ],
                1_000_000_000,
            )
            self.assertEqual(
                flat_config.cli_args["train_for_env_steps"],
                1_000_000_000,
            )

    def test_grid_memory_rejects_even_or_non_square_observations(self):
        memory = GridMemory()
        for shape in ((4, 4), (3, 5), (3,)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                memory.update(0, 0, np.zeros(shape, dtype=np.float32))

    def test_obsolete_training_keys_are_rejected(self):
        config_path = (
            PROJECT_ROOT
            / "learning"
            / "train_epom_trace_paper_conv_fusion_r5_500m.yaml"
        )
        modern = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        obsolete_yaml = copy.deepcopy(modern)
        obsolete_yaml["async_ppo"]["ppo_epochs"] = obsolete_yaml["async_ppo"].pop(
            "num_epochs"
        )
        with self.assertRaises(ValidationError):
            validate_config(obsolete_yaml)

        legacy = copy.deepcopy(obsolete_yaml)
        legacy["async_ppo"]["num_minibatches_to_accumulate"] = -1
        legacy["global_settings"]["experiments_root"] = None
        legacy.setdefault("evaluation", {})["record_to"] = "../recs"
        legacy["experiment_settings"]["caar_freeze_backbone"] = False
        with self.assertRaises(ValidationError):
            validate_config(legacy)

    def test_corrupt_checkpoint_raises_last_root_cause_after_retries(self):
        root_cause = OSError("corrupt checkpoint")
        with patch("train.torch.load", side_effect=root_cause) as mocked_load:
            with self.assertRaises(RuntimeError) as raised:
                Learner.load_checkpoint(
                    [Path("broken-checkpoint.pth")],
                    torch.device("cpu"),
                )
        self.assertEqual(mocked_load.call_count, 3)
        self.assertIs(raised.exception.__cause__, root_cause)


if __name__ == "__main__":
    unittest.main()
