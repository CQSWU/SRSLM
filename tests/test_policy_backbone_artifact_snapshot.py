import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from agents.policy_backbone import PolicyBackbone


class PolicyBackboneArtifactSnapshotTests(unittest.TestCase):
    def test_checkpoint_hash_and_load_use_the_same_byte_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pth"
            original = b"original checkpoint bytes"
            replacement = b"replacement checkpoint bytes"
            path.write_bytes(original)

            def replace_path_after_snapshot(stream, **_kwargs):
                path.write_bytes(replacement)
                self.assertEqual(stream.read(), original)
                return {"model": "loaded original"}

            with patch(
                "agents.policy_backbone.torch.load",
                side_effect=replace_path_after_snapshot,
            ):
                checkpoint, digest = PolicyBackbone._load_checkpoint_path(
                    path,
                    torch.device("cpu"),
                    "latest",
                )

            self.assertEqual(checkpoint["model"], "loaded original")
            self.assertEqual(digest, hashlib.sha256(original).hexdigest())
            self.assertEqual(path.read_bytes(), replacement)

    def test_config_hash_and_parse_use_the_same_byte_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            payload = json.dumps({"full_config": {"seed": 7}}).encode()
            path.write_bytes(payload)
            config, digest = PolicyBackbone._load_config_snapshot(path)
            path.write_text('{"full_config": {"seed": 99}}')

            self.assertEqual(config["full_config"]["seed"], 7)
            self.assertEqual(digest, hashlib.sha256(payload).hexdigest())

    def test_incompatible_critic_tensors_cannot_be_silently_dropped(self):
        model = torch.nn.Module()
        model.actor = torch.nn.Linear(2, 2)
        model.trace_value_head = torch.nn.Linear(4, 1)
        old_actor_weight = torch.full_like(model.actor.weight, 3.0)
        old_actor_bias = torch.full_like(model.actor.bias, 2.0)
        current_critic_weight = model.trace_value_head.weight.detach().clone()
        checkpoint = {
            "actor.weight": old_actor_weight,
            "actor.bias": old_actor_bias,
            "trace_value_head.weight": torch.zeros(1, 256),
            "trace_value_head.bias": torch.zeros(1),
            "critic_trace_encoder.0.weight": torch.zeros(1),
            "fixed_entropy_threshold": torch.tensor(0.5),
        }

        with self.assertRaisesRegex(RuntimeError, "Checkpoint architecture"):
            PolicyBackbone._load_model_state(model, checkpoint, "old-paper-checkpoint")

        self.assertFalse(torch.equal(model.actor.weight, old_actor_weight))
        self.assertFalse(torch.equal(model.actor.bias, old_actor_bias))
        self.assertTrue(
            torch.equal(model.trace_value_head.weight, current_critic_weight)
        )


if __name__ == "__main__":
    unittest.main()
