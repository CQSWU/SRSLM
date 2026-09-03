import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from agents.caar import CAAR


class CAARArtifactSnapshotTests(unittest.TestCase):
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
                "agents.caar.torch.load",
                side_effect=replace_path_after_snapshot,
            ):
                checkpoint, digest = CAAR._load_checkpoint_path(
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
            config, digest = CAAR._load_config_snapshot(path)
            path.write_text('{"full_config": {"seed": 99}}')

            self.assertEqual(config["full_config"]["seed"], 7)
            self.assertEqual(digest, hashlib.sha256(payload).hexdigest())


if __name__ == "__main__":
    unittest.main()
