import copy
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from agents.epom import EPOM, OFFICIAL_EPOM_CHECKPOINT
from learning.epom_finetune_actor_critic import EPOMFineTuneActorCritic
from pomapf_env.wrappers import GridMemoryObservationWrapper


def _find_wrapper(env, wrapper_type):
    current = env
    while current is not None:
        if isinstance(current, wrapper_type):
            return current
        child = getattr(current, "env", None)
        if child is current:
            break
        current = child
    return None


class EPOMLifelongFineTuneConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]

    def test_smoke_and_formal_configs_keep_the_audited_contract(self):
        from train import validate_config

        expected_steps = {
            "train_epom_lifelong_finetune_r5_smoke.yaml": 1_048_576,
            "train_epom_lifelong_finetune_r5_100m.yaml": 100_000_000,
        }
        for filename, steps in expected_steps.items():
            with self.subTest(filename=filename):
                config = yaml.safe_load(
                    (self.root / "learning" / filename).read_text(encoding="utf-8")
                )
                experiment, flat = validate_config(config)
                grid = experiment.environment.grid_config
                self.assertEqual(experiment.environment.name, "POMAPF-EPOM-v0")
                self.assertEqual(experiment.environment.grid_memory_obs_radius, 7)
                self.assertEqual(grid.obs_radius, 5)
                self.assertEqual(grid.num_agents, 200)
                self.assertEqual(grid.max_episode_steps, 512)
                self.assertEqual(grid.on_target, "restart")
                self.assertEqual(grid.collision_system, "block_both")
                self.assertEqual(grid.map_name, "maps/train.yaml")
                self.assertFalse(flat.normalize_input)
                self.assertEqual(flat.train_for_env_steps, steps)
                self.assertEqual(flat.hidden_size, 512)
                self.assertEqual(flat.recurrence, 32)
                self.assertEqual(flat.rollout, 32)

    def test_invalid_normalization_is_rejected(self):
        from pydantic import ValidationError
        from train import validate_config

        path = (
            self.root
            / "learning"
            / "train_epom_lifelong_finetune_r5_smoke.yaml"
        )
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        config["experiment_settings"]["normalize_input"] = True
        with self.assertRaisesRegex(ValidationError, "normalize_input=false"):
            validate_config(config)

    def test_formal_config_can_extend_in_place_to_250m(self):
        from train import validate_config

        path = (
            self.root
            / "learning"
            / "train_epom_lifelong_finetune_r5_100m.yaml"
        )
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        config["experiment_settings"]["train_for_env_steps"] = 250_000_000
        _, flat = validate_config(config)
        self.assertEqual(flat.train_for_env_steps, 250_000_000)


class EPOMLifelongFineTuneIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]
        weights = root / "weights" / "EPOM" / "EPOM"
        if not weights.is_dir():
            raise unittest.SkipTest("Official EPOM weights are unavailable.")

        from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
        from sample_factory.algo.utils.tensor_dict import TensorDict
        from sample_factory.envs.create_env import create_env
        from sample_factory.model.actor_critic import (
            create_actor_critic,
            default_make_actor_critic_func,
        )
        from sample_factory.model.model_utils import get_rnn_size
        from train import register_custom_components, validate_config

        config_path = (
            root
            / "learning"
            / "train_epom_lifelong_finetune_r5_smoke.yaml"
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        register_custom_components()
        _, cls.cfg = validate_config(config)
        cls.env = create_env(cls.cfg.env, cfg=cls.cfg, env_config={})
        cls.model = create_actor_critic(
            cls.cfg,
            cls.env.observation_space,
            cls.env.action_space,
        )
        cls.checkpoint = torch.load(
            weights / "checkpoint_p0" / OFFICIAL_EPOM_CHECKPOINT,
            map_location="cpu",
            weights_only=False,
        )

        reference_cfg = copy.deepcopy(cls.cfg)
        reference_cfg.encoder_custom = "pogema_residual"
        reference_cfg.full_config["experiment_settings"][
            "encoder_custom"
        ] = "pogema_residual"
        cls.reference = default_make_actor_critic_func(
            reference_cfg,
            cls.env.observation_space,
            cls.env.action_space,
        )
        EPOM._load_model(cls.reference, cls.checkpoint["model"])

        observations, _ = cls.env.reset()
        batch = TensorDict(
            {
                key: torch.from_numpy(
                    np.stack([observation[key] for observation in observations])
                ).float()
                for key in observations[0]
            }
        )
        cls.batch = prepare_and_normalize_obs(
            cls.model,
            TensorDict({key: value.clone() for key, value in batch.items()}),
        )
        cls.reference_batch = prepare_and_normalize_obs(
            cls.reference,
            TensorDict({key: value.clone() for key, value in batch.items()}),
        )
        cls.rnn_states = torch.zeros(
            (len(observations), get_rnn_size(cls.cfg)),
            dtype=torch.float32,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "env"):
            cls.env.close()

    def setUp(self):
        EPOM._load_model(self.model, self.checkpoint["model"])
        self.model.zero_grad(set_to_none=True)

    def test_epom_only_environment_has_r7_memory_and_no_tau(self):
        self.assertEqual(tuple(self.env.observation_space["obs"].shape), (3, 15, 15))
        self.assertNotIn("tau", self.env.observation_space.spaces)
        wrapper = _find_wrapper(self.env, GridMemoryObservationWrapper)
        self.assertIsNotNone(wrapper)
        self.assertEqual(wrapper.memory_radius, 7)

    def test_all_28_official_tensors_are_loaded_exactly(self):
        self.assertIsInstance(self.model, EPOMFineTuneActorCritic)
        state = self.model.state_dict()
        self.assertEqual(len(self.checkpoint["model"]), 28)
        for old_key, expected in self.checkpoint["model"].items():
            key = old_key.replace(
                "encoder.fc_after_enc.",
                "encoder.fc_blocks.",
                1,
            )
            with self.subTest(key=key):
                torch.testing.assert_close(state[key], expected, rtol=0, atol=0)

    def test_every_actor_critic_parameter_is_trainable(self):
        frozen = [
            name for name, parameter in self.model.named_parameters()
            if not parameter.requires_grad
        ]
        self.assertEqual(frozen, [])

    def test_step_zero_matches_a_separately_loaded_official_policy(self):
        self.model.eval()
        self.reference.eval()
        torch.manual_seed(1729)
        actual = self.model(self.batch, self.rnn_states.clone())
        torch.manual_seed(1729)
        expected = self.reference(
            self.reference_batch,
            self.rnn_states.clone(),
        )
        for key in ("action_logits", "values", "actions"):
            with self.subTest(key=key):
                torch.testing.assert_close(
                    actual[key],
                    expected[key],
                    rtol=0,
                    atol=0,
                )

    def test_fresh_optimizer_updates_encoder_gru_actor_and_critic(self):
        self.model.train()
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            eps=self.cfg.adam_eps,
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
        )
        self.assertEqual(len(optimizer.state), 0)
        outputs = self.model(self.batch, self.rnn_states.clone())
        loss = (
            outputs["action_logits"].square().mean()
            + outputs["values"].square().mean()
        )
        loss.backward()
        for prefix in (
            "encoder.",
            "core.",
            "action_parameterization.",
            "critic_linear.",
        ):
            gradients = [
                parameter.grad
                for name, parameter in self.model.named_parameters()
                if name.startswith(prefix) and parameter.grad is not None
            ]
            with self.subTest(prefix=prefix):
                self.assertTrue(gradients)
                self.assertGreater(
                    sum(float(gradient.abs().sum()) for gradient in gradients),
                    0.0,
                )
        actor = self.model.action_parameterization.distribution_linear.weight
        before = actor.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, actor.detach()))
        self.assertGreater(len(optimizer.state), 0)


if __name__ == "__main__":
    unittest.main()
