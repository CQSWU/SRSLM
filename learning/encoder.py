import torch
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.actor_critic import default_make_actor_critic_func
from sample_factory.model.encoder import Encoder, ResBlock, default_make_encoder_func
from sample_factory.model.model_utils import create_mlp, nonlinearity
from torch import nn


class EPOMEncoder(Encoder):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)
        settings = cfg.full_config["experiment_settings"]
        obs_shape = obs_space["obs"].shape
        channels = settings["pogema_encoder_num_filters"]
        num_blocks = settings["pogema_encoder_num_res_blocks"]

        layers = [nn.Conv2d(obs_shape[0], channels, kernel_size=3, padding=1)]
        for _ in range(num_blocks):
            layers.append(ResBlock(cfg, channels, channels))
        layers.append(nonlinearity(cfg))
        self.conv_head = nn.Sequential(*layers)
        self.conv_head_out_size = calc_num_elements(self.conv_head, obs_shape)

        self.coordinates_mlp = nn.Sequential(
            nn.Linear(4, cfg.hidden_size),
            nn.ReLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.ReLU(),
        )
        self.fc_blocks = create_mlp(
            [cfg.hidden_size],
            self.conv_head_out_size + cfg.hidden_size,
            nonlinearity(cfg),
        )
        self.encoder_out_size = cfg.hidden_size

    def _coordinate_features(self, observations):
        coordinates = torch.cat(
            [observations["xy"], observations["target_xy"]],
            dim=-1,
        )
        scale = torch.maximum(
            torch.abs(coordinates),
            torch.tensor(
                64.0,
                device=coordinates.device,
                dtype=coordinates.dtype,
            ),
        )
        return coordinates / scale

    def forward(self, observations):
        coordinates = self.coordinates_mlp(self._coordinate_features(observations))

        spatial = self.conv_head(observations["obs"])
        spatial = spatial.contiguous().view(-1, self.conv_head_out_size)
        return self.fc_blocks(torch.cat([spatial, coordinates], dim=-1))

    def get_out_size(self):
        return self.encoder_out_size


def _encoder_kind(cfg):
    kind = getattr(cfg, "encoder_custom", None)
    if kind not in {
        None,
        "pogema_residual",
        "epom_trace_context",
        "switcher",
    }:
        raise ValueError(f"Unsupported or retired encoder_custom: {kind!r}")
    return kind


def make_encoder(cfg, obs_space):
    kind = _encoder_kind(cfg)

    if kind == "switcher":
        from learning.switcher_actor_critic import SwitcherEncoder

        return SwitcherEncoder(cfg, obs_space)

    if kind in (
        "pogema_residual",
        "epom_trace_context",
    ):
        return EPOMEncoder(cfg, obs_space)

    return default_make_encoder_func(cfg, obs_space)


def make_actor_critic(cfg, obs_space, action_space):
    kind = _encoder_kind(cfg)

    if kind == "switcher":
        from learning.switcher_actor_critic import SwitcherActorCritic

        actor_critic = SwitcherActorCritic

    elif kind == "epom_trace_context":
        from learning.arpe_actor_critic import (
            ARPEActorCritic,
        )

        actor_critic = ARPEActorCritic
    else:
        return default_make_actor_critic_func(cfg, obs_space, action_space)

    return actor_critic(global_model_factory(), obs_space, action_space, cfg)


global_model_factory().register_encoder_factory(make_encoder)
global_model_factory().register_actor_critic_factory(make_actor_critic)
