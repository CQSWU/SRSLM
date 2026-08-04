import torch
from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.encoder import Encoder, ResBlock
from sample_factory.model.model_utils import create_mlp, nonlinearity
from torch import nn


SUPPORTED_COORDINATE_ENCODINGS = ("absolute_v1",)


class EPOMEncoder(Encoder):
    """Encoder architecture used by the original EPOM checkpoint."""

    def __init__(self, cfg, obs_space):
        super().__init__(cfg)
        settings = cfg.full_config["experiment_settings"]
        obs_shape = obs_space["obs"].shape
        channels = settings["pogema_encoder_num_filters"]
        num_blocks = settings["pogema_encoder_num_res_blocks"]
        self.coordinate_encoding = getattr(
            cfg,
            "coordinate_encoding",
            "absolute_v1",
        )
        if self.coordinate_encoding not in SUPPORTED_COORDINATE_ENCODINGS:
            raise ValueError(
                "coordinate_encoding must be one of "
                f"{SUPPORTED_COORDINATE_ENCODINGS}, got "
                f"{self.coordinate_encoding!r}."
            )

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
        num_fc_layers = max(1, getattr(cfg, "encoder_extra_fc_layers", 1))
        self.fc_blocks = create_mlp(
            [cfg.hidden_size for _ in range(num_fc_layers)],
            self.conv_head_out_size + cfg.hidden_size,
            nonlinearity(cfg),
        )
        self.encoder_out_size = cfg.hidden_size

    def _coordinate_features(self, observations):
        # Keep the original absolute-coordinate preprocessing byte-for-byte
        # equivalent for old checkpoints.
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
        coordinates = self.coordinates_mlp(
            self._coordinate_features(observations)
        )

        spatial = self.conv_head(observations["obs"])
        spatial = spatial.contiguous().view(-1, self.conv_head_out_size)
        return self.fc_blocks(torch.cat([spatial, coordinates], dim=-1))

    def get_out_size(self):
        return self.encoder_out_size

    def get_encoder_out_size(self):
        return self.get_out_size()
