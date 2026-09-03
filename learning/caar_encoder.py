"""Shared recurrent policy encoder for CAAR and NoReweight."""

import torch
from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.encoder import Encoder, ResBlock
from sample_factory.model.model_utils import create_mlp, nonlinearity
from torch import nn as nn


class CAAREncoder(Encoder):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)
        obs_shape = obs_space["obs"].shape
        self.context_ch = min(3, obs_shape[0])

        hidden = cfg.hidden_size
        height, width = obs_shape[1], obs_shape[2]
        num_filters = getattr(cfg, "caar_num_filters", 64)
        num_res = getattr(cfg, "caar_num_res_blocks", 3)

        layers = [nn.Conv2d(self.context_ch, num_filters, 3, stride=1, padding=1)]
        for _ in range(num_res):
            layers.append(ResBlock(cfg, num_filters, num_filters))
        layers.append(nonlinearity(cfg))
        self.spatial_conv = nn.Sequential(*layers)
        self.conv_out_size = calc_num_elements(
            self.spatial_conv,
            (self.context_ch, height, width),
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(4, hidden),
            nonlinearity(cfg),
            nn.Linear(hidden, hidden),
            nonlinearity(cfg),
        )
        num_fc_layers = max(1, getattr(cfg, "encoder_extra_fc_layers", 1))
        self.output = create_mlp(
            [hidden for _ in range(num_fc_layers)],
            self.conv_out_size + hidden,
            nonlinearity(cfg),
        )
        self.encoder_out_size = hidden

    def forward(self, x):
        context = x["obs"][:, : self.context_ch]
        spatial_feat = self.spatial_conv(context)
        spatial_vec = spatial_feat.contiguous().view(spatial_feat.size(0), -1)

        coords = torch.cat([x["xy"], x["target_xy"]], dim=-1).float()
        scale = 64.0
        coords = coords / torch.maximum(
            torch.abs(coords),
            torch.tensor(scale, device=coords.device, dtype=coords.dtype),
        )
        coord_feat = self.coord_mlp(coords)
        return self.output(torch.cat([spatial_vec, coord_feat], dim=-1))

    def get_out_size(self) -> int:
        return self.encoder_out_size
