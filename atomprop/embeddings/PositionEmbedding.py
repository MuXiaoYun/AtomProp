"""
Library module for position embeddings in 3D space.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from atomprop.utils.mlp import MLP        

class Learned3DPositionEmbedding(nn.Module):
    """
    A :class:`Learned3DPositionEmbedding` is a module that implements learned position embeddings for relative 3D coordinates.
    It uses a MLP to map relative 3D coordinates to embeddings.
    """

    def __init__(self, hidden_dim: int, output_dim: int, num_layers: int, output_activation: bool = False, dropout: float = 0.0, negative_slope: float = 0.2):
        super(Learned3DPositionEmbedding, self).__init__()
        self.mlp = MLP(input_dim=3, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, output_activation=output_activation, dropout=dropout, negative_slope=negative_slope)

    def forward(self, relative_positions: torch.Tensor):
        """
        Forward pass of the Learned3DPositionEmbedding.
        """
        return self.mlp(relative_positions)

class Sinusoidal3DPositionEmbedding:
    """
    A :class:`Sinusoidal3DPositionEmbedding` is a module that implements sinusoidal position embeddings for 3D coordinates.
    It uses sine and cosine functions of different frequencies to encode the positions.
    """

    def __init__(self, num_pos_feats: int, temperature: int = 10000, scale: float = 2 * math.pi):
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.scale = scale

    def __call__(self, positions: torch.Tensor):
        """
        Embed the input positions using sinusoidal functions. Clip the output to num_pos_feats.
        :param positions: Input positions of shape (batch_size, num_atoms, 3)
        :return: Sinusoidal position embeddings of shape (batch_size, num_atoms, num_pos_feats)
        """
        assert positions.dim() == 3 and positions.size(-1) == 3, "Input positions must be of shape (batch_size, num_atoms, 3)"
        batch_size, num_atoms, _ = positions.size()
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=positions.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = positions[:, :, 0] * self.scale
        pos_y = positions[:, :, 1] * self.scale
        pos_z = positions[:, :, 2] * self.scale

        pos_x = pos_x[:, :, None] / dim_t
        pos_y = pos_y[:, :, None] / dim_t
        pos_z = pos_z[:, :, None] / dim_t

        pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=3).flatten(2)
        pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=3).flatten(2)
        pos_z = torch.stack((pos_z.sin(), pos_z.cos()), dim=3).flatten(2)

        pos = torch.cat((pos_x, pos_y, pos_z), dim=2)
        return pos[:, :, :self.num_pos_feats]