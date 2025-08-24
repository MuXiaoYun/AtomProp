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


def Sinusoidal3DPositionEmbedding(num_pos_feats: int, temperature: int = 10000, scale: float = 2 * math.pi):
    """
    This function generates sinusoidal position embeddings for 3D coordinates.
    """
    if num_pos_feats % 3 != 0:
        raise ValueError("num_pos_feats should be divisible by 3")
    num_feats = num_pos_feats // 3
    dim_t = torch.arange(num_feats, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_feats)

    def embed(positions):
        if positions.dim() == 2:
            x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        elif positions.dim() == 3:
            x, y, z = positions[:, :, 0], positions[:, :, 1], positions[:, :, 2]
        else:
            raise ValueError("positions should be of shape (N, 3) or (N, M, 3)")

        x = x[:, None] / dim_t
        y = y[:, None] / dim_t
        z = z[:, None] / dim_t

        pos_x = torch.stack((x.sin(), x.cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((y.sin(), y.cos()), dim=-1).flatten(-2)
        pos_z = torch.stack((z.sin(), z.cos()), dim=-1).flatten(-2)

        pos = torch.cat((pos_x, pos_y, pos_z), dim=-1)
        return pos * scale

    return embed
