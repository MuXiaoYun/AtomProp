"""
Library module for heads
"""

import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from atomprop.utils.mlp import MLP
from atomprop.models.GeAT import GlobalAttnConv
from typing import Optional

class GatedPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.gate = nn.Linear(input_dim, 1)

    def forward(self, x, batch):
        # x: [num_nodes, dim]
        gate = torch.sigmoid(self.gate(x))          # [num_nodes, 1]
        x = x * gate                                
        return torch_geometric.nn.global_add_pool(x, batch)

class DownstreamHead(nn.Module):
    def __init__(self, 
                 input_dim: int,
                 bottle_neck_dim: int,
                 bottle_neck_layers: int,
                 hidden_dim: int,
                 output_dim: int,
                 mlp_num_layers: int,
                 attn_num_layers: int,
                 dropout: float = 0.3,
                 hidden_activation = F.relu,
                 output_activation = None):
        super().__init__()

        self.attn = None
        if attn_num_layers > 0:
            self.attn = GlobalAttnConv(
                embed_dim=input_dim,
                dropout=dropout,
                attn_num_layers=attn_num_layers
            )

        # node-level adapter
        self.node_adapter = MLP(
            input_dim=input_dim,
            hidden_dim=bottle_neck_dim,
            output_dim=input_dim,
            num_layers=bottle_neck_layers,
            dropout=dropout,
            batch_norm=False,
            hidden_activation=hidden_activation,
            output_activation=None
        )

        self.norm = nn.LayerNorm(input_dim)

        # graph-level gated pooling
        self.pool = GatedPooling(input_dim)

        self.graph_adapter = MLP(
            input_dim=input_dim,
            hidden_dim=input_dim // 4,   # 512 -> 128 -> 512
            output_dim=input_dim,
            num_layers=2,
            dropout=dropout,
            batch_norm=False,
            hidden_activation=hidden_activation,
            output_activation=None
        )

        self.mlp = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=mlp_num_layers,
            dropout=dropout,
            batch_norm=False,
            hidden_activation=hidden_activation,
            output_activation=output_activation
        )

    def forward(self, x, batch):
        if self.attn is not None:
            x = self.norm(self.attn(x, None, batch) + x)

        x = self.node_adapter(x)

        # graph embedding
        g = self.pool(x, batch)

        # graph-level adapter + residual
        g = g + self.graph_adapter(g)

        out = self.mlp(g)
        return out
