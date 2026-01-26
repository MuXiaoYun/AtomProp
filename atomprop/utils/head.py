"""
Library module for mlps
"""

import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from atomprop.utils.mlp import MLP
from atomprop.models.GeAT import GlobalAttnConv
from typing import Optional

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
                 batch_norm = False,
                 hidden_activation = F.relu,
                 output_activation = None):
        super().__init__()
        self.attn = GlobalAttnConv(
            embed_dim=input_dim,
            dropout=dropout,
            attn_num_layers=attn_num_layers
        )
        self.adaptor = MLP(
            input_dim = input_dim,
            hidden_dim = bottle_neck_dim,
            output_dim = input_dim,
            num_layers = bottle_neck_layers,
            dropout = dropout,
            batch_norm = batch_norm,
            hidden_activation = hidden_activation,
            output_activation = output_activation
        )
        self.mlp = MLP(
            input_dim = input_dim,
            hidden_dim = hidden_dim,
            output_dim = output_dim,
            num_layers = mlp_num_layers,
            dropout = dropout,
            batch_norm = batch_norm,
            hidden_activation = hidden_activation,
            output_activation = output_activation
        )
        
    def forward(self, x, batch):
        attn_emb = self.attn(x, None, batch)
        adapted_emb = self.adaptor(attn_emb)
        aggr_emb = torch_geometric.nn.global_mean_pool(attn_emb, batch)
        outputs = self.mlp(aggr_emb)
        return outputs
    