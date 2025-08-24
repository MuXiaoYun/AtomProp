"""
Library module for mlps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    """
    A :class:`MLP` is a module that implements a multi-layer perceptron.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, output_activation: bool = False, dropout: float = 0.0, negative_slope: float = 0.2):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.output_activation = output_activation
        self.dropout = dropout
        self.negative_slope = negative_slope

        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.layers = nn.ModuleList(layers)
        self.activation = nn.LeakyReLU(negative_slope)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        for i in range(self.num_layers):
            x = self.layers[i](x)
            if i != self.num_layers - 1 or self.output_activation:
                x = self.activation(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x