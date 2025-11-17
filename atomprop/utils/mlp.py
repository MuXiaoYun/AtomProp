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

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, dropout: float = 0.0, batch_norm = False, hidden_activation = F.relu, output_activation = None):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.dropout = dropout

        layers = []
        if num_layers == 1:
            layers.append(nn.Linear(input_dim, output_dim))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Linear(hidden_dim, output_dim))

        self.layers = nn.ModuleList(layers)

        self.batch_norm = None
        if batch_norm:
            # add a batch_norm layer between output layer and activation
            self.batch_norm = nn.BatchNorm1d(output_dim)

    def init_params(self, gain: float = 1.0):
        """
        Initialize the parameters of the MLP.

        Args:
            gain (float): Gain value for Xavier initialization.
        """
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

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
            if i != self.num_layers - 1:
                x = self.hidden_activation(x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.output_activation is not None:
            x = self.output_activation(x)
        return x