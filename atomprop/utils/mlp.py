"""
Library module for mlps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class MLP(nn.Module):
    """
    A :class:`MLP` is a module that implements a multi-layer perceptron.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 num_layers: int,
                 dropout: float = 0.1,
                 batch_norm = False,
                 hidden_activation = F.relu,
                 output_activation = None,
                 zero_init: bool = False):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.dropout = dropout
        self.zero_init = zero_init

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
            
        self.init_params()

    def init_params(self, gain: float = 1.0):
        """
        Initialize the parameters of the MLP.
        Args:
            gain (float): Gain value for Xavier initialization.
        """
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                if self.zero_init and i == len(self.layers)-1:
                    nn.init.zeros_(layer.weight)
                else:
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

class MoE(nn.Module):
    """
    Mixture of Experts (MoE) layer.
    Replaces a standard FFN with N experts. Each input token is routed to top-k experts.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        num_experts: int = 8,
        top_k: int = 2,
        expert_hidden_layers: int = 2,
        dropout: float = 0.0,
        batch_norm: bool = False,
        hidden_activation=F.relu,
        output_activation=None,
        gating_dropout: float = 0.2,
    ):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim
        self.top_k = top_k
        self.gating_dropout = gating_dropout

        assert top_k <= num_experts, "top_k cannot be greater than num_experts"

        # 1. Gating network: maps input to logits over experts
        self.gate = nn.Linear(input_dim, num_experts, bias=False)

        # 2. Experts: list of MLPs (each is an expert)
        self.experts = nn.ModuleList([
            MLP(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=self.output_dim,
                num_layers=expert_hidden_layers,
                dropout=dropout,
                batch_norm=batch_norm,
                hidden_activation=hidden_activation,
                output_activation=output_activation,
                zero_init=True
            )
            for _ in range(num_experts)
        ])

        self.init_weights()

    def init_weights(self):
        # Initialize gate with small weights to encourage balanced routing early on
        nn.init.normal_(self.gate.weight, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sparse top-k routing.
        Args:
            x: (batch_size, seq_len, input_dim) or (N, input_dim)
        Returns:
            out: same shape as x (with output_dim in last dim)
        """
        original_shape = x.shape
        # Flatten to (N, D) where N = batch_size * seq_len
        if x.dim() == 3:
            batch_size, seq_len, _ = x.shape
            x_flat = x.view(-1, self.input_dim)  # (B*L, D)
        else:
            x_flat = x
            batch_size = x.size(0)
            seq_len = 1

        N = x_flat.size(0)

        # 1. Compute gating logits
        gate_logits = self.gate(x_flat)  # (N, E)
        if self.training and self.gating_dropout > 0:
            gate_logits = F.dropout(gate_logits, p=self.gating_dropout, training=True)

        # 2. Top-k selection
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=1)  # (N, k), (N, k)
        top_k_scores = F.softmax(top_k_logits, dim=1)  # (N, k)

        # 3. Prepare output tensor
        out = torch.zeros(N, self.output_dim, device=x.device, dtype=x.dtype)

        # 4. Route to experts (loop over k experts)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]  # (N,)
            expert_score = top_k_scores[:, i]  # (N,)

            # Group tokens by expert for efficient computation (optional but recommended for large N/E)
            # Here we use simple loop (fine for moderate num_experts)
            for e in range(self.num_experts):
                mask = (expert_idx == e)
                if mask.any():
                    expert_out = self.experts[e](x_flat[mask])  # (M, D_out)
                    out[mask] += expert_score[mask].unsqueeze(1) * expert_out

        # Reshape back
        if len(original_shape) == 3:
            out = out.view(batch_size, seq_len, self.output_dim)
        
        return out