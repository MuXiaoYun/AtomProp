"""
Module for GNNs, including GCN, GAT, GraphSAGE, GIN
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class GCNconv(MessagePassing):
    """
    GCN layer.
    """
    def __init__(self, embed_dim, aggr='add'):
        super(GCNconv, self).__init__()
        self.lin = nn.Linear(embed_dim, embed_dim)
        self.root_emb = nn.Parameter(torch.zeros(embed_dim))
        self.reset_parameters()
        self.aggr = aggr

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        nn.init.zeros_(self.root_emb)

    def normalize(self, edge_index, num_nodes):
        # Compute normalization
        row, col = edge_index
        deg = torch.bincount(row, minlength=num_nodes).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return norm

    def forward(self, x, edge_index):
        # x has shape [N, embed_dim]
        # edge_index has shape [2, E]
        num_nodes = x.size(0)
        edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=num_nodes)
        norm = self.normalize(edge_index, num_nodes)  # Shape [E]
        x = self.lin(x)  # Shape [N, embed_dim]
        out = self.propagate(self.aggr, edge_index, x=x, norm=norm)  # Shape [N, embed_dim]
        out = out + self.root_emb * x  # Add skip connection
        return out

    def message(self, x_j):
        return x_j

class GATconv(MessagePassing):
    """
    GAT layer.

    att: 'concat_linear' or 'dot_product'
    1. 'concat_linear': a^T [Wx_i || Wx_j]
    2. 'dot_product': (Wx_i)^T (Wx_j) / sqrt(d_k)

    attt: 'local' or 'global'
    1. 'local': softmax over neighbors of i
    2. 'global': softmax over all nodes
    """
    def __init__(self, embed_dim, output_negative_slope=0.2, aggr='add', att='concat_linear', attt='local'):
        super(GATconv, self).__init__()
        self.att_type = att
        self.att_scope = attt
        self.embed_dim = embed_dim
        self.lin = nn.Linear(embed_dim, embed_dim)
        if att == 'concat_linear':
            self.att = nn.Parameter(torch.Tensor(2 * embed_dim))
            nn.init.xavier_uniform_(self.att.view(1, -1))
        elif att == 'dot_product':
            self.att = None
            self.scale = embed_dim ** 0.5
        else:
            raise ValueError("Invalid attention type. Choose 'concat_linear' or 'dot_product'.")
        self.leaky_relu = nn.LeakyReLU(negative_slope=output_negative_slope)
        self.root_emb = nn.Parameter(torch.zeros(embed_dim))
        self.reset_parameters()
        self.aggr = aggr

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        if self.att is not None:
            nn.init.xavier_uniform_(self.att.view(1, -1))
        nn.init.zeros_(self.root_emb)

    def forward(self, x, edge_index):
        # x has shape [N, embed_dim]
        # edge_index has shape [2, E]
        x = self.lin(x)  # Shape [N, embed_dim]
        out = self.propagate(self.aggr, edge_index, x=x)  # Shape [N, embed_dim]
        out = out + self.root_emb * x  # Add skip connection
        return out

    def message(self, x_i, x_j, index, ptr, size_i):
        if self.att_type == 'concat_linear':
            # a^T [Wx_i || Wx_j]
            alpha = torch.cat([x_i, x_j], dim=-1)  # Shape [E, 2*embed_dim]
            alpha = (alpha * self.att).sum(dim=-1)  # Shape [E]
        elif self.att_type == 'dot_product':
            # (Wx_i)^T (Wx_j) / sqrt(d_k)
            alpha = (x_i * x_j).sum(dim=-1) / self.scale  # Shape [E]
        else:
            raise ValueError("Invalid attention type. Choose 'concat_linear' or 'dot_product'.")
        
        alpha = self.leaky_relu(alpha)

        if self.att_scope == 'local':
            alpha = torch.softmax(alpha, index, ptr, size_i)  # Softmax over neighbors of i
        elif self.att_scope == 'global':
            alpha = torch.softmax(alpha, dim=0)  # Softmax over all nodes
        else:
            raise ValueError("Invalid attention scope. Choose 'local' or 'global'.")

        return x_j * alpha.view(-1, 1)  # Shape [E, embed_dim]

class GraphSAGEconv(MessagePassing):
    """
    GraphSAGE layer.
    """
    def __init__(self, embed_dim, aggr='mean', sample=False):
        super(GraphSAGEconv, self).__init__()
        self.lin = nn.Linear(2 * embed_dim, embed_dim)
        self.root_emb = nn.Parameter(torch.zeros(embed_dim))
        self.reset_parameters()
        self.aggr = aggr

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        nn.init.zeros_(self.root_emb)

    def forward(self, x, edge_index):
        # x has shape [N, embed_dim]
        # edge_index has shape [2, E]
        out = self.propagate(self.aggr, edge_index, x=x)  # Shape [N, embed_dim]
        out = torch.cat([out, x], dim=-1)  # Shape [N, 2*embed_dim]
        out = self.lin(out)  # Shape [N, embed_dim]
        out = out + self.root_emb * x  # Add skip connection
        return out

    def message(self, x_j):
        return x_j

class GINconv(MessagePassing):
    """
    GIN layer.
    """
    def __init__(self, embed_dim, aggr='add'):
        super(GINconv, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.ReLU(),
            nn.Linear(2 * embed_dim, embed_dim)
        )
        self.eps = nn.Parameter(torch.zeros(1))
        self.reset_parameters()
        self.aggr = aggr

    def reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.eps)

    def forward(self, x, edge_index):
        # x has shape [N, embed_dim]
        # edge_index has shape [2, E]
        out = self.propagate(self.aggr, edge_index, x=x)  # Shape [N, embed_dim]
        out = (1 + self.eps) * x + out  # Shape [N, embed_dim]
        out = self.mlp(out)  # Shape [N, embed_dim]
        return out

    def message(self, x_j):
        return x_j

class GNN(nn.Module):
    def __init__(self, gnn_type='gcn', num_layers, embed_dim, JK='last', dropout, **kwargs):
        """
        A module for stacking multiple GNN layers.
        :param gnn_type: The type of GNN layer to use ('gcn', 'gat', 'graphsage', 'gin').
        :param num_layers: The number of GNN layers to stack.
        :param embed_dim: The embedding dimension.
        :param JK: The type of Jumping Knowledge to use ('last', 'sum', 'max', 'concat').
        :param dropout: The dropout rate.
        :param kwargs: Additional arguments for the GNN layers.
        """
        super(GNN, self).__init__()

        if num_layers < 2:
            raise ValueError("Number of GNN layers must be at least 2.")

        self.num_layers = num_layers
        self.JK = JK
        self.dropout = dropout

        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            if gnn_type == 'gcn':
                conv = GCNconv(embed_dim, **kwargs)
            elif gnn_type == 'gat':
                conv = GATconv(embed_dim, **kwargs)
            elif gnn_type == 'graphsage':
                conv = GraphSAGEconv(embed_dim, **kwargs)
            elif gnn_type == 'gin':
                conv = GINconv(embed_dim, **kwargs)
            else:
                raise ValueError("Invalid GNN type. Choose from 'gcn', 'gat', 'graphsage', 'gin'.")
            self.convs.append(conv)

        if JK == 'concat':
            self.jump = nn.Linear(num_layers * embed_dim, embed_dim)
            nn.init.xavier_uniform_(self.jump.weight)
            nn.init.zeros_(self.jump.bias)
        else:
            self.jump = None
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.jump is not None:
            nn.init.xavier_uniform_(self.jump.weight)
            nn.init.zeros_(self.jump.bias)
        
    def forward(self, x, edge_index):
        # x has shape [N, embed_dim]
        # edge_index has shape [2, E]
        layer_outputs = []
        for conv in self.convs:
            x = conv(x, edge_index)  # Shape [N, embed_dim]
            x = torch.relu(x)
            x = torch.dropout(x, p=self.dropout, train=self.training)
            layer_outputs.append(x)

        if self.JK == 'last':
            out = layer_outputs[-1]
        elif self.JK == 'sum':
            out = torch.stack(layer_outputs, dim=0).sum(dim=0)
        elif self.JK == 'max':
            out = torch.stack(layer_outputs, dim=0).max(dim=0)[0]
        elif self.JK == 'concat':
            out = torch.cat(layer_outputs, dim=-1)  # Shape [N, num_layers * embed_dim]
            out = self.jump(out)  # Shape [N, embed_dim]
        else:
            raise ValueError("Invalid JK type. Choose from 'last', 'sum', 'max', 'concat'.")
        
        return out
