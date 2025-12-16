"""
Module for GNNs, including GCN, GAT, GraphSAGE, GIN
"""

import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from atomprop.embeddings.AtomEmbedding import BondTypes, BondDirections, AtomChirals

class MaskedBCELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(MaskedBCELoss, self).__init__()
        self.reduction = reduction
        
    def forward(self, pred, label):
        mask = (label != -1)
        valid_labels = label[mask].float()
        valid_preds = pred[mask]
        
        if valid_labels.numel() == 0:
            return torch.tensor(0.0, device=pred.device)
        loss = F.binary_cross_entropy_with_logits(
            valid_preds, valid_labels, reduction=self.reduction
        )
        return loss

class MaskedFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(MaskedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, pred, label):
        """
        pred: logits tensor of shape (N, *)
        label: label tensor of shape (N, *), values in {0, 1, -1}
        -1 indicates missing labels
        """
        mask = (label != -1)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device)
            
        valid_labels = label[mask].float()
        valid_preds = pred[mask]
        
        # Convert logits to probabilities
        p = torch.sigmoid(valid_preds)
        # Focal Loss calculation
        ce_loss = F.binary_cross_entropy_with_logits(valid_preds, valid_labels, reduction='none')
        # pt = p if y=1, else 1-p
        p_t = p * valid_labels + (1 - p) * (1 - valid_labels)
        # alpha_t = alpha if y=1, else 1-alpha
        alpha_t = self.alpha * valid_labels + (1 - self.alpha) * (1 - valid_labels)
        # Focal loss
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-1, reduction='mean'):
        """
        Masked CrossEntropy Loss for multi-class classification with missing labels.
        
        Args:
            ignore_index: Value that indicates missing/invalid labels (default: -1)
            reduction: Reduction method: 'mean', 'sum', or 'none'
        """
        super(MaskedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        
    def forward(self, pred, label):
        """
        Compute masked cross-entropy loss.
        
        Args:
            pred: Logits tensor of shape (N, C) or (N, C, *), where C is number of classes
            label: Label tensor of shape (N, *) with values in {0, 1, ..., C-1, ignore_index}
        
        Returns:
            Loss value
        """
        # Create mask for valid labels (non-ignore_index)
        mask = (label != self.ignore_index)
        
        # If no valid labels, return zero loss
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # Flatten the tensors if needed (for handling multi-dimensional cases)
        if pred.dim() > 2:
            # For semantic segmentation or similar tasks
            N, C = pred.shape[0], pred.shape[1]
            pred = pred.permute(0, 2, 3, *range(4, pred.dim()), 1).contiguous()
            pred = pred.view(-1, C)  # (N*H*W*..., C)
            label = label.view(-1)   # (N*H*W*...)
            mask = mask.view(-1)     # (N*H*W*...)
        
        # Get valid elements
        valid_labels = label[mask].long()  # Convert to long for indexing
        valid_preds = pred[mask] if pred.dim() > 1 else pred[mask].unsqueeze(-1)
        
        # Apply cross entropy loss
        loss = F.cross_entropy(
            valid_preds, 
            valid_labels, 
            reduction=self.reduction
        )
        return loss

class GCNconv(MessagePassing):
    """
    GCN layer.
    """
    def __init__(self, embed_dim, aggr='add'):
        super(GCNconv, self).__init__(aggr=aggr)
        self.lin = nn.Linear(embed_dim, embed_dim)
        self.root_emb = nn.Parameter(torch.zeros(embed_dim))
        self.edge_type_embedding = nn.Embedding(len(BondTypes.get_bond_types())+1, embed_dim)
        self.edge_direction_embedding = nn.Embedding(len(BondDirections.get_bond_directions())+1, embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        nn.init.zeros_(self.root_emb)
        nn.init.xavier_uniform_(self.edge_type_embedding.weight)
        nn.init.xavier_uniform_(self.edge_direction_embedding.weight)

    def normalize(self, edge_index, num_nodes):
        # Compute normalization
        row, col = edge_index
        deg = torch.bincount(row, minlength=num_nodes).float()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return norm

    def forward(self, x, edge_index, edge_attr):
        # x has shape [B_N, embed_dim]
        # edge_index has shape [2, E]
        # edge_attr has shape [E, 2]
        num_nodes = x.size(0)
        edge_index, _ = torch_geometric.utils.add_self_loops(edge_index, num_nodes=num_nodes)
        norm = self.normalize(edge_index, num_nodes)  # Shape [E]
        x = self.lin(x)  # Shape [B_N, embed_dim]
        edge_embeddings = self.edge_type_embedding(edge_attr[:,0]) + self.edge_direction_embedding(edge_attr[:,1])
        out = self.propagate(edge_index=edge_index, x=x, norm=norm, edge_attr=edge_embeddings)  # Shape [B_N, embed_dim]
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
        super(GATconv, self).__init__(aggr=aggr)
        self.att_type = att
        self.att_scope = attt
        self.embed_dim = embed_dim
        self.lin = nn.Linear(embed_dim, embed_dim)
        self.edge_type_embedding = nn.Embedding(len(BondTypes.get_bond_types())+1, embed_dim)
        self.edge_direction_embedding = nn.Embedding(len(BondDirections.get_bond_directions())+1, embed_dim)
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

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        if self.att is not None:
            nn.init.xavier_uniform_(self.att.view(1, -1))
        nn.init.zeros_(self.root_emb)
        nn.init.xavier_uniform_(self.edge_type_embedding.weight)
        nn.init.xavier_uniform_(self.edge_direction_embedding.weight)

    def forward(self, x, edge_index, edge_attr):
        # x has shape [B_N, embed_dim]
        # edge_index has shape [2, E]
        # edge_attr has shape [E, 2]
        x = self.lin(x)  # Shape [B_N, embed_dim]
        edge_embeddings = self.edge_type_embedding(edge_attr[:,0]) + self.edge_direction_embedding(edge_attr[:,1])
        out = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_embeddings)  # Shape [B_N, embed_dim]
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
            alpha = torch_geometric.utils.softmax(alpha, index, ptr, size_i)  # Softmax over neighbors of i
        elif self.att_scope == 'global':
            alpha = torch_geometric.nn.softmax(alpha, index)  # Softmax over all nodes
        else:
            raise ValueError("Invalid attention scope. Choose 'local' or 'global'.")

        return x_j * alpha.view(-1, 1)  # Shape [E, embed_dim]

class GraphSAGEconv(MessagePassing):
    """
    GraphSAGE layer.
    """
    def __init__(self, embed_dim, aggr='mean', sample=False):
        super(GraphSAGEconv, self).__init__(aggr=aggr)
        self.lin = nn.Linear(2 * embed_dim, embed_dim)
        self.root_emb = nn.Parameter(torch.zeros(embed_dim))
        self.edge_type_embedding = nn.Embedding(len(BondTypes.get_bond_types())+1, embed_dim)
        self.edge_direction_embedding = nn.Embedding(len(BondDirections.get_bond_directions())+1, embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)
        nn.init.zeros_(self.root_emb)
        nn.init.xavier_uniform_(self.edge_type_embedding.weight)
        nn.init.xavier_uniform_(self.edge_direction_embedding.weight)

    def forward(self, x, edge_index, edge_attr):
        # x has shape [B_N, embed_dim]
        # edge_index has shape [2, E]
        # edge_attr has shape [E, 2]
        edge_embeddings = self.edge_type_embedding(edge_attr[:,0]) + self.edge_direction_embedding(edge_attr[:,1])
        out = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_embeddings)  # Shape [B_N, embed_dim]
        out = torch.cat([out, x], dim=-1)  # Shape [N, 2*embed_dim]
        out = self.lin(out)  # Shape [B_N, embed_dim]
        out = out + self.root_emb * x  # Add skip connection
        return out

    def message(self, x_j):
        return x_j

class GINconv(MessagePassing):
    """
    GIN layer.
    """
    def __init__(self, embed_dim, aggr='add'):
        super(GINconv, self).__init__(aggr=aggr)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 2 * embed_dim),
            nn.ReLU(),
            nn.Linear(2 * embed_dim, embed_dim)
        )
        self.eps = nn.Parameter(torch.zeros(1))
        self.edge_type_embedding = nn.Embedding(len(BondTypes.get_bond_types())+1, embed_dim)
        self.edge_direction_embedding = nn.Embedding(len(BondDirections.get_bond_directions())+1, embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.eps)
        nn.init.xavier_uniform_(self.edge_type_embedding.weight)
        nn.init.xavier_uniform_(self.edge_direction_embedding.weight)

    def forward(self, x, edge_index, edge_attr):
        # x has shape [B_N, embed_dim]
        # edge_index has shape [2, E]
        # edge_attr has shape [E, 2]
        edge_embeddings = self.edge_type_embedding(edge_attr[:,0]) + self.edge_direction_embedding(edge_attr[:,1])
        out = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_embeddings)  # Shape [B_N, embed_dim]
        out = (1 + self.eps) * x + out  # Shape [B_N, embed_dim]
        out = self.mlp(out)  # Shape [B_N, embed_dim]
        return out

    def message(self, x_j):
        return x_j

class Embedder(nn.Module):
    """
    A module for embedding atom types and atom chirals.
    """
    def __init__(self, num_atom_types, embed_dim):
        super(Embedder, self).__init__()
        self.embedding = nn.Embedding(num_atom_types, embed_dim)
        self.embedding_chiral = nn.Embedding(len(AtomChirals.get_atom_chirals())+1, embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.embedding_chiral.weight)

    def forward(self, atom_attr):
        return self.embedding(atom_attr[:,0]) + self.embedding_chiral(atom_attr[:,1])  # Shape [B_N, embed_dim]

class GNN(nn.Module):
    def __init__(self, num_layers, embed_dim, dropout, gnn_type='gcn', JK='last', **kwargs):
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
        
    def forward(self, data):
        # x has shape [B_N, embed_dim]
        # edge_index has shape [2, E]
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        assert edge_attr is not None
        layer_outputs = []
        for conv in self.convs:
            x = conv(x=x, edge_index=edge_index, edge_attr=edge_attr)  # Shape [B_N, embed_dim]
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
            out = self.jump(out)  # Shape [B_N, embed_dim]
            assert False, "Concat Processing Not Implemented Yet. There will be dimension mismatch."
        else:
            raise ValueError("Invalid JK type. Choose from 'last', 'sum', 'max', 'concat'.")
        
        return out # Shape [B_N, embed_dim]

class GNNAggr(nn.Module):
    """
    A module for graph-level representation by aggregating node features.
    """
    def __init__(self, embed_dim, aggr='mean', layers=None, head=8):
        super(GNNAggr, self).__init__()
        self.aggr = aggr
        self.aggr_fn = None
        self.layers = layers
        
        if aggr == 'mean':
            self.aggr_fn = torch_geometric.nn.global_mean_pool
        elif aggr == 'sum':
            self.aggr_fn = torch_geometric.nn.global_add_pool
        elif aggr == 'max':
            self.aggr_fn = torch_geometric.nn.global_max_pool
        elif aggr == 'min':
            self.aggr_fn = torch_geometric.nn.global_min_pool
        elif aggr == 'attention':
            assert layers is not None, "Layers must be specified for attention aggregation"
            
            self.layers = layers
            
            # Initialize multi-head attention layers if multiple layers are requested
            if layers > 1:
                self.attns = nn.ModuleList([
                    nn.MultiheadAttention(embed_dim, head, batch_first=True) 
                    for _ in range(layers - 1)
                ])
            else:
                self.attns = nn.ModuleList()
            
            # Global attention with learnable gating mechanism
            self.aggr_fn = torch_geometric.nn.GlobalAttention(
                gate_nn=nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.ReLU(),
                    nn.Linear(embed_dim, 1)
                )
            )
        else:
            raise ValueError("Invalid aggregation type. Choose from 'mean', 'sum', 'max', 'min', 'attention'.")

    def forward(self, x, batch):
        """
        Forward pass for graph-level aggregation.
        
        Args:
            x (torch.Tensor): Node features with shape [num_nodes, embed_dim]
            batch (torch.Tensor): Batch indices with shape [num_nodes]
            
        Returns:
            torch.Tensor: Graph-level representations with shape [batch_size, embed_dim]
        """
        # Apply multi-head attention layers if using attention aggregation with multiple layers
        if self.aggr == 'attention' and self.layers > 1:
            # Reshape for batch_first attention (if needed)
            for attn in self.attns:
                x_reshaped = x.unsqueeze(0)  # Add batch dimension
                attn_output, _ = attn(x_reshaped, x_reshaped, x_reshaped)
                x = attn_output.squeeze(0)
        
        # Apply the selected aggregation function
        return self.aggr_fn(x, batch)  # Shape: [batch_size, embed_dim]
    