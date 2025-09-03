"""
Module for Graph Edge Attention Transformer (GeAT) for molecular property prediction.
"""

import torch
import torch.nn as nn
from atomprop.embeddings.AtomEmbedding import AtomEmbedding
from atomprop.utils.mlp import MLP
import atomprop.embeddings.PositionEmbedding as PE
from atomprop.models.EdgeAttention import EdgeAttention, MultiHeadEdgeAttention
    
class GeATLayer(nn.Module):
    """
    A :class:`GeAT` (Graph Edge Attention Transformer) for molecular property prediction.
    This model weighs the importance of only neighboring atoms.
    """

    def __init__(self, atom_embedding_dim: int, num_bond_types: int, num_heads: int = 8, dropout: float = 0.2, output_negative_slope: float = 0.2, parallel_between_bondtypes: bool = True):
        super(GeATLayer, self).__init__()
        self.num_heads = num_heads
        self.atom_embedding_dim = atom_embedding_dim
        self.num_bond_types = num_bond_types
        # Linear layers for query, key, and value transformations
        self.Q_w = nn.Linear(atom_embedding_dim, atom_embedding_dim*num_heads)
        self.K_w = nn.Linear(atom_embedding_dim, atom_embedding_dim*num_heads)
        self.V_w = nn.Linear(atom_embedding_dim, atom_embedding_dim*num_heads)
        # for each bond type, we use a different attention mechanism
        self.edge_attentions = MultiHeadEdgeAttention(parallel_between_bondtypes=parallel_between_bondtypes, atom_embedding_dim=atom_embedding_dim, num_bond_types=num_bond_types, num_heads=num_heads, output_negative_slope=output_negative_slope)
        self.dropout_layer = nn.Dropout(dropout)
        self.project = nn.Linear(atom_embedding_dim * num_heads, atom_embedding_dim)

    def forward(self, atom_embeddings, edges = None):
        B = atom_embeddings.size(0)
        N = atom_embeddings.size(1)
        src_embeddings = self.Q_w(atom_embeddings)
        dst_embeddings = self.K_w(atom_embeddings)
        value_embeddings = self.V_w(atom_embeddings)
        # Calculate attention scores
        attention_scores = self.edge_attentions(src_embeddings, dst_embeddings, edges)
        # Softmax
        attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1)
        # Dropout
        attention_scores = self.dropout_layer(attention_scores)
        # Compute attention output, which is suppose to be (b, n, d*num_heads)
        attention_output = torch.matmul(attention_scores, value_embeddings.reshape(B, -1, self.num_heads, self.atom_embedding_dim).permute(0, 2, 1, 3)) # (b, num_heads, n, d)
        attention_output = attention_output.reshape(B, N, -1) # (b, n, d*num_heads)
        # Project to atom embedding dimension
        attention_output = self.project(attention_output)
        return attention_output
    
class GeATLayerWithSingleHead(nn.Module):
    """
    A :class:`GeATLayerWithSingleHead` is a simplified version of :class:`GeATLayer` that uses a single head for attention.
    This model weighs the importance of only neighboring atoms.
    """

    def __init__(self, atom_embedding_dim: int, num_bond_types: int, dropout: float = 0.2, output_negative_slope: float = 0.2):
        super(GeATLayerWithSingleHead, self).__init__()
        self.Q_w = nn.Linear(atom_embedding_dim, atom_embedding_dim)
        self.K_w = nn.Linear(atom_embedding_dim, atom_embedding_dim)
        self.V_w = nn.Linear(atom_embedding_dim, atom_embedding_dim)
        self.edge_attention = EdgeAttention(atom_embedding_dim, num_bond_types, output_negative_slope)
        self.dropout_layer = nn.Dropout(dropout)
        self.project = nn.Linear(atom_embedding_dim, atom_embedding_dim)

    def forward(self, atom_embeddings, edges = None):
        src_embeddings = self.Q_w(atom_embeddings)
        dst_embeddings = self.K_w(atom_embeddings)
        value_embeddings = self.V_w(atom_embeddings)
        # Calculate attention scores
        attention_scores = self.edge_attention(src_embeddings, dst_embeddings, edges)
        # Softmax
        attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1)
        # Dropout
        attention_scores = self.dropout_layer(attention_scores)
        # Compute attention output, which is suppose to be (b, n, d)
        attention_output = torch.matmul(attention_scores, value_embeddings)
        # Project to atom embedding dimension
        attention_output = self.project(attention_output)
        return attention_output

class GeATBackbone(nn.Module):
    """
    A :class:`GeATBackbone` is a module for molecular representation learning using GeAT. It outputs atom embeddings.
    """
    def __init__(self, atom_embedding_dim: int, num_atom_types: int, num_bond_types: int, num_heads: int = 8, output_negative_slope: float = 0.2, dropout: int = 0.2, geat_num_layers: int = 3, parallel_between_bondtypes: bool = True):
        super(GeATBackbone, self).__init__()
        self.atom_embedding = AtomEmbedding(atom_embedding_dim=atom_embedding_dim, num_atom_types=num_atom_types)
        self.geat_layers = nn.ModuleList([GeATLayer(atom_embedding_dim=atom_embedding_dim, num_bond_types=num_bond_types, num_heads=num_heads, output_negative_slope=output_negative_slope, dropout=dropout, parallel_between_bondtypes=parallel_between_bondtypes) for _ in range(geat_num_layers)])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(atom_embedding_dim) for _ in range(geat_num_layers)])

    def forward(self, atoms, edges):
        atom_embeddings = self.atom_embedding(atoms)
        for i, layer in enumerate(self.geat_layers):
            atom_embeddings = atom_embeddings + layer(atom_embeddings, edges)
            atom_embeddings = self.norm_layers[i](atom_embeddings)
        return atom_embeddings

class GeATNeck(nn.Module):
    """
    A :class:`GeATNeck` is a module for global attention mechanism in GeAT. It outputs a single vector representing the aggregated information from all atoms.
    """

    def __init__(self, atom_embedding_dim: int, num_bond_types: int, num_heads: int = 8, global_num_heads = 8, dropout: int = 0.2):
        super(GeATNeck, self).__init__()
        self.global_attention = nn.MultiheadAttention(embed_dim=atom_embedding_dim*global_num_heads, num_heads=global_num_heads, dropout=dropout)
        self.Q_w_global = nn.Linear(atom_embedding_dim, atom_embedding_dim*global_num_heads)
        self.K_w_global = nn.Linear(atom_embedding_dim, atom_embedding_dim*global_num_heads)
        self.V_w_global = nn.Linear(atom_embedding_dim, atom_embedding_dim*global_num_heads)
        self.norm_layer = nn.LayerNorm(atom_embedding_dim*global_num_heads)
        
    def forward(self, atom_embeddings):
        atom_embeddings = atom_embeddings.reshape(atom_embeddings.size(0), -1, atom_embeddings.size(-1))
        global_q = self.Q_w_global(atom_embeddings)
        global_k = self.K_w_global(atom_embeddings)
        global_v = self.V_w_global(atom_embeddings)
        global_attention_output, _ = self.global_attention(global_q, global_k, global_v)
        global_attention_output = self.norm_layer(global_attention_output)
        global_attention_output = global_attention_output.mean(dim=1)
        return global_attention_output

class GeATHead(nn.Module):
    """
    A :class:`GeATHead` is a module for molecular property prediction using GeAT. It outputs a single scalar value representing the predicted property of the molecule.
    """

    def __init__(self, atom_embedding_dim: int, num_heads: int = 8, global_num_heads = 8, mlp_hidden_dim: int = 64, output_negative_slope: float = 0.2, dropout: int = 0.2, mlp_num_layers: int = 2):
        super(GeATHead, self).__init__()
        self.output_mlp = MLP(input_dim=atom_embedding_dim*global_num_heads, hidden_dim=mlp_hidden_dim, output_dim=1, num_layers=mlp_num_layers, output_activation=False, dropout=dropout, negative_slope=output_negative_slope)
        self.output_sigmoid = nn.Sigmoid()
        
    def forward(self, global_attention_output):
        x = self.output_mlp(global_attention_output)
        x = self.output_sigmoid(x)
        return x

class GeATNet(nn.Module):
    """
    A :class:`GeATNet` is a module for molecular property prediction using GeAT. It outputs a single scalar value representing the predicted property of the molecule.
    :class:`GeATNet` follows 3 steps:
    1. uses multiple :class:`GeATLayer` instances to compute new embeddings for atoms based on their neighbors. To note, before each inner layer, the embeddings are residual added to the embeddings from the previous layer and then layer normalized.
    2. applies an extra global attention mechanism to aggregate the information from all atoms.
    3. applies a feedforward network to predict the molecular property.
    """
    
    def __init__(self, atom_embedding_dim: int, num_atom_types: int, num_bond_types: int, num_heads: int = 8, global_num_heads = 8, mlp_hidden_dim: int = 64, output_negative_slope: float = 0.2, backbone_dropout: int = 0.2, neck_dropout = 0.2, head_dropout = 0.2, geat_num_layers: int = 3, mlp_num_layers: int = 2, parallel_between_bondtypes: bool = True):
        super(GeATNet, self).__init__()
        self.backbone = GeATBackbone(atom_embedding_dim=atom_embedding_dim, num_atom_types=num_atom_types, num_bond_types=num_bond_types, num_heads=num_heads, output_negative_slope=output_negative_slope, dropout=backbone_dropout, geat_num_layers=geat_num_layers, parallel_between_bondtypes=parallel_between_bondtypes)
        self.neck = GeATNeck(atom_embedding_dim=atom_embedding_dim, num_bond_types=num_bond_types, num_heads=num_heads, global_num_heads=global_num_heads, dropout=neck_dropout)
        self.head = GeATHead(atom_embedding_dim=atom_embedding_dim, num_heads=num_heads, global_num_heads=global_num_heads, mlp_hidden_dim=mlp_hidden_dim, output_negative_slope=output_negative_slope, dropout=head_dropout, mlp_num_layers=mlp_num_layers)
        
    def forward(self, atoms, edges):
        """
        Forward pass of the GeATNet.
        :param atoms: Atom type indices of shape (batch_size, num_atoms)
        :param edges: Edge indices of shape (batch_size, num_atoms, num_atoms)
        :return: Predicted property of shape (batch_size, 1)
        """
        atom_embeddings = self.backbone(atoms, edges)
        global_attention_output = self.neck(atom_embeddings)
        output = self.head(global_attention_output)
        return output