import torch
import torch.nn as nn
from atomprop.embeddings.AtomEmbedding import AtomEmbedding, bond_types        

class EdgeAttention(nn.Module):
    """
    An :class:`EdgeAttention` is a module for computing attention scores between pairs of atom embeddings in :class:`GeAT`.
    This module uses a feedforward neural network to compute attention scores.
    """

    def __init__(self, atom_embedding_dim: int, hidden_dim: int = 64, attention_num_layers: int = 2, output_negative_slope: float = 0.2):
        super(EdgeAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(atom_embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(attention_num_layers - 1)],
            nn.Linear(hidden_dim, 1),
            nn.LeakyReLU(negative_slope=output_negative_slope)
        )

    def forward(self, src_embeddings, dst_embeddings):
        """
        Compute attention scores for edges between source and destination atom embeddings.
        """
        edge_features = torch.cat([src_embeddings, dst_embeddings], dim=-1)
        attention_scores = self.attention(edge_features)
        return attention_scores

class MultiHeadEdgeAttention(nn.Module):
    """
    A module for computing multi-head edge attention scores.
    This module applies multiple edge attention mechanisms in parallel.
    """

    def __init__(self, atom_embedding_dim: int, num_heads: int = 8, hidden_dim: int = 64, attention_num_layers: int = 2, output_negative_slope: float = 0.2):
        super(MultiHeadEdgeAttention, self).__init__()
        self.heads = nn.ModuleList([EdgeAttention(atom_embedding_dim, hidden_dim, attention_num_layers, output_negative_slope) for _ in range(num_heads)])

    def forward(self, src_embeddings, dst_embeddings):
        return torch.stack([head(src_embeddings, dst_embeddings) for head in self.heads], dim=1)

class GeATLayer(nn.Module):
    """
    A :class:`GeAT` (Graph Edge Attention Transformer) for molecular property prediction.
    This model weighs the importance of only neighboring atoms.
    """

    def __init__(self, atom_embedding_dim: int, num_bond_types: int, num_heads: int = 8, hidden_dim: int = 64, attention_num_layers: int = 2, output_negative_slope: float = 0.2):
        super(GeATLayer, self).__init__()
        self.Q_w = nn.Linear(atom_embedding_dim, atom_embedding_dim)
        self.K_w = nn.Linear(atom_embedding_dim, atom_embedding_dim)
        self.V_w = nn.Linear(atom_embedding_dim, atom_embedding_dim)
        # for each bond type, we use a different attention mechanism
        self.edge_attentions = nn.ModuleList([MultiHeadEdgeAttention(atom_embedding_dim, num_heads, hidden_dim, attention_num_layers, output_negative_slope) for _ in range(num_bond_types)])
        self.project = nn.Linear(atom_embedding_dim * num_heads, atom_embedding_dim)

    def forward(self, embeddings, edges: list):
        node_num = embeddings.size(0)
        src_embeddings = self.Q_w(embeddings)
        dst_embeddings = self.K_w(embeddings)
        value_embeddings = self.V_w(embeddings)
        edge_attention_scores = torch.full((node_num, node_num), float('-inf'))
        for (src, dst), bond_types in edges:
            # attention of src to dst
            src_embedding = src_embeddings[src]
            dst_embedding = dst_embeddings[dst]
            edge_attention_scores[src, dst] = torch.stack([self.edge_attentions[bond_type](src_embedding, dst_embedding) for bond_type in bond_types])
            # attention of dst to src
            edge_attention_scores[dst, src] = torch.stack([self.edge_attentions[bond_type](dst_embedding, src_embedding) for bond_type in bond_types])
        # Mask padding parts
        src_mask = (src == -1)
        dst_mask = (dst == -1)
        edge_attention_scores[:, src_mask] = -torch.inf
        edge_attention_scores[dst_mask, :] = -torch.inf

        edge_attention_scores = torch.softmax(edge_attention_scores, dim=-1)
        weighed_value_embeddings = edge_attention_scores.unsqueeze(-1) * value_embeddings.unsqueeze(0)
        projected_value_embeddings = self.project(weighed_value_embeddings)
        return projected_value_embeddings

class GeATNet(nn.Module):
    """
    A :class:`GeATNet` is a module for molecular property prediction using GeAT. It outputs a single scalar value representing the predicted property of the molecule.
    :class:`GeATNet` follows 3 steps:
    1. uses multiple :class:`GeATLayer` instances to compute new embeddings for atoms based on their neighbors. To note, before each inner layer, the embeddings are residual added to the embeddings from the previous layer and then layer normalized.
    2. applies an extra global attention mechanism to aggregate the information from all atoms.
    3. applies a feedforward network to predict the molecular property. 
    """
    def __init__(self, atom_embedding_dim: int, num_atom_types: int, num_bond_types: int, num_heads: int = 8, hidden_dim: int = 64, attention_num_layers: int = 2, output_negative_slope: float = 0.2, geat_num_layers: int = 2):
        super(GeATNet, self).__init__()
        self.atom_embedding = AtomEmbedding(atom_embedding_dim=atom_embedding_dim, num_atom_types=num_atom_types)
        self.geat_layers = nn.ModuleList([GeATLayer(atom_embedding_dim, num_bond_types, num_heads, hidden_dim, attention_num_layers, output_negative_slope) for _ in range(geat_num_layers)])
        self.global_attention = nn.MultiheadAttention(embed_dim=atom_embedding_dim, num_heads=num_heads)
        self.fc = nn.Linear(atom_embedding_dim, 1)

    def forward(self, atom_type_indices, edges):
        atom_embeddings = self.atom_embedding(atom_type_indices)
        for layer in self.geat_layers:
            atom_embeddings = atom_embeddings + layer(atom_embeddings, edges)
            atom_embeddings = nn.LayerNorm(atom_embeddings.size()[1:])(atom_embeddings)
        global_attention_output, _ = self.global_attention(atom_embeddings.unsqueeze(0), atom_embeddings.unsqueeze(0), atom_embeddings.unsqueeze(0))
        global_attention_output = global_attention_output.squeeze(0)
        return self.fc(global_attention_output)
