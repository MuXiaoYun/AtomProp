import torch
import torch.nn as nn
from atomprop.embeddings.AtomEmbedding import AtomEmbedding, bond_types

class GAT(nn.Module):
    """
    A :class:`GAT` (Graph Attention Network) for molecular property prediction.
    This model uses attention mechanisms to weigh the importance of neighboring atoms.
    """

    def __init__(self, atom_embedding_dim: int, num_atom_types: int, num_heads: int = 8):
        super(GAT, self).__init__()
        self.atom_embedding = nn.Embedding(num_embeddings=num_atom_types, embedding_dim=atom_embedding_dim)
        self.attention = nn.MultiheadAttention(embed_dim=atom_embedding_dim, num_heads=num_heads)

    def forward(self, atom_type_indices):
        atom_embeddings = self.atom_embedding(atom_type_indices)
        attention_output, _ = self.attention(atom_embeddings.unsqueeze(0), atom_embeddings.unsqueeze(0), atom_embeddings.unsqueeze(0))
        return attention_output.squeeze(0)  # Remove the batch dimension
    
class EdgeAttention(nn.Module):
    """
    An :class:`EdgeAttention` is a module for computing attention scores between pairs of atom embeddings in :class:`GeAT`.
    This module uses a feedforward neural network to compute attention scores.
    """

    def __init__(self, atom_embedding_dim: int, hidden_dim: int = 64, num_layers: int = 2):
        super(EdgeAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(atom_embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(num_layers - 1)],
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, src_embeddings, dst_embeddings):
        """
        Compute attention scores for edges between source and destination atom embeddings.
        """
        edge_features = torch.cat([src_embeddings, dst_embeddings], dim=-1)
        attention_scores = self.attention(edge_features)
        return torch.sigmoid(attention_scores)

class MultiHeadEdgeAttention(nn.Module):
    """
    A module for computing multi-head edge attention scores.
    This module applies multiple edge attention mechanisms in parallel.
    """

    def __init__(self, atom_embedding_dim: int, num_heads: int = 8, hidden_dim: int = 64):
        super(MultiHeadEdgeAttention, self).__init__()
        self.heads = nn.ModuleList([EdgeAttention(atom_embedding_dim, hidden_dim) for _ in range(num_heads)])

    def forward(self, src_embeddings, dst_embeddings):
        return torch.stack([head(src_embeddings, dst_embeddings) for head in self.heads], dim=1)

class GeAT(nn.Module):
    """
    A :class:`GeAT` (Graph Edge Attention Transformer) for molecular property prediction.
    This model uses masked edge attention mechanisms to weigh the importance of only neighboring atoms.
    """

    def __init__(self, atom_embedding_dim: int, num_atom_types: int, num_bond_types: int, num_heads: int = 8):
        super(GeAT, self).__init__()
        self.atom_embedding = AtomEmbedding(atom_embedding_dim=atom_embedding_dim, num_atom_types=num_atom_types)
        self.Q_w = nn.Linear(atom_embedding_dim, atom_embedding_dim)
        self.K_w = nn.Linear(atom_embedding_dim, atom_embedding_dim)
        self.edge_attention = MultiHeadEdgeAttention(atom_embedding_dim=atom_embedding_dim, num_heads=num_heads)

    def forward(self, atom_type_indices, src_indices, dst_indices):
        atom_embeddings = self.atom_embedding(atom_type_indices)
        src_embeddings = self.Q_w(atom_embeddings[src_indices])
        dst_embeddings = self.K_w(atom_embeddings[dst_indices])
        edge_attention_scores = self.edge_attention(src_embeddings, dst_embeddings)
        return edge_attention_scores
    
class GeATNet(nn.Module):
    """
    A :class:`GeATNet` is a module for molecular property prediction using GeAT.
    :class:`GeATNet` follows 3 steps:
    1. combines atom embeddings and MASKED edge attention mechanisms to calculate new value embeddings for atoms. The masked attention mechanism ensures that only neighboring atoms are considered.
    2. applies an extra global attention mechanism to aggregate the information from all atoms and uses V_w to transform the value embeddings.
    3. applies a feedforward network to predict the molecular property. 
    """
    def __init__(self, atom_embedding_dim: int, num_atom_types: int, num_bond_types: int, num_heads: int = 8, hidden_dim: int = 64, edges: list = []):
        super(GeATNet, self).__init__()
        self.geat = GeAT(atom_embedding_dim=atom_embedding_dim, num_atom_types=num_atom_types, num_bond_types=num_bond_types, num_heads=num_heads)
        self.V_w = nn.Linear(atom_embedding_dim * num_heads, atom_embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(atom_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, atom_type_indices, src_indices, dst_indices):
        edge_attention_scores = self.geat(atom_type_indices, src_indices, dst_indices)
        value_embeddings = self.V_w(edge_attention_scores.view(edge_attention_scores.size(0), -1))
        return self.ffn(value_embeddings)
