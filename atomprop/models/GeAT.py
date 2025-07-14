import torch
import torch.nn as nn
from atomprop.embeddings.AtomEmbedding import AtomEmbedding   

class EdgeAttention(nn.Module):
    """
    An :class:`EdgeAttention` is a module for computing attention scores between pairs of atom embeddings in :class:`GeAT`.
    This module uses a bilinear attentiom mechism to compute attention scores.
    To note, there's a hidden batch dimension in front of all inputs.
    """

    def __init__(self, atom_embedding_dim: int, num_bond_types: int, output_negative_slope: float = 0.2):
        super(EdgeAttention, self).__init__()
        self.atom_d = atom_embedding_dim
        self.num_bond_types = num_bond_types
        self.a = nn.Parameter(torch.Tensor(atom_embedding_dim, atom_embedding_dim, num_bond_types)) # (d, d, T)
        self.output_negative_slope = output_negative_slope
        nn.init.xavier_uniform_(self.a, gain=1.414)

    def forward(self, src_embeddings, dst_embeddings, edges):
        """
        Compute attention scores for edges between source and destination atom embeddings.
        :param src_embeddings: Source atom embeddings of shape (batch_size, num_atoms, atom_embedding_dim)
        :param dst_embeddings: Destination atom embeddings of shape (batch_size, num_atoms, atom_embedding_dim)
        :param edges: Optional edges tensor of shape (batch_size, num_atoms, num_atoms), each number representing the bond type index
        :return: Attention scores of shape (batch_size, num_atoms, num_atoms)
        """
        B, N, d = src_embeddings.shape
        T = self.num_bond_types
        # attention = qT a k 
        transformed_src = (src_embeddings @ self.a.view(d, d*T)).view(B, N, d, T).permute(0, 3, 1, 2)
        attention_scores = (transformed_src @ dst_embeddings.transpose(1, 2)).permute(0, 2, 3, 1) # (B, N, N, T)
        # Edge mask
        attention_scores = attention_scores.gather(-1, edges.clamp(min=0).unsqueeze(-1)).squeeze(-1) # (B, N, N)
        attention_scores = attention_scores.masked_fill(edges == -1, float('-inf')) # (B, N, N)
        # Leaky ReLU activation
        attention_scores = torch.nn.functional.leaky_relu(attention_scores, negative_slope=self.output_negative_slope)
        return attention_scores 

class MultiHeadEdgeAttention(nn.Module):
    pass

class GeATLayer(nn.Module):
    """
    A :class:`GeAT` (Graph Edge Attention Transformer) for molecular property prediction.
    This model weighs the importance of only neighboring atoms.
    To note, there's a hidden batch dimension in front of all inputs.
    """

    def __init__(self, atom_embedding_dim: int, num_bond_types: int, num_heads: int = 8, output_negative_slope: float = 0.2, dropout: float = 0.1):
        super(GeATLayer, self).__init__()
        self.Q_w = nn.Linear(atom_embedding_dim, atom_embedding_dim*num_heads)
        self.K_w = nn.Linear(atom_embedding_dim, atom_embedding_dim*num_heads)
        self.V_w = nn.Linear(atom_embedding_dim, atom_embedding_dim*num_heads)
        # for each bond type, we use a different attention mechanism
        self.edge_attentions = MultiHeadEdgeAttention(atom_embedding_dim, num_bond_types, num_heads, output_negative_slope)
        self.project = nn.Linear(atom_embedding_dim * num_heads, atom_embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, atom_embeddings, edges = None):
        N = atom_embeddings.size(0)
        src_embeddings = self.Q_w(atom_embeddings)
        dst_embeddings = self.K_w(atom_embeddings)
        value_embeddings = self.V_w(atom_embeddings)
        # Calculate attention scores
        attention_scores = self.edge_attentions(src_embeddings, dst_embeddings, edges)
        # Softmax and dropout
        attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)
        # Compute attention output, which is suppose to be (b, n, d*num_heads)
        attention_output = torch.matmul(attention_scores, value_embeddings.reshape(N, -1, self.num_heads, self.atom_embedding_dim).permute(0, 2, 1, 3)) # (b, num_heads, n, d)
        attention_output = attention_output.reshape(N, -1) # (b, n, d*num_heads)
        # Project to atom embedding dimension
        attention_output = self.project(attention_output)
        return attention_output
    
class GeATLayerWithSingleHead(nn.Module):
    """
    A :class:`GeATLayerWithSingleHead` is a simplified version of :class:`GeATLayer` that uses a single head for attention.
    This model weighs the importance of only neighboring atoms.
    To note, there's a hidden batch dimension in front of all inputs.
    TEST MODULE. PLS DELETE AFTER DEVELOPMENT IS DONE.
    """

    def __init__(self, atom_embedding_dim: int, num_bond_types: int, output_negative_slope: float = 0.2, dropout: float = 0.1):
        super(GeATLayerWithSingleHead, self).__init__()
        self.Q_w = nn.Linear(atom_embedding_dim, atom_embedding_dim)
        self.K_w = nn.Linear(atom_embedding_dim, atom_embedding_dim)
        self.V_w = nn.Linear(atom_embedding_dim, atom_embedding_dim)
        self.edge_attention = EdgeAttention(atom_embedding_dim, num_bond_types, output_negative_slope)
        self.project = nn.Linear(atom_embedding_dim, atom_embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, atom_embeddings, edges = None):
        N = atom_embeddings.size(0)
        src_embeddings = self.Q_w(atom_embeddings)
        dst_embeddings = self.K_w(atom_embeddings)
        value_embeddings = self.V_w(atom_embeddings)
        # Calculate attention scores
        attention_scores = self.edge_attention(src_embeddings, dst_embeddings, edges)
        # Softmax and dropout
        attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)
        # Compute attention output, which is suppose to be (b, n, d)
        attention_output = torch.matmul(attention_scores, value_embeddings)
        # Project to atom embedding dimension
        attention_output = self.project(attention_output)
        return attention_output

class GeATNet(nn.Module):
    """
    A :class:`GeATNet` is a module for molecular property prediction using GeAT. It outputs a single scalar value representing the predicted property of the molecule.
    :class:`GeATNet` follows 3 steps:
    1. uses multiple :class:`GeATLayer` instances to compute new embeddings for atoms based on their neighbors. To note, before each inner layer, the embeddings are residual added to the embeddings from the previous layer and then layer normalized.
    2. applies an extra global attention mechanism to aggregate the information from all atoms.
    3. applies a feedforward network to predict the molecular property. 
    To note, there's a hidden batch dimension in front of all inputs.
    """
    def __init__(self, atom_embedding_dim: int, num_atom_types: int, num_bond_types: int, num_heads: int = 8, output_negative_slope: float = 0.2, geat_num_layers: int = 3):
        super(GeATNet, self).__init__()
        self.atom_embedding = AtomEmbedding(atom_embedding_dim=atom_embedding_dim, num_atom_types=num_atom_types)
        self.geat_layers = nn.ModuleList([GeATLayerWithSingleHead(atom_embedding_dim, num_bond_types, num_heads, output_negative_slope) for _ in range(geat_num_layers)])
        self.global_attention = nn.MultiheadAttention(embed_dim=atom_embedding_dim, num_heads=num_heads)
        self.fc = nn.Linear(atom_embedding_dim, 1)

    def forward(self, atoms, edges):
        """
        Forward pass of the GeATNet.
        :param atoms: Atom type indices of shape (batch_size, num_atoms)
        :param edges: Edge indices of shape (batch_size, num_atoms, num_atoms)
        :return: Predicted property of shape (batch_size, 1)
        """
        atom_embeddings = self.atom_embedding(atoms)
        for layer in self.geat_layers:
            atom_embeddings = atom_embeddings + layer(atom_embeddings, edges)
            atom_embeddings = nn.LayerNorm(atom_embeddings.size()[1:])(atom_embeddings)
        global_attention_output, _ = self.global_attention(atom_embeddings.unsqueeze(0), atom_embeddings.unsqueeze(0), atom_embeddings.unsqueeze(0))
        global_attention_output = global_attention_output.squeeze(0)
        return self.fc(global_attention_output)
