import torch
import torch.nn as nn
from atomprop.embeddings.AtomEmbedding import AtomEmbedding   

class EdgeAttention(nn.Module):
    """
    An :class:`EdgeAttention` is a module for computing attention scores between pairs of atom embeddings in :class:`GeAT`.
    This module uses a feedforward neural network to compute attention scores.
    """

    def __init__(self, atom_embedding_dim: int, num_bond_types: int, output_negative_slope: float = 0.2):
        super(EdgeAttention, self).__init__()
        self.atom_d = atom_embedding_dim
        self.num_bond_types = num_bond_types
        self.a = nn.Parameter(torch.Tensor(2*atom_embedding_dim*num_bond_types, 1))
        self.output_negative_slope = output_negative_slope
        nn.init.xavier_uniform_(self.a, gain=1.414)

    def forward(self, src_embeddings, dst_embeddings, edges=None):
        """
        Compute attention scores for edges between source and destination atom embeddings.
        :param src_embeddings: Source atom embeddings of shape (num_src_atoms, atom_embedding_dim)
        :param dst_embeddings: Destination atom embeddings of shape (num_dst_atoms, atom_embedding_dim)
        :param edges: Optional edges tensor of shape (num_dst_atoms, num_dst_atoms), each number representing the bond type index 
        """
        # d for atom embedding dimension
        # Concat src and dst embeddings into n*n*2d
        N = src_embeddings.size(1)
        input_concat = torch.cat([src_embeddings.repeat(1, 1, N).reshape(-1, N, N, 2*self.atom_d), dst_embeddings.repeat(1, N, 1).reshape(-1, N, N, 2*self.atom_d)], dim=-1)
        # Sparse one-hot encoding
        *lead, n, l = input_concat.shape
        lt = self.num_bond_types * l
        sparsed_input = torch.zeros(*lead, n, n, lt)
        mask = edges >= 0  
        k = edges.clamp(min=0)
        start = k * l 
        offset = torch.arange(l) 
        idx = start.unsqueeze(-1) + offset
        idx = idx.clamp(max=lt-1)
        mask_l = mask.unsqueeze(-1)
        idx = idx.masked_fill(~mask_l, 0)
        sparsed_input.scatter_(-1, idx.long(), input_concat * mask_l.float()) # c shape is (b, n, n, t*l)
        # Compute attention scores
        attention_scores = torch.matmul(sparsed_input, self.a).squeeze(-1) # shape is (b, n, n, 1)
        # Leaky ReLU activation
        attention_scores = torch.nn.functional.leaky_relu(attention_scores, negative_slope=self.output_negative_slope)
        return attention_scores

class MultiHeadEdgeAttention(nn.Module):
    """
    A module for computing multi-head edge attention scores.
    """

    def __init__(self, atom_embedding_dim: int, num_bond_types: int, num_heads: int = 8, output_negative_slope: float = 0.2):
        super(MultiHeadEdgeAttention, self).__init__()
        self.atom_d = atom_embedding_dim
        self.num_bond_types = num_bond_types
        self.num_heads = num_heads
        self.a = nn.Parameter(torch.Tensor(2*atom_embedding_dim*num_bond_types*num_heads, num_heads))
        self.output_negative_slope = output_negative_slope
        nn.init.xavier_uniform_(self.a, gain=1.414)

    def forward(self, src_embeddings, dst_embeddings, edges=None):
        """
        Compute attention scores for edges between source and destination atom embeddings.
        :param src_embeddings: Source atom embeddings of shape (num_src_atoms, atom_embedding_dim*num_heads)
        :param dst_embeddings: Destination atom embeddings of shape (num_dst_atoms, atom_embedding_dim*num_heads)
        :param edges: Optional edges tensor of shape (num_dst_atoms, num_dst_atoms), each number representing the bond type index 
        :return: Attention scores of shape (num_dst_atoms, num_dst_atoms, num_heads)
        """
        # d for atom embedding dimension, e for edge embedding dimension
        # Concat src and dst embeddings into n*n*2d
        N = src_embeddings.size(1)
        input_concat = torch.cat([src_embeddings.repeat(1, 1, N).reshape(-1, N, N, 2*self.atom_d), dst_embeddings.repeat(1, N, 1).reshape(-1, N, N, 2*self.atom_d)], dim=-1)
        # Sparse one-hot encoding
        *lead, n, l = input_concat.shape
        lt = self.num_bond_types * l * self.num_heads
        sparsed_input = torch.zeros(*lead, n, n, lt)
        mask = edges >= 0  
        k = edges.clamp(min=0)
        start = k * l * self.num_heads
        offset = torch.arange(l * self.num_heads) 
        idx = start.unsqueeze(-1) + offset
        idx = idx.clamp(max=lt-1)
        mask_l = mask.unsqueeze(-1)
        idx = idx.masked_fill(~mask_l, 0)
        sparsed_input.scatter_(-1, idx.long(), input_concat * mask_l.float())
        # Compute attention scores
        attention_scores = torch.matmul(sparsed_input, self.a).squeeze(-1)
        # Leaky ReLU activation
        attention_scores = torch.nn.functional.leaky_relu(attention_scores, negative_slope=self.output_negative_slope)  
        return attention_scores

class GeATLayer(nn.Module):
    """
    A :class:`GeAT` (Graph Edge Attention Transformer) for molecular property prediction.
    This model weighs the importance of only neighboring atoms.
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

class GeATNet(nn.Module):
    """
    A :class:`GeATNet` is a module for molecular property prediction using GeAT. It outputs a single scalar value representing the predicted property of the molecule.
    :class:`GeATNet` follows 3 steps:
    1. uses multiple :class:`GeATLayer` instances to compute new embeddings for atoms based on their neighbors. To note, before each inner layer, the embeddings are residual added to the embeddings from the previous layer and then layer normalized.
    2. applies an extra global attention mechanism to aggregate the information from all atoms.
    3. applies a feedforward network to predict the molecular property. 
    """
    def __init__(self, atom_embedding_dim: int, num_atom_types: int, num_bond_types: int, num_heads: int = 8, output_negative_slope: float = 0.2, geat_num_layers: int = 3):
        super(GeATNet, self).__init__()
        self.atom_embedding = AtomEmbedding(atom_embedding_dim=atom_embedding_dim, num_atom_types=num_atom_types)
        self.geat_layers = nn.ModuleList([GeATLayer(atom_embedding_dim, num_bond_types, num_heads, output_negative_slope) for _ in range(geat_num_layers)])
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
