"""
Module for Graph Edge Attention Transformer (GeAT) for molecular property prediction.
"""

import torch
import torch.nn as nn
from atomprop.embeddings.AtomEmbedding import AtomEmbedding
from atomprop.utils.mlp import MLP
import atomprop.embeddings.PositionEmbedding as PE

class EdgeAttention(nn.Module):
    """
    An :class:`EdgeAttention` is a module for computing attention scores between pairs of atom embeddings in :class:`GeAT`.
    This module uses a bilinear attention mechism to compute attention scores.
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
        transformed_src = (src_embeddings @ self.a.view(d, d*T)).view(B, N, d, T) # (B, N, d, T)
        attention_scores = (transformed_src.transpose(2, 3).reshape(B, N*T, d) @ dst_embeddings.transpose(1, 2)).transpose(1, 2).view(B, N, N, T) # (B, N, N, T)
        # Edge mask
        attention_scores = attention_scores.gather(-1, edges.clamp(min=0).unsqueeze(-1)).squeeze(-1) # (B, N, N)
        attention_scores = attention_scores.masked_fill(edges==-1, -1e10) # (B, N, N)
        # Leaky ReLU activation
        attention_scores = torch.nn.functional.leaky_relu(attention_scores, negative_slope=self.output_negative_slope)
        return attention_scores 

class MultiHeadEdgeAttention(nn.Module):
    def __new__(cls, parallel_between_bondtypes: bool = True, *args, **kwargs):
        """
        A :class:`MultiHeadEdgeAttention` is a module for computing multi-head attention scores between pairs of atom embeddings in :class:`GeATLayer`.
        This module uses a bilinear attention mechanism to compute attention scores.
        It receives a boolean argument `parallel_between_bondtypes` to determine whether to use parallel or serial attention between bond types.
        """

        if parallel_between_bondtypes:
            return MultiHeadEdgeAttention_ParallelBetweenBondtypes(*args, **kwargs)
        else:
            return MultiHeadEdgeAttention_SerialBetweenBondtypes(*args, **kwargs)

class MultiHeadEdgeAttention_ParallelBetweenBondtypes(nn.Module):
    """
    A :class:`MultiHeadEdgeAttention_ParallelBetweenBondtypes` is a subclass of :class:`MultiHeadEdgeAttention` that computes multi-head attention scores in parallel between bond types.
    This module will use extra memory in calculation.
    """

    def __init__(self, atom_embedding_dim: int, num_bond_types: int, num_heads: int = 8, output_negative_slope: float = 0.2):
        super(MultiHeadEdgeAttention_ParallelBetweenBondtypes, self).__init__()
        self.num_heads = num_heads
        self.atom_d = atom_embedding_dim
        self.num_bond_types = num_bond_types
        self.a = nn.Parameter(torch.Tensor(atom_embedding_dim * num_heads, atom_embedding_dim, num_bond_types))
        self.output_negative_slope = output_negative_slope
        nn.init.xavier_uniform_(self.a, gain=1.414)
        
    def forward(self, src_embeddings, dst_embeddings, edges):
        """
        Compute multi-head attention scores for edges between source and destination atom embeddings.
        :param src_embeddings: Source atom embeddings of shape (batch_size, num_atoms, atom_embedding_dim * num_heads)
        :param dst_embeddings: Destination atom embeddings of shape (batch_size, num_atoms, atom_embedding_dim * num_heads)
        :param edges: Optional edges tensor of shape (batch_size, num_atoms, num_atoms), each number representing the bond type index
        :return: Attention scores of shape (batch_size, num_heads, num_atoms, num_atoms)
        """
        B, N, d_ = src_embeddings.shape
        d = d_ // self.num_heads
        T = self.num_bond_types
        # attention = qT a k
        src_embeddings = src_embeddings.reshape(B, N, self.num_heads, d).permute(0, 2, 1, 3) # (B, num_heads, N, d)
        dst_embeddings = dst_embeddings.reshape(B, N, self.num_heads, d).permute(0, 2, 1, 3) # (B, num_heads, N, d)
        transformed_src = (src_embeddings @ self.a.view(self.num_heads, d, d*T)).view(B, self.num_heads, N, d, T) # (B, num_heads, N, d, T)
        attention_scores = (transformed_src.transpose(3, 4).reshape(B, self.num_heads, N*T, d) @ dst_embeddings.transpose(2, 3)).transpose(2, 3).view(B, self.num_heads, N, N, T) # (B, num_heads, N, N, T)
        # Edge mask
        # expand edges to match the number of heads
        edges = edges.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # (B, num_heads, N, N)
        attention_scores = attention_scores.gather(-1, edges.clamp(min=0).unsqueeze(-1)).squeeze(-1) # (B, num_heads, N, N)
        attention_scores = attention_scores.masked_fill(edges==-1, -1e10) # (B, num_heads, N, N)
        # Leaky ReLU activation
        attention_scores = torch.nn.functional.leaky_relu(attention_scores, negative_slope=self.output_negative_slope)
        return attention_scores  # (B, num_heads, N, N)

class MultiHeadEdgeAttention_SerialBetweenBondtypes(nn.Module):
    """
    A :class:`MultiHeadEdgeAttention_SerialBetweenBondtypes` is a subclass of :class:`nn.Module` that computes multi-head attention scores sequentially between bond types.
    This module will use less memory compared to the parallel version but may have slightly increased computation time.
    """

    def __init__(self, atom_embedding_dim: int, num_bond_types: int, num_heads: int = 8, output_negative_slope: float = 0.2):
        super(MultiHeadEdgeAttention_SerialBetweenBondtypes, self).__init__()
        self.num_heads = num_heads
        self.atom_d = atom_embedding_dim
        self.num_bond_types = num_bond_types
        self.a = nn.Parameter(torch.Tensor(atom_embedding_dim * num_heads, atom_embedding_dim, num_bond_types))
        self.output_negative_slope = output_negative_slope
        nn.init.xavier_uniform_(self.a, gain=1.414)

    def forward(self, src_embeddings, dst_embeddings, edges):
        """
        Compute multi-head attention scores for edges between source and destination atom embeddings.
        :param src_embeddings: Source atom embeddings of shape (batch_size, num_atoms, atom_embedding_dim * num_heads)
        :param dst_embeddings: Destination atom embeddings of shape (batch_size, num_atoms, atom_embedding_dim * num_heads)
        :param edges: Optional edges tensor of shape (batch_size, num_atoms, num_atoms), each number representing the bond type index
        :return: Attention scores of shape (batch_size, num_heads, num_atoms, num_atoms)
        """
        B, N, d_ = src_embeddings.shape
        d = d_ // self.num_heads
        T = self.num_bond_types
        # attention = qT a k
        src_embeddings = src_embeddings.reshape(B, N, self.num_heads, d).permute(0, 2, 1, 3) # (B, num_heads, N, d)
        dst_embeddings = dst_embeddings.reshape(B, N, self.num_heads, d).permute(0, 2, 1, 3) # (B, num_heads, N, d)
        attention_scores = []
        for t in range(T):
            transformed_src = (src_embeddings @ self.a[:, :, t].view(self.num_heads, d, 1)).view(B, self.num_heads, N, d)
            attention_scores_t = (transformed_src.transpose(2, 3).reshape(B, self.num_heads, N, d) @ dst_embeddings.transpose(2, 3)).transpose(2, 3).view(B, self.num_heads, N, N) # (B, num_heads, N, N)
            # Edge mask
            edges_t = edges == t
            attention_scores_t = attention_scores_t.masked_fill(edges_t.unsqueeze(1), -1e10) # (B, num_heads, N, N)
            # Leaky ReLU activation
            attention_scores_t = torch.nn.functional.leaky_relu(attention_scores_t, negative_slope=self.output_negative_slope)
            attention_scores.append(attention_scores_t.unsqueeze(-1))  # (B, num_heads, N, N, 1)
        attention_scores = torch.cat(attention_scores, dim=-1)  # (B, num_heads, N, N, T)
        attention_scores = attention_scores.sum(dim=-1)  # Sum over bond types to get final attention scores
        return attention_scores  # (B, num_heads, N, N)
