"""  
Module for Graph Edge Attention Transformer (GeAT) for molecular property prediction.   
"""  
  
import torch  
import torch.nn as nn  
from atomprop.utils.mlp import MLP  
from atomprop.embeddings.AtomEmbedding import BondTypes, BondDirections  
from atomprop.models.EdgeAttention import EdgeAttention, MultiHeadEdgeAttention
import torch_geometric 
  
class GeATLayer(nn.Module):
    """
    Graph Edge Attention Transformer Layer using explicit Edge Attention.
    Replaces manual attention with MultiHeadEdgeAttention_ParallelBetweenBondtypes.
    """

    def __init__(
        self,
        embed_dim: int,
        num_bond_types: int,
        num_heads: int = 8,
        dropout: float = 0.2,
        output_negative_slope: float = 0.2,
    ):
        super(GeATLayer, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.num_bond_types = num_bond_types

        # Linear projections for Q, K, V (shared across heads in input)
        self.Q_w = nn.Linear(embed_dim, embed_dim * num_heads)
        self.K_w = nn.Linear(embed_dim, embed_dim * num_heads)
        self.V_w = nn.Linear(embed_dim, embed_dim * num_heads)

        # Use the powerful edge-aware multi-head attention
        self.edge_attention = MultiHeadEdgeAttention(
            atom_embedding_dim=embed_dim,
            num_bond_types=num_bond_types,
            num_heads=num_heads,
            output_negative_slope=output_negative_slope,
        )

        self.dropout_layer = nn.Dropout(dropout)
        self.project = nn.Linear(embed_dim * num_heads, embed_dim)
        self.norm_after_attn = nn.LayerNorm(embed_dim * num_heads)  # optional but stabilizing

    def forward(self, atom_embeddings, edge_index=None, edge_attr=None):
        """
        Args:
            atom_embeddings: [B_N, embed_dim]
            edge_index: [2, E]
            edge_attr: [E, 2] — (bond_type, bond_direction); only bond_type used
        Returns:
            out: [B_N, embed_dim]
        """
        B_N = atom_embeddings.size(0)

        # Project to multi-head space
        Q = self.Q_w(atom_embeddings)  # [B_N, embed_dim * num_heads]
        K = self.K_w(atom_embeddings)  # [B_N, embed_dim * num_heads]
        V = self.V_w(atom_embeddings)  # [B_N, embed_dim * num_heads]

        # Compute multi-head edge-aware attention scores: [E, num_heads]
        attn_scores = self.edge_attention(
            src_embeddings=Q,
            dst_embeddings=K,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )  # shape: (num_edges, num_heads)

        # Apply softmax over neighbors for each target node (per head)
        row, col = edge_index
        attn_probs = torch_geometric.utils.softmax(attn_scores, col, num_nodes=B_N)  # [E, num_heads]
        attn_probs = self.dropout_layer(attn_probs)

        # Gather source values
        V_src = V[row]  # [E, embed_dim * num_heads]
        V_src = V_src.view(-1, self.num_heads, self.embed_dim)  # [E, num_heads, embed_dim]

        # Weighted aggregation per head
        messages = attn_probs.unsqueeze(-1) * V_src  # [E, num_heads, embed_dim]
        out = torch.zeros(
            B_N, self.num_heads, self.embed_dim,
            device=atom_embeddings.device,
            dtype=atom_embeddings.dtype,
        )
        out = out.index_add_(0, col, messages)  # [B_N, num_heads, embed_dim]

        # Reshape and project back
        out = out.view(B_N, self.embed_dim * self.num_heads)  # [B_N, embed_dim * num_heads]
        out = self.project(out)  # [B_N, embed_dim]

        return out
  
class GeATBackbone(nn.Module):  
    """  
    A :class:`GeATBackbone` is a module for molecular representation learning using GeAT. 
    It outputs atom embeddings.  
    """  
    def __init__(self, embed_dim: int, num_bond_types: int, num_heads: int = 8, output_negative_slope: float = 0.2, dropout: int = 0.2, geat_num_layers: int = 3):  
        super(GeATBackbone, self).__init__()  
        self.geat_layers = nn.ModuleList([GeATLayer(embed_dim=embed_dim, num_bond_types=num_bond_types, num_heads=num_heads, output_negative_slope=output_negative_slope, dropout=dropout) for _ in range(geat_num_layers)])  
        self.norm_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(geat_num_layers)])  
  
    def forward(self, atom_embeddings, edge_index=None, edge_attr=None):
        atom_embeddings_c = atom_embeddings.clone()
        for i, layer in enumerate(self.geat_layers):
            residual = atom_embeddings_c
            atom_embeddings_c = layer(atom_embeddings_c, edge_index, edge_attr)
            atom_embeddings_c = self.norm_layers[i](residual + atom_embeddings_c)  # ← Add this!
        return atom_embeddings_c
  
class GeATNeck(nn.Module):
    """
    A :class:`GeATNeck` module that applies global multi-head self-attention within each graph.
    It processes atom embeddings in batched format (with padding) and respects graph boundaries
    via key_padding_mask. This implementation avoids manual Q/K/V projection to ensure numerical stability.
    """

    def __init__(self, embed_dim: int, global_num_heads: int = 8, dropout: float = 0.2):
        super(GeATNeck, self).__init__()
        # Use MultiheadAttention with the original embed_dim.
        # PyTorch will internally split it into `global_num_heads` heads.
        self.global_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,          # ← Critical: use original embed_dim
            num_heads=global_num_heads,
            dropout=dropout,
            batch_first=False
        )
        # LayerNorm applied after residual connection
        self.norm_layer = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def forward(self, atom_embeddings: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with batched processing of multiple graphs.

        Args:
            atom_embeddings: [B_N, embed_dim] - Atom-level embeddings from backbone.
            batch: [B_N] - Batch assignment vector (from PyG).

        Returns:
            torch.Tensor: Updated atom embeddings of shape [B_N, embed_dim].
        """
        assert batch is not None, "batch tensor must be provided for graph-level attention."

        B_N = atom_embeddings.size(0)
        device = atom_embeddings.device
        dtype = atom_embeddings.dtype

        # Determine number of graphs and their sizes
        batch_size = batch.max().item() + 1
        graph_indices_list = []
        graph_sizes = []

        for i in range(batch_size):
            mask = (batch == i)
            indices = mask.nonzero(as_tuple=True)[0]
            graph_indices_list.append(indices)
            graph_sizes.append(len(indices))

        max_graph_size = max(graph_sizes)

        # Pad atom embeddings to [batch_size, max_graph_size, embed_dim]
        X_padded = torch.zeros(
            batch_size, max_graph_size, self.embed_dim,
            device=device, dtype=dtype
        )

        for i in range(batch_size):
            indices = graph_indices_list[i]
            if len(indices) > 0:
                X_padded[i, :len(indices)] = atom_embeddings[indices]

        # Create key_padding_mask: True means ignore, False means attend
        key_padding_mask = torch.ones(batch_size, max_graph_size, dtype=torch.bool, device=device)
        for i in range(batch_size):
            key_padding_mask[i, :graph_sizes[i]] = False

        # Transpose to [seq_len, batch_size, embed_dim] for MultiheadAttention
        X_t = X_padded.transpose(0, 1)  # [max_graph_size, batch_size, embed_dim]

        # Apply global self-attention with padding mask
        attn_output, _ = self.global_attention(
            X_t, X_t, X_t,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )

        # Transpose back to [batch_size, max_graph_size, embed_dim]
        attn_output = attn_output.transpose(0, 1)

        # Residual connection + LayerNorm
        output_padded = self.norm_layer(attn_output + X_padded)

        # Scatter back to original node order [B_N, embed_dim]
        final_output = torch.zeros(B_N, self.embed_dim, device=device, dtype=dtype)
        for i in range(batch_size):
            indices = graph_indices_list[i]
            if len(indices) > 0:
                final_output[indices] = output_padded[i, :len(indices)]

        return final_output
        
class GeATNet(nn.Module):  
    """  
    A :class:`GeATNet` is a module for molecular embeddings generation using GeAT.
    :class:`GeATNet` follows 3 steps:  
    1. uses multiple :class:`GeATLayer` instances to compute new embeddings for atoms based on their neighbors. To note, before each inner layer, the embeddings are residual added to the embeddings from the previous layer and then layer normalized.  
    2. applies an extra global attention mechanism to aggregate the information from all atoms.  
    3. applies a feedforward network to predict the molecular property.  
    """  
      
    def __init__(self, embed_dim: int, num_bond_types = None, num_heads: int = 8, global_num_heads = 8, output_negative_slope: float = 0.2, dropout: int = 0.2, geat_num_layers: int = 5):  
        super(GeATNet, self).__init__()
        if num_bond_types is None:
            num_bond_types = len(BondTypes.get_bond_types())+1  
        self.backbone = GeATBackbone(embed_dim=embed_dim, num_bond_types=num_bond_types, num_heads=num_heads, output_negative_slope=output_negative_slope, dropout=dropout, geat_num_layers=geat_num_layers)  
        self.neck = GeATNeck(embed_dim=embed_dim, global_num_heads=global_num_heads, dropout=dropout)  
                  
    def forward(self, data, batch=None):  
        """  
        Forward pass of the GeATNet.  
        :param data: PyG data object for graphs 
        :param batch: Batch indices for sparse format  
        :return: Graph emb of shape (B_N, embed_dim)  
        """  
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        atom_embeddings = self.backbone(x, edge_index, edge_attr)  
        output = self.neck(atom_embeddings, batch)
        return output