"""  
Module for Graph Edge Attention Transformer (GeAT) for molecular property prediction.  
"""  
  
import torch  
import torch.nn as nn  
import torch.nn.functional as F
from atomprop.utils.mlp import MLP

default_attention_type = 'bilinear'
 
class EdgeAttention(nn.Module):  
    """  
    An :class:`EdgeAttention` is a module for computing attention scores between pairs of atom embeddings in :class:`GeAT`.  
    This module supports two attention mechanisms: 'bilinear' and 'mlp'.  
    - 'bilinear': Uses bilinear transformation to compute attention scores (default)  
    - 'mlp': Uses a small MLP on concatenated inputs to compute attention scores  
    """  
    def __init__(self,
                 atom_embedding_dim: int,
                 num_bond_types: int, 
                 output_negative_slope: float = 0.2,
                 attention_type: str = default_attention_type,
                 mlp_hidden_dim: int = 1024,
                 mlp_num_layers: int = 2,
                 use_edge_embedding: bool = False):  
        super(EdgeAttention, self).__init__()  
        self.atom_d = atom_embedding_dim  
        self.num_bond_types = num_bond_types  
        self.attention_type = attention_type  
        self.output_negative_slope = output_negative_slope  
        self.use_edge_embedding = use_edge_embedding
        
        if attention_type not in ['bilinear', 'mlp']:
            raise ValueError(f"attention_type must be 'bilinear' or 'mlp', got {attention_type}")
        
        if attention_type == 'bilinear':
            # Bilinear attention: q^T A_t k, where A_t is a learnable matrix for each bond type t
            self.a = nn.Parameter(torch.Tensor(atom_embedding_dim, atom_embedding_dim, num_bond_types))  # (d, d, T)  
            nn.init.xavier_uniform_(self.a, gain=1.414)  
        else:
            # MLP attention: MLP([src, dst]) for each bond type
            self.mlp_hidden_dim = mlp_hidden_dim
            # Create separate MLP for each bond type
            self.mlps = nn.ModuleList([
                MLP(input_dim=atom_embedding_dim * 2, 
                    hidden_dim=mlp_hidden_dim, 
                    output_dim=1, 
                    num_layers=mlp_num_layers,
                    output_activation=F.relu)
                for _ in range(num_bond_types)
            ])
  
    def forward(self, src_embeddings, dst_embeddings, edge_embeddings, edge_index=None, edge_attr=None):  
        """  
        Compute attention scores for edges between source and destination atom embeddings.  
        :param src_embeddings: Source atom embeddings of shape (batch_size, num_atoms, atom_embedding_dim) or (total_atoms, atom_embedding_dim)  
        :param dst_embeddings: Destination atom embeddings of shape (batch_size, num_atoms, atom_embedding_dim) or (total_atoms, atom_embedding_dim)  
        :param edge_index: Edge indices of shape (2, num_edges) for sparse format  
        :param edge_attr: Edge attributes of shape (num_edges, 2) for sparse format (bond_type, bond_direction)  
        :param edges: Optional edges tensor of shape (batch_size, num_atoms, num_atoms) for dense format (deprecated)  
        :return: Attention scores of shape (batch_size, num_atoms, num_atoms) or (num_edges,)  
        """  
        # Get source and target node features for each edge  
        row, col = edge_index  # row=source, col=target  
        src_features = src_embeddings[row]  # [E, d]
        if self.use_edge_embedding:  
            dst_features = dst_embeddings[col]+edge_embeddings  # [E, d]  
        else:
            dst_features = dst_embeddings[col]
          
        # Get bond types from edge_attr  
        bond_types = edge_attr[:, 0]  # [E]  
          
        if self.attention_type == 'bilinear':
            # Compute attention scores using bond-type-specific bilinear transforms  
            attention_scores = []  
            for t in range(self.num_bond_types):  
                # Get mask for this bond type  
                mask = (bond_types == t)  
                if mask.sum() > 0:  
                    # Compute q^T A_t k for edges of this bond type  
                    src_t = src_features[mask]  # [E_t, d]  
                    dst_t = dst_features[mask]  # [E_t, d]  
                    A_t = self.a[:, :, t]  # [d, d]  
                      
                    # Bilinear attention: q^T A k  
                    scores_t = (src_t @ A_t * dst_t).sum(dim=-1)  # [E_t]  
                    attention_scores.append(scores_t)  
          
            # Concatenate all scores  
            if attention_scores:  
                attention_scores = torch.cat(attention_scores, dim=0)  # [E]  
            else:  
                attention_scores = torch.zeros(edge_index.size(1), device=src_embeddings.device)  # [E]  
        else:
            # Compute attention scores using bond-type-specific MLPs
            attention_scores = []  
            for t in range(self.num_bond_types):  
                # Get mask for this bond type  
                mask = (bond_types == t)  
                if mask.sum() > 0:  
                    # Get features for this bond type  
                    src_t = src_features[mask]  # [E_t, d]  
                    dst_t = dst_features[mask]  # [E_t, d]  
                      
                    # Concatenate src and dst features  
                    concat_features = torch.cat([src_t, dst_t], dim=-1)  # [E_t, 2*d]  
                      
                    # MLP attention: MLP([src, dst])  
                    scores_t = self.mlps[t](concat_features).squeeze(-1)  # [E_t]  
                    attention_scores.append(scores_t)  
          
            # Concatenate all scores  
            if attention_scores:  
                attention_scores = torch.cat(attention_scores, dim=0)  # [E]  
            else:  
                attention_scores = torch.zeros(edge_index.size(1), device=src_embeddings.device)  # [E]  
          
        # Apply LeakyReLU  
        attention_scores = torch.nn.functional.leaky_relu(attention_scores, negative_slope=self.output_negative_slope)  
          
        return attention_scores  
 
 
class MultiHeadEdgeAttention(nn.Module):  
    def __new__(cls, parallel_between_bondtypes: bool = True, *args, **kwargs):  
        """  
        A :class:`MultiHeadEdgeAttention` is a module for computing multi-head attention scores between pairs of atom embeddings in :class:`GeATLayer`.  
        This module uses a bilinear or MLP attention mechanism to compute attention scores based on attention_type.  
        It receives a boolean argument `parallel_between_bondtypes` to determine whether to use parallel or serial attention between bond types.  
        """  
  
        if parallel_between_bondtypes:  
            return MultiHeadEdgeAttention_ParallelBetweenBondtypes(*args, **kwargs)  
        else:  
            return MultiHeadEdgeAttention_SerialBetweenBondtypes(*args, **kwargs)  
 
 
class MultiHeadEdgeAttention_ParallelBetweenBondtypes(nn.Module):
    """
    Multi-head edge attention with bond-type-specific bilinear or MLP interaction.
    
    For each bond type t, learns a separate bilinear matrix A_t ∈ R^(H × D × D) or MLP,
    and computes attention score for edge e of type t as:
        - Bilinear: score_e = src_e^T @ A_t @ dst_e   (per head)
        - MLP: score_e = MLP_t([src_e, dst_e])   (per head)
 
    Args:
        atom_embedding_dim (int): Dimension of atom embeddings (D)
        num_bond_types (int): Number of distinct bond types (e.g., SINGLE=0, DOUBLE=1, ...)
        num_heads (int): Number of attention heads (H)
        output_negative_slope (float): LeakyReLU negative slope (optional, for compatibility)
        attention_type (str): Type of attention mechanism ('bilinear' or 'mlp')
        mlp_hidden_dim (int): Hidden dimension for MLP (if attention_type='mlp')
        mlp_num_layers (int): Number of layers for MLP (if attention_type='mlp')
    """
 
    def __init__(
        self,
        atom_embedding_dim: int,
        num_bond_types: int,
        num_heads: int = 8,
        output_negative_slope: float = 0.2,
        attention_type: str = default_attention_type,
        mlp_hidden_dim: int = 1024,
        mlp_num_layers: int = 2,
        use_edge_embedding: bool = False
    ):
        super().__init__()
        self.atom_embedding_dim = atom_embedding_dim
        self.num_bond_types = num_bond_types
        self.num_heads = num_heads
        self.attention_type = attention_type
        self.use_edge_embedding = use_edge_embedding
        
        if attention_type not in ['bilinear', 'mlp']:
            raise ValueError(f"attention_type must be 'bilinear' or 'mlp', got {attention_type}")
 
        if attention_type == 'bilinear':
            # [num_bond_types, num_heads, D, D]
            self.a = nn.Parameter(
                torch.empty(num_bond_types, num_heads, atom_embedding_dim, atom_embedding_dim)
            )
            # Initialize each A_t with Xavier uniform (as in GAT)
            nn.init.xavier_uniform_(self.a.view(-1, atom_embedding_dim, atom_embedding_dim))
        else:
            # MLP attention: [num_bond_types, num_heads, MLP]
            self.mlps = nn.ModuleList([
                nn.ModuleList([
                    MLP(input_dim=atom_embedding_dim * 2,
                        hidden_dim=mlp_hidden_dim,
                        output_dim=1,
                        num_layers=mlp_num_layers)
                    for _ in range(num_heads)
                ])
                for _ in range(num_bond_types)
            ])
            # Initialize MLP parameters
            for t in range(num_bond_types):
                for h in range(num_heads):
                    self.mlps[t][h].init_params(gain=1.414)
 
        # Optional: keep LeakyReLU if needed elsewhere (not used in score computation here)
        self.leakyrelu = nn.LeakyReLU(negative_slope=output_negative_slope)
 
    def forward(self, src_embeddings, dst_embeddings, edge_embeddings, edge_index=None, edge_attr=None):
        """
        Compute edge-wise attention scores using bond-type-specific bilinear forms or MLPs.
 
        Args:
            src_embeddings: [N, H * D] — source node features (before view)
            dst_embeddings: [N, H * D] — target node features (before view)
            edge_index: [2, E] — (src, dst) indices
            edge_attr: [E, ?] — at least edge_attr[:, 0] is bond_type (long)
 
        Returns:
            attn_scores: [E, num_heads] — raw attention scores (pre-softmax)
        """
        device = src_embeddings.device
        E = edge_index.size(1)
 
        # Reshape to multi-head format: [N, H, D]
        N = src_embeddings.size(0)
        src = src_embeddings.view(N, self.num_heads, self.atom_embedding_dim)
        dst = dst_embeddings.view(N, self.num_heads, self.atom_embedding_dim)
 
        # Initialize output scores
        attn_scores = torch.zeros(E, self.num_heads, device=device, dtype=src.dtype)
 
        bond_types = edge_attr[:, 0]  # [E], must be long tensor in [0, num_bond_types)
 
        # Process each bond type separately
        for t in range(self.num_bond_types):
            mask_t = (bond_types == t)  # [E]
            if not mask_t.any():
                continue
 
            src_idx = edge_index[0, mask_t]  # [E_t]
            dst_idx = edge_index[1, mask_t]  # [E_t]
 
            N_E = dst_idx.shape[0]
            src_t = src[src_idx]  # [E_t, H, D]
            if self.use_edge_embedding:
                dst_t = dst[dst_idx] + edge_embeddings[mask_t].view(N_E, self.num_heads, -1)  # [E_t, H, D]
            else:
                dst_t = dst[dst_idx] 
 
            if self.attention_type == 'bilinear':
                A_t = self.a[t]  # [H, D, D]
                # Compute: score_{e,h} = sum_{i,j} src_{e,h,i} * A_{h,i,j} * dst_{e,h,j}
                # Using einsum: 'ehi,hij,ehj -> eh'
                scores_t = torch.einsum('ehi,hij,ehj->eh', src_t, A_t, dst_t)  # [E_t, H]
            else:
                # MLP attention for each head
                scores_t = []
                for h in range(self.num_heads):
                    # Concatenate src and dst for head h
                    concat_features = torch.cat([src_t[:, h, :], dst_t[:, h, :]], dim=-1)  # [E_t, 2*D]
                    # Apply MLP for head h
                    scores_h = self.mlps[t][h](concat_features).squeeze(-1)  # [E_t]
                    scores_t.append(scores_h)
                scores_t = torch.stack(scores_t, dim=-1)  # [E_t, H]
 
            attn_scores[mask_t] = scores_t
 
        return attn_scores  # [E, H]
 
 
class MultiHeadEdgeAttention_SerialBetweenBondtypes(nn.Module):  
    """  
    A :class:`MultiHeadEdgeAttention_SerialBetweenBondtypes` is a subclass of :class:`nn.Module` that computes multi-head attention scores sequentially between bond types.  
    This module will use less memory compared to the parallel version but may have slightly increased computation time.  
    Supports both bilinear and MLP attention mechanisms.  
    Rewritten to handle edge lists.  
    """  
  
    def __init__(self,
                 atom_embedding_dim: int,
                 num_bond_types: int,
                 num_heads: int = 8, 
                 output_negative_slope: float = 0.2,
                 attention_type: str = default_attention_type,
                 mlp_hidden_dim: int = 1024,
                 mlp_num_layers: int = 2,
                 use_edge_embedding: bool = False):  
        super(MultiHeadEdgeAttention_SerialBetweenBondtypes, self).__init__()  
        self.num_heads = num_heads  
        self.atom_d = atom_embedding_dim  
        self.num_bond_types = num_bond_types  
        self.attention_type = attention_type
        self.output_negative_slope = output_negative_slope
        self.use_edge_embedding = use_edge_embedding
        
        if attention_type not in ['bilinear', 'mlp']:
            raise ValueError(f"attention_type must be 'bilinear' or 'mlp', got {attention_type}")
 
        if attention_type == 'bilinear':
            # Bilinear parameter: [D*H, D, T]
            self.a = nn.Parameter(torch.Tensor(atom_embedding_dim * num_heads, atom_embedding_dim, num_bond_types))  
            nn.init.xavier_uniform_(self.a, gain=1.414)
        else:
            # MLP attention: [T, H, MLP]
            self.mlps = nn.ModuleList([
                nn.ModuleList([
                    MLP(input_dim=atom_embedding_dim * 2,
                        hidden_dim=mlp_hidden_dim,
                        output_dim=1,
                        num_layers=mlp_num_layers)
                    for _ in range(num_heads)
                ])
                for _ in range(num_bond_types)
            ])
            # Initialize MLP parameters
            for t in range(num_bond_types):
                for h in range(num_heads):
                    self.mlps[t][h].init_params(gain=1.414)
  
    def forward(self, src_embeddings, dst_embeddings, edge_embeddings, edge_index=None, edge_attr=None):  
        """  
        Compute multi-head attention scores for edges between source and destination atom embeddings.  
        :param src_embeddings: Source atom embeddings of shape (batch_size, num_atoms, atom_embedding_dim * num_heads) or (total_atoms, atom_embedding_dim * num_heads)  
        :param dst_embeddings: Destination atom embeddings of shape (batch_size, num_atoms, atom_embedding_dim * num_heads) or (total_atoms, atom_embedding_dim * num_heads)  
        :param edge_index: Edge indices of shape (2, num_edges) for sparse format  
        :param edge_attr: Edge attributes of shape (num_edges, 2) for sparse format  
        :param edges: Optional edges tensor of shape (batch_size, num_atoms, num_atoms) for dense format (deprecated)  
        :return: Attention scores of shape (batch_size, num_heads, num_atoms, num_atoms) or (num_edges, num_heads)  
        """   
        B_N, d_ = src_embeddings.shape  
        d = d_ // self.num_heads  
          
        # Get source and target node features for each edge  
        row, col = edge_index  # row=source, col=target  
        src_features = src_embeddings[row]  # [E, d*num_heads] 
        if self.use_edge_embedding: 
            dst_features = dst_embeddings[col]+edge_embeddings  # [E, d*num_heads]  
        else:
            dst_features = dst_embeddings[col]
          
        # Reshape for multi-head processing  
        src_features = src_features.view(-1, self.num_heads, d)  # [E, num_heads, d]  
        dst_features = dst_features.view(-1, self.num_heads, d)  # [E, num_heads, d]  
          
        # Get bond types from edge_attr  
        bond_types = edge_attr[:, 0]  # [E]  
          
        # Initialize attention scores  
        attention_scores = torch.zeros(edge_index.size(1), self.num_heads, device=src_embeddings.device)  # [E, num_heads]  
          
        # Process each bond type serially  
        for t in range(self.num_bond_types):  
            # Get mask for this bond type  
            mask = (bond_types == t)  
            if mask.sum() > 0:  
                # Get edges of this bond type  
                src_t = src_features[mask]  # [E_t, num_heads, d]  
                dst_t = dst_features[mask]  # [E_t, num_heads, d]
                
                if self.attention_type == 'bilinear':
                    A_t = self.a[:, :, t]  # [d*num_heads, d]
                    # Compute attention scores for all heads simultaneously
                    # Reshape A_t for multi-head: [num_heads, d, d]
                    A_t_multihead = A_t.view(self.num_heads, d, d)
                    # Using einsum: 'ehi,hij,ehj -> eh'
                    scores_t = torch.einsum('ehi,hij,ehj->eh', src_t, A_t_multihead, dst_t)  # [E_t, num_heads]
                else:
                    # MLP attention for each head
                    scores_t_list = []
                    for h in range(self.num_heads):
                        # Concatenate src and dst for head h
                        concat_features = torch.cat([src_t[:, h, :], dst_t[:, h, :]], dim=-1)  # [E_t, 2*d]
                        # Apply MLP for head h
                        scores_h = self.mlps[t][h](concat_features).squeeze(-1)  # [E_t]
                        scores_t_list.append(scores_h)
                    scores_t = torch.stack(scores_t_list, dim=-1)  # [E_t, num_heads]
                
                attention_scores[mask] = scores_t
          
        # Apply LeakyReLU activation  
        attention_scores = torch.nn.functional.leaky_relu(attention_scores, negative_slope=self.output_negative_slope)  
          
        return attention_scores
    
class GlobalEdgeAttn(nn.Module):
    """
    A :class:`GlobalEdgeAttn` module that applies global multi-head self-attention within each graph.
    It processes atom embeddings in batched format (with padding) and respects graph boundaries
    via key_padding_mask. This implementation avoids manual Q/K/V projection to ensure numerical stability.
    """

    def __init__(self, embed_dim: int, global_num_heads: int = 8, dropout: float = 0.2):
        super(GlobalEdgeAttn, self).__init__()
        # Use MultiheadAttention with the original embed_dim.
        # PyTorch will internally split it into `global_num_heads` heads.
        self.global_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=global_num_heads,
            dropout=dropout,
            batch_first=False
        )
        # LayerNorm applied after residual connection
        self.norm_layer = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim  

    def forward(self, atom_embeddings, edge_embeddings, batch=None):
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