"""  
Module for Graph Edge Attention Transformer (GeAT) for molecular property prediction.  
"""  
  
import torch  
import torch.nn as nn  
import torch.nn.functional as F
from atomprop.utils.mlp import MLP

default_attention_type = 'mlp'
 
class EdgeAttention(nn.Module):  
    """  
    An :class:`EdgeAttention` is a module for computing attention scores between pairs of atom embeddings in :class:`GeAT`.  
    This module supports two attention mechanisms: 'bilinear' and 'mlp'.  
    - 'bilinear': Uses bilinear transformation to compute attention scores (default)  
    - 'mlp': Uses a small MLP on concatenated inputs to compute attention scores  
    Rewritten to handle edge lists.  
    """  
  
    def __init__(self, atom_embedding_dim: int, num_bond_types: int, 
                 output_negative_slope: float = 0.2, attention_type: str = default_attention_type,
                 mlp_hidden_dim: int = 512, mlp_num_layers: int = 3):  
        super(EdgeAttention, self).__init__()  
        self.atom_d = atom_embedding_dim  
        self.num_bond_types = num_bond_types  
        self.attention_type = attention_type  
        self.output_negative_slope = output_negative_slope  
        
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
                    num_layers=mlp_num_layers)
                for _ in range(num_bond_types)
            ])
            # Initialize MLP parameters
            for mlp in self.mlps:
                mlp.init_params(gain=1.414)
  
    def forward(self, src_embeddings, dst_embeddings, edge_index=None, edge_attr=None, edges=None):  
        """  
        Compute attention scores for edges between source and destination atom embeddings.  
        :param src_embeddings: Source atom embeddings of shape (batch_size, num_atoms, atom_embedding_dim) or (total_atoms, atom_embedding_dim)  
        :param dst_embeddings: Destination atom embeddings of shape (batch_size, num_atoms, atom_embedding_dim) or (total_atoms, atom_embedding_dim)  
        :param edge_index: Edge indices of shape (2, num_edges) for sparse format  
        :param edge_attr: Edge attributes of shape (num_edges, 2) for sparse format (bond_type, bond_direction)  
        :param edges: Optional edges tensor of shape (batch_size, num_atoms, num_atoms) for dense format (deprecated)  
        :return: Attention scores of shape (batch_size, num_atoms, num_atoms) or (num_edges,)  
        """  
        # Handle both old edge matrix and new edge list formats  
        if edges is not None:  
            # Legacy behavior for backward compatibility - only supported for bilinear attention
            if self.attention_type == 'mlp':
                raise NotImplementedError("MLP attention does not support legacy edge matrix format. Use edge_index instead.")
                
            B, N, d = src_embeddings.shape  
            T = self.num_bond_types  
            # attention = qT a k
            transformed_src = (src_embeddings @ self.a.view(d, d*T)).view(B, N, d, T)  # (B, N, d, T)  
            attention_scores = (transformed_src.transpose(2, 3).reshape(B, N*T, d) @ dst_embeddings.transpose(1, 2)).transpose(1, 2).view(B, N, N, T)  # (B, N, N, T)  
            # Edge mask  
            attention_scores = attention_scores.gather(-1, edges.clamp(min=0).unsqueeze(-1)).squeeze(-1)  # (B, N, N)  
            attention_scores = attention_scores.masked_fill(edges==-1, -1e10)  # (B, N, N)  
            # Leaky ReLU activation  
            attention_scores = torch.nn.functional.leaky_relu(attention_scores, negative_slope=self.output_negative_slope)  
            return attention_scores  
          
        # New edge list implementation  
        atom_embedding_dim = src_embeddings.shape[-1]  
          
        # Get source and target node features for each edge  
        row, col = edge_index  # row=source, col=target  
        src_features = src_embeddings[row]  # [E, d]  
        dst_features = dst_embeddings[col]  # [E, d]  
          
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
        mlp_hidden_dim: int = 512,
        mlp_num_layers: int = 3,
    ):
        super().__init__()
        self.atom_embedding_dim = atom_embedding_dim
        self.num_bond_types = num_bond_types
        self.num_heads = num_heads
        self.attention_type = attention_type
        
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
 
    def forward(
        self,
        src_embeddings: torch.Tensor,
        dst_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
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
 
            src_t = src[src_idx]  # [E_t, H, D]
            dst_t = dst[dst_idx]  # [E_t, H, D]
 
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
  
    def __init__(self, atom_embedding_dim: int, num_bond_types: int, num_heads: int = 8, 
                 output_negative_slope: float = 0.2, attention_type: str = default_attention_type,
                 mlp_hidden_dim: int = 512, mlp_num_layers: int = 3):  
        super(MultiHeadEdgeAttention_SerialBetweenBondtypes, self).__init__()  
        self.num_heads = num_heads  
        self.atom_d = atom_embedding_dim  
        self.num_bond_types = num_bond_types  
        self.attention_type = attention_type
        self.output_negative_slope = output_negative_slope
        
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
  
    def forward(self, src_embeddings, dst_embeddings, edge_index=None, edge_attr=None, edges=None):  
        """  
        Compute multi-head attention scores for edges between source and destination atom embeddings.  
        :param src_embeddings: Source atom embeddings of shape (batch_size, num_atoms, atom_embedding_dim * num_heads) or (total_atoms, atom_embedding_dim * num_heads)  
        :param dst_embeddings: Destination atom embeddings of shape (batch_size, num_atoms, atom_embedding_dim * num_heads) or (total_atoms, atom_embedding_dim * num_heads)  
        :param edge_index: Edge indices of shape (2, num_edges) for sparse format  
        :param edge_attr: Edge attributes of shape (num_edges, 2) for sparse format  
        :param edges: Optional edges tensor of shape (batch_size, num_atoms, num_atoms) for dense format (deprecated)  
        :return: Attention scores of shape (batch_size, num_heads, num_atoms, num_atoms) or (num_edges, num_heads)  
        """  
        # Handle both old edge matrix and new edge list formats  
        if edges is not None:  
            # Legacy behavior for backward compatibility - only supported for bilinear attention
            if self.attention_type == 'mlp':
                raise NotImplementedError("MLP attention does not support legacy edge matrix format. Use edge_index instead.")
                
            B, N, d_ = src_embeddings.shape  
            d = d_ // self.num_heads  
            T = self.num_bond_types  
            # attention = qT a k  
            src_embeddings = src_embeddings.reshape(B, N, self.num_heads, d).permute(0, 2, 1, 3)  # (B, num_heads, N, d)  
            dst_embeddings = dst_embeddings.reshape(B, N, self.num_heads, d).permute(0, 2, 1, 3)  # (B, num_heads, N, d)  
            attention_scores = []  
            for t in range(T):  
                transformed_src = (src_embeddings @ self.a[:, :, t].view(self.num_heads, d, 1)).view(B, self.num_heads, N, d)  
                attention_scores_t = (transformed_src.transpose(2, 3).reshape(B, self.num_heads, N, d) @ dst_embeddings.transpose(2, 3)).transpose(2, 3).view(B, self.num_heads, N, N)  # (B, num_heads, N, N)  
                # Edge mask  
                edges_t = edges == t  
                attention_scores_t = attention_scores_t.masked_fill(edges_t.unsqueeze(1), -1e10)  # (B, num_heads, N, N)  
                # Leaky ReLU activation  
                attention_scores_t = torch.nn.functional.leaky_relu(attention_scores_t, negative_slope=self.output_negative_slope)  
                attention_scores.append(attention_scores_t.unsqueeze(-1))  # (B, num_heads, N, N, 1)  
            attention_scores = torch.cat(attention_scores, dim=-1)  # (B, num_heads, N, N, T)  
            attention_scores = attention_scores.sum(dim=-1)  # Sum over bond types to get final attention scores  
            return attention_scores  # (B, num_heads, N, N)  
          
        # New edge list implementation - serial processing  
        B_N, d_ = src_embeddings.shape  
        d = d_ // self.num_heads  
          
        # Get source and target node features for each edge  
        row, col = edge_index  # row=source, col=target  
        src_features = src_embeddings[row]  # [E, d*num_heads]  
        dst_features = dst_embeddings[col]  # [E, d*num_heads]  
          
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