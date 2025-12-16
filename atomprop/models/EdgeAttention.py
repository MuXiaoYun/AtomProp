"""  
Module for Graph Edge Attention Transformer (GeAT) for molecular property prediction.  
"""  
  
import torch  
import torch.nn as nn  
  
class EdgeAttention(nn.Module):  
    """  
    An :class:`EdgeAttention` is a module for computing attention scores between pairs of atom embeddings in :class:`GeAT`.  
    This module uses a bilinear attention mechanism to compute attention scores.  
    Rewritten to handle edge lists.  
    """  
  
    def __init__(self, atom_embedding_dim: int, num_bond_types: int, output_negative_slope: float = 0.2):  
        super(EdgeAttention, self).__init__()  
        self.atom_d = atom_embedding_dim  
        self.num_bond_types = num_bond_types  
        self.a = nn.Parameter(torch.Tensor(atom_embedding_dim, atom_embedding_dim, num_bond_types)) # (d, d, T)  
        self.output_negative_slope = output_negative_slope  
        nn.init.xavier_uniform_(self.a, gain=1.414)  
  
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
            # Legacy behavior for backward compatibility  
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
          
        # New edge list implementation  
        B_N = atom_embedding_dim = src_embeddings.shape[-1]  
          
        # Get source and target node features for each edge  
        row, col = edge_index  # row=source, col=target  
        src_features = src_embeddings[row]  # [E, d]  
        dst_features = dst_embeddings[col]  # [E, d]  
          
        # Get bond types from edge_attr  
        bond_types = edge_attr[:, 0]  # [E]  
          
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
          
        # Apply LeakyReLU  
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
    Rewritten to handle edge lists.  
    """  
  
    def __init__(self, atom_embedding_dim: int, num_bond_types: int, num_heads: int = 8, output_negative_slope: float = 0.2):  
        super(MultiHeadEdgeAttention_ParallelBetweenBondtypes, self).__init__()  
        self.num_heads = num_heads  
        self.atom_d = atom_embedding_dim  
        self.num_bond_types = num_bond_types  
        self.a = nn.Parameter(torch.Tensor(atom_embedding_dim * num_heads, atom_embedding_dim, num_bond_types))  
        self.output_negative_slope = output_negative_slope  
        nn.init.xavier_uniform_(self.a, gain=1.414)  
          
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
            # Legacy behavior for backward compatibility  
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
          
        # New edge list implementation  
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
          
        # Compute attention scores for each head and bond type  
        attention_scores = torch.zeros(edge_index.size(1), self.num_heads, device=src_embeddings.device)  # [E, num_heads]  
          
        for t in range(self.num_bond_types):  
            # Get mask for this bond type  
            mask = (bond_types == t)  
            if mask.sum() > 0:  
                # Get edges of this bond type  
                src_t = src_features[mask]  # [E_t, num_heads, d]  
                dst_t = dst_features[mask]  # [E_t, num_heads, d]  
                A_t = self.a[:, :, t]  # [d*num_heads, d]  
                  
                # Reshape A_t for multi-head computation  
                A_t = A_t.view(self.num_heads, d, d)  # [num_heads, d, d]  
                  
                # Compute bilinear attention for each head: q^T A k  
                # src_t: [E_t, num_heads, d], A_t: [num_heads, d, d], dst_t: [E_t, num_heads, d]  
                scores_t = torch.einsum('ehd,hed,ehd->eh', src_t, A_t, dst_t)  # [E_t, num_heads]  
                  
                # Store scores  
                attention_scores[mask] = scores_t  
          
        # Apply LeakyReLU  
        attention_scores = torch.nn.functional.leaky_relu(attention_scores, negative_slope=self.output_negative_slope)  
          
        return attention_scores  # [E, num_heads]  
  
class MultiHeadEdgeAttention_SerialBetweenBondtypes(nn.Module):  
    """  
    A :class:`MultiHeadEdgeAttention_SerialBetweenBondtypes` is a subclass of :class:`nn.Module` that computes multi-head attention scores sequentially between bond types.  
    This module will use less memory compared to the parallel version but may have slightly increased computation time.  
    Rewritten to handle edge lists.  
    """  
  
    def __init__(self, atom_embedding_dim: int, num_bond_types: int, num_heads: int = 8, output_negative_slope: float = 0.2):  
        super(MultiHeadEdgeAttention_SerialBetweenBondtypes, self).__init__()  
        self.num_heads = num_heads  
        self.atom_d = atom_embedding_dim  
        self.num_bond_types = num_bond_types  
        self.a = nn.Parameter(torch.Tensor(atom_embedding_dim * num_heads, atom_embedding_dim, num_bond_types))  
        self.output_negative_slope = output_negative_slope  
        nn.init.xavier_uniform_(self.a, gain=1.414)  
  
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
            # Legacy behavior for backward compatibility  
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