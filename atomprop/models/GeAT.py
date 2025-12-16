"""  
Module for Graph Edge Attention Transformer (GeAT) for molecular property prediction.   
"""  
  
import torch  
import torch.nn as nn  
from atomprop.embeddings.AtomEmbedding import AtomEmbedding  
from atomprop.utils.mlp import MLP  
from atomprop.embeddings.AtomEmbedding import BondTypes, BondDirections  
from atomprop.models.EdgeAttention import EdgeAttention, MultiHeadEdgeAttention 
import torch_geometric 
  
class GeATLayer(nn.Module):  
    """  
    A :class:`GeAT` (Graph Edge Attention Transformer) for molecular property prediction.  
    This model weighs the importance of only neighboring atoms.  
    Rewritten to handle edge lists instead of edge matrices.  
    """  
  
    def __init__(self, embed_dim: int, num_bond_types: int, num_heads: int = 8, dropout: float = 0.2, output_negative_slope: float = 0.2):  
        super(GeATLayer, self).__init__()  
        self.num_heads = num_heads  
        self.embed_dim = embed_dim  
        self.num_bond_types = num_bond_types  
          
        # Linear layers for query, key, and value transformations  
        self.Q_w = nn.Linear(embed_dim, embed_dim * num_heads)  
        self.K_w = nn.Linear(embed_dim, embed_dim * num_heads)  
        self.V_w = nn.Linear(embed_dim, embed_dim * num_heads)  
          
        # Edge embeddings for bond types and directions  
        self.edge_type_embedding = nn.Embedding(len(BondTypes.get_bond_types()) + 1, embed_dim * num_heads)  
        self.edge_direction_embedding = nn.Embedding(len(BondDirections.get_bond_directions()) + 1, embed_dim * num_heads)  
          
        self.dropout_layer = nn.Dropout(dropout)  
        self.project = nn.Linear(embed_dim * num_heads, embed_dim)  
        self.leaky_relu = nn.LeakyReLU(negative_slope=output_negative_slope)  
  
    def forward(self, atom_embeddings, edge_index=None, edge_attr=None):  
        """  
        Forward pass with edge list inputs.  
        :param atom_embeddings: [B_N, embed_dim] or [B, N, embed_dim]  
        :param edge_index: [2, E] - sparse edge indices  
        :param edge_attr: [E, 2] - edge attributes (bond_type, bond_direction)   
        """  
        B_N = atom_embeddings.size(0)  
          
        # Compute Q, K, V for all nodes  
        src_embeddings = self.Q_w(atom_embeddings)  # [B_N, embed_dim * num_heads]  
        dst_embeddings = self.K_w(atom_embeddings)  # [B_N, embed_dim * num_heads]  
        value_embeddings = self.V_w(atom_embeddings)  # [B_N, embed_dim * num_heads]  
          
        # Get source and target node features for each edge  
        row, col = edge_index  # row=source, col=target  
        src_features = src_embeddings[row]  # [E, embed_dim * num_heads]  
        dst_features = dst_embeddings[col]  # [E, embed_dim * num_heads]  
        value_features = value_embeddings[row]  # [E, embed_dim * num_heads]  
          
        # Compute edge embeddings  
        edge_embeddings = (self.edge_type_embedding(edge_attr[:, 0]) +   
                          self.edge_direction_embedding(edge_attr[:, 1]))  # [E, embed_dim * num_heads]  
          
        # Compute attention scores for each edge  
        # Dot product attention between source and target, modulated by edge embeddings  
        attention_scores = (src_features * dst_features).sum(dim=-1) / (self.embed_dim * self.num_heads) ** 0.5  
        attention_scores = attention_scores + (src_features * edge_embeddings).sum(dim=-1) * 0.1  # Edge modulation  
        attention_scores = self.leaky_relu(attention_scores)  
          
        # Apply softmax over neighbors for each target node  
        attention_scores = torch_geometric.utils.softmax(attention_scores, col, num_nodes=B_N)  
        attention_scores = self.dropout_layer(attention_scores)  
          
        # Aggregate messages  
        out = torch.zeros(B_N, self.embed_dim * self.num_heads, 
                         device=atom_embeddings.device, dtype=atom_embeddings.dtype) 
        out = out.index_add(0, col, attention_scores.unsqueeze(-1) * value_features)  # Scatter add to target nodes  
          
        # Project back to original dimension  
        out = self.project(out)  # [B_N, embed_dim]  
          
        return out  
  
class GeATLayerWithSingleHead(nn.Module):  
    """  
    A :class:`GeATLayerWithSingleHead` is a simplified version of :class:`GeATLayer` that uses a single head for attention.  
    This model weighs the importance of only neighboring atoms.  
    """  
  
    def __init__(self, embed_dim: int, num_bond_types: int, dropout: float = 0.2, output_negative_slope: float = 0.2):  
        super(GeATLayerWithSingleHead, self).__init__()  
        self.Q_w = nn.Linear(embed_dim, embed_dim)  
        self.K_w = nn.Linear(embed_dim, embed_dim)  
        self.V_w = nn.Linear(embed_dim, embed_dim)  
        self.edge_attention = EdgeAttention(embed_dim, num_bond_types, output_negative_slope)  
        self.dropout_layer = nn.Dropout(dropout)  
        self.project = nn.Linear(embed_dim, embed_dim)  
  
    def forward(self, atom_embeddings, edge_index=None, edge_attr=None):  
        B_N = atom_embeddings.size(0)  
          
        src_embeddings = self.Q_w(atom_embeddings)  
        dst_embeddings = self.K_w(atom_embeddings)  
        value_embeddings = self.V_w(atom_embeddings)  
          
        row, col = edge_index  
        src_features = src_embeddings[row]  
        dst_features = dst_embeddings[col]  
        value_features = value_embeddings[row]  
          
        # Simple dot product attention  
        attention_scores = (src_features * dst_features).sum(dim=-1) / (self.embed_dim) ** 0.5  
        attention_scores = torch.nn.functional.leaky_relu(attention_scores, negative_slope=0.2)  
        attention_scores = torch_geometric.utils.softmax(attention_scores, col, num_nodes=B_N)  
        attention_scores = self.dropout_layer(attention_scores)  
          
        out = torch.zeros(B_N, self.embed_dim * self.num_heads, 
                         device=atom_embeddings.device, dtype=atom_embeddings.dtype)  
        out = out.index_add(0, col, attention_scores.unsqueeze(-1) * value_features)  
        out = self.project(out)  
          
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
        """  
        Forward pass with edge list inputs.  
        :param atoms: [B_N, 2] (atom_type, chirality)  
        :param edge_index: [2, E] - sparse edge indices  
        :param edge_attr: [E, 2] - edge attributes  
        """
        atom_embeddings_c = atom_embeddings.clone()
        for i, layer in enumerate(self.geat_layers):  
            atom_embeddings_c = atom_embeddings_c + layer(atom_embeddings_c, edge_index, edge_attr)
        return atom_embeddings_c  
  
class GeATNeck(nn.Module):  
    """  
    A :class:`GeATNeck` is a module for global attention mechanism in GeAT. 
    It leverages graph infomation from batch, uses global attention for aggregation in each graph and outputs atom embeddings.  
    """  

    def __init__(self, embed_dim: int, global_num_heads: int = 8, dropout: float = 0.2):  
        super(GeATNeck, self).__init__()  
        # Multi-head attention for global attention within each graph
        self.global_attention = nn.MultiheadAttention(
            embed_dim=embed_dim * global_num_heads, 
            num_heads=global_num_heads, 
            dropout=dropout,
            batch_first=False
        )  
        # Linear transformations
        self.Q_w_global = nn.Linear(embed_dim, embed_dim * global_num_heads)  
        self.K_w_global = nn.Linear(embed_dim, embed_dim * global_num_heads)  
        self.V_w_global = nn.Linear(embed_dim, embed_dim * global_num_heads)  
        self.norm_layer = nn.LayerNorm(embed_dim * global_num_heads)  
        self.embed_dim = embed_dim
        self.global_num_heads = global_num_heads
        self.proj = nn.Linear(embed_dim * global_num_heads, embed_dim)  # Project back to original dimension
        
    def forward(self, atom_embeddings: torch.Tensor, batch: torch.Tensor = None) -> torch.Tensor:  
        """
        Forward pass with batched processing of all graphs.
        
        Args:
            atom_embeddings: [B_N, embed_dim] - Atom embeddings
            batch: [B_N] - Batch indices
            
        Returns:
            torch.Tensor: Updated atom embeddings
        """
        assert batch is not None
        
        B_N = atom_embeddings.size(0)
        device = atom_embeddings.device
        dtype = atom_embeddings.dtype
        
        # Transform to multi-head dimension
        Q = self.Q_w_global(atom_embeddings)  # [B_N, embed_dim*global_num_heads]
        K = self.K_w_global(atom_embeddings)  # [B_N, embed_dim*global_num_heads]
        V = self.V_w_global(atom_embeddings)  # [B_N, embed_dim*global_num_heads]
        
        # Get batch information
        batch_size = batch.max().item() + 1
        
        # Find maximum graph size for padding
        graph_sizes = []
        graph_indices_list = []
        for i in range(batch_size):
            mask = (batch == i)
            indices = mask.nonzero(as_tuple=True)[0]
            graph_sizes.append(len(indices))
            graph_indices_list.append(indices)
        
        max_graph_size = max(graph_sizes)
        
        # Create padded tensors for batch processing
        Q_padded = torch.zeros(batch_size, max_graph_size, 
                              self.embed_dim * self.global_num_heads,
                              device=device, dtype=dtype)
        K_padded = torch.zeros_like(Q_padded)
        V_padded = torch.zeros_like(Q_padded)
        
        # Fill padded tensors
        for i in range(batch_size):
            indices = graph_indices_list[i]
            if len(indices) > 0:
                Q_padded[i, :len(indices)] = Q[indices]
                K_padded[i, :len(indices)] = K[indices]
                V_padded[i, :len(indices)] = V[indices]
        
        # Create key padding mask (for MultiheadAttention)
        # True positions will be ignored in attention
        key_padding_mask = torch.ones(batch_size, max_graph_size, dtype=torch.bool, device=device)
        for i in range(batch_size):
            key_padding_mask[i, :graph_sizes[i]] = False  # Valid positions are False
        
        # Transpose for MultiheadAttention: [seq_len, batch_size, embed_dim]
        Q_t = Q_padded.transpose(0, 1)  # [max_graph_size, batch_size, embed_dim*global_num_heads]
        K_t = K_padded.transpose(0, 1)  # [max_graph_size, batch_size, embed_dim*global_num_heads]
        V_t = V_padded.transpose(0, 1)  # [max_graph_size, batch_size, embed_dim*global_num_heads]
        
        # Apply global attention with padding mask
        attn_output, _ = self.global_attention(
            Q_t, 
            K_t, 
            V_t,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        
        # Transpose back: [batch_size, max_graph_size, embed_dim*global_num_heads]
        attn_output = attn_output.transpose(0, 1)
        # Apply residual connection and layer norm
        output_padded = self.norm_layer(attn_output + Q_padded)
        # Gather back to original order
        final_output = torch.zeros(B_N, self.embed_dim * self.global_num_heads, 
                                 device=device, dtype=dtype)
        for i in range(batch_size):
            indices = graph_indices_list[i]
            if len(indices) > 0:
                final_output[indices] = output_padded[i, :len(indices)]
        
        # Project back to original dimension
        final_output = self.proj(final_output)  # [B_N, embed_dim]
        return final_output
        
class GeATNet(nn.Module):  
    """  
    A :class:`GeATNet` is a module for molecular property prediction using GeAT. It outputs a single scalar value representing the predicted property of the molecule.  
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