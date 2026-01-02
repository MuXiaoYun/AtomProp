"""  
Module for Graph Edge Attention Transformer (GeAT) for molecular property prediction.   
"""  
  
import torch  
import torch.nn as nn  
from atomprop.utils.mlp import MLP, MoE 
from atomprop.embeddings.AtomEmbedding import BondTypes, BondDirections  
from atomprop.models.EdgeAttention import EdgeAttention, MultiHeadEdgeAttention, GlobalEdgeAttn
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
        
        self.edge_w = nn.Linear(embed_dim, embed_dim * num_heads)

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

    def forward(self, atom_embeddings, edge_embeddings, edge_index=None, edge_attr=None):
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
        E = self.edge_w(edge_embeddings)

        # Compute multi-head edge-aware attention scores: [E, num_heads]
        attn_scores = self.edge_attention(
            src_embeddings=Q,
            dst_embeddings=K,
            edge_embeddings=E,
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
  
class GeATConv(nn.Module):  
    """  
    A :class:`GeATConv` is a module for molecular representation learning using GeAT. 
    It uses residual connections and outputs atom embeddings.  
    """  
    def __init__(self, embed_dim: int, num_bond_types: int, num_heads: int = 8, output_negative_slope: float = 0.2, dropout: float = 0.2, geat_num_layers: int = 5):  
        super(GeATConv, self).__init__()  
        self.geat_layers = nn.ModuleList([GeATLayer(embed_dim=embed_dim, num_bond_types=num_bond_types, num_heads=num_heads, output_negative_slope=output_negative_slope, dropout=dropout) for _ in range(geat_num_layers)])  
        self.norm_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(geat_num_layers)])  
  
    def forward(self, atom_embeddings, edge_embeddings, edge_index=None, edge_attr=None):
        atom_embeddings_c = atom_embeddings.clone()
        for i, layer in enumerate(self.geat_layers):
            residual = atom_embeddings_c
            atom_embeddings_c = layer(atom_embeddings_c, edge_embeddings, edge_index, edge_attr)
            atom_embeddings_c = self.norm_layers[i](residual + atom_embeddings_c)
        return atom_embeddings_c
  
class GlobalAttnConv(nn.Module):  
    """  
    A :class:`GlobalAttnConv` is a module for global attention over all atoms in the molecule.
    It uses residual connections and outputs atom embeddings.   
    """  
    def __init__(self, embed_dim: int, global_num_heads: int = 8, dropout: int = 0.2, attn_num_layers: int = 2):  
        super(GlobalAttnConv, self).__init__()  
        self.global_attns = nn.ModuleList([GlobalEdgeAttn(embed_dim=embed_dim, global_num_heads=global_num_heads, dropout=dropout) for _ in range(attn_num_layers)])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(attn_num_layers)])
        
    def forward(self, atom_embeddings, edge_embeddings, batch=None):
        atom_embeddings_c = atom_embeddings.clone()
        for i, layer in enumerate(self.global_attns):
            residual = atom_embeddings_c
            atom_embeddings_c = layer(atom_embeddings_c, edge_embeddings, batch)
            atom_embeddings_c = self.norm_layers[i](residual + atom_embeddings_c)
        return atom_embeddings_c
        
class GeATNet(nn.Module):  
    """  
    A :class:`GeATNet` is a module for molecular embeddings generation using GeAT.  
    1. uses multiple :class:`GeATLayer` instances to compute new embeddings for atoms based on their neighbors. To note, before each inner layer, the embeddings are residual added to the embeddings from the previous layer and then layer normalized.  
    2. applies an extra global attention mechanism to aggregate the information from all atoms.
    3. FFN for all atoms to get the final atom embeddings.  
    """  
      
    def __init__(self,
                 embed_dim: int,
                 num_bond_types = None,
                 num_heads: int = 8,
                 global_num_heads = 8,
                 output_negative_slope: float = 0.2,
                 dropout: int = 0.2,
                 geat_num_layers: int = 4,
                 aggr_num_layers: int = 2,
                 FFN_num_layers: int = 2,
                 FFN_hidden_dim: int = 2048,
                 FFN_num_experts: int = 8,
                 FFN_top_k: int = 2,
                 ):  
        super(GeATNet, self).__init__()
        if num_bond_types is None:
            num_bond_types = len(BondTypes.get_bond_types())+1  
        self.backbone = GeATConv(embed_dim=embed_dim,
                                 num_bond_types=num_bond_types,
                                 num_heads=num_heads,
                                 output_negative_slope=output_negative_slope,
                                 dropout=dropout,
                                 geat_num_layers=geat_num_layers)  
        self.neck = GlobalAttnConv(embed_dim=embed_dim,
                                   global_num_heads=global_num_heads,
                                   dropout=dropout,
                                   attn_num_layers=aggr_num_layers)
        # self.ffn = MLP(input_dim=embed_dim, hidden_dim=FFN_hidden_dim, output_dim=embed_dim, num_layers=FFN_num_layers, dropout=dropout, activation='relu')
        self.ffn = MoE(input_dim=embed_dim,
                       hidden_dim=FFN_hidden_dim,
                       output_dim=embed_dim,
                       num_experts=FFN_num_experts,
                       top_k=FFN_top_k,
                       expert_hidden_layers=FFN_num_layers,
                       dropout=dropout,
                       hidden_activation=nn.ReLU(),
                       output_activation=None)
        self.edge_type_embedding = nn.Embedding(len(BondTypes.get_bond_types())+1, embed_dim)
        self.edge_direction_embedding = nn.Embedding(len(BondDirections.get_bond_directions())+1, embed_dim)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """  
        Reset parameters of the model.  
        """  
        nn.init.xavier_uniform_(self.edge_direction_embedding.weight)
        nn.init.xavier_uniform_(self.edge_type_embedding.weight)      
    
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
        edge_embeddings = self.edge_type_embedding(edge_attr[:,0]) + self.edge_direction_embedding(edge_attr[:,1])
        geat_embeddings = self.backbone(x, edge_embeddings, edge_index, edge_attr)  
        aggr_embeddings = self.neck(geat_embeddings, edge_embeddings, batch)  
        output = self.ffn(aggr_embeddings)
        return output

    def print_params(self):
        """  
        Print the number of trainable parameters for each sub-module and total.  
        """
        def count_params(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        total_params = 0
        print("=" * 60)
        print("Parameter Count per Submodule in GeATNet")
        print("=" * 60)

        # Backbone (GeATConv)
        backbone_params = count_params(self.backbone)
        print(f"{'Backbone (GeATConv)':<40}: {backbone_params:>12,}")
        total_params += backbone_params

        # Neck (GlobalAttnConv)
        neck_params = count_params(self.neck)
        print(f"{'Neck (GlobalAttnConv)':<40}: {neck_params:>12,}")
        total_params += neck_params

        # FFN (MoE)
        ffn_params = count_params(self.ffn)
        print(f"{'FFN (MoE)':<40}: {ffn_params:>12,}")
        total_params += ffn_params

        # Edge type embedding
        edge_type_params = count_params(self.edge_type_embedding)
        print(f"{'Edge Type Embedding':<40}: {edge_type_params:>12,}")
        total_params += edge_type_params

        # Edge direction embedding
        edge_dir_params = count_params(self.edge_direction_embedding)
        print(f"{'Edge Direction Embedding':<40}: {edge_dir_params:>12,}")
        total_params += edge_dir_params

        print("-" * 60)
        print(f"{'Total Trainable Parameters':<40}: {total_params:>12,}")
        print("=" * 60)