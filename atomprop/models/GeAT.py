import torch
import torch.nn as nn
from atomprop.embeddings.AtomEmbedding import AtomEmbedding   

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
        self.parallel_between_bondtypes = parallel_between_bondtypes
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
    
class GeATLayer(nn.Module):
    """
    A :class:`GeAT` (Graph Edge Attention Transformer) for molecular property prediction.
    This model weighs the importance of only neighboring atoms.
    """

    def __init__(self, atom_embedding_dim: int, num_bond_types: int, num_heads: int = 8, output_negative_slope: float = 0.2):
        super(GeATLayer, self).__init__()
        self.num_heads = num_heads
        self.atom_embedding_dim = atom_embedding_dim
        self.num_bond_types = num_bond_types
        # Linear layers for query, key, and value transformations
        self.Q_w = nn.Linear(atom_embedding_dim, atom_embedding_dim*num_heads)
        self.K_w = nn.Linear(atom_embedding_dim, atom_embedding_dim*num_heads)
        self.V_w = nn.Linear(atom_embedding_dim, atom_embedding_dim*num_heads)
        # for each bond type, we use a different attention mechanism
        self.edge_attentions = MultiHeadEdgeAttention(atom_embedding_dim, num_bond_types, num_heads, output_negative_slope)
        self.project = nn.Linear(atom_embedding_dim * num_heads, atom_embedding_dim)

    def forward(self, atom_embeddings, edges = None):
        B = atom_embeddings.size(0)
        N = atom_embeddings.size(1)
        src_embeddings = self.Q_w(atom_embeddings)
        dst_embeddings = self.K_w(atom_embeddings)
        value_embeddings = self.V_w(atom_embeddings)
        # Calculate attention scores
        attention_scores = self.edge_attentions(src_embeddings, dst_embeddings, edges)
        # Softmax
        attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1)
        # Compute attention output, which is suppose to be (b, n, d*num_heads)
        attention_output = torch.matmul(attention_scores, value_embeddings.reshape(B, -1, self.num_heads, self.atom_embedding_dim).permute(0, 2, 1, 3)) # (b, num_heads, n, d)
        attention_output = attention_output.reshape(B, N, -1) # (b, n, d*num_heads)
        # Project to atom embedding dimension
        attention_output = self.project(attention_output)
        return attention_output
    
class GeATLayerWithSingleHead(nn.Module):
    """
    A :class:`GeATLayerWithSingleHead` is a simplified version of :class:`GeATLayer` that uses a single head for attention.
    This model weighs the importance of only neighboring atoms.
    """

    def __init__(self, atom_embedding_dim: int, num_bond_types: int, output_negative_slope: float = 0.2):
        super(GeATLayerWithSingleHead, self).__init__()
        self.Q_w = nn.Linear(atom_embedding_dim, atom_embedding_dim)
        self.K_w = nn.Linear(atom_embedding_dim, atom_embedding_dim)
        self.V_w = nn.Linear(atom_embedding_dim, atom_embedding_dim)
        self.edge_attention = EdgeAttention(atom_embedding_dim, num_bond_types, output_negative_slope)
        self.project = nn.Linear(atom_embedding_dim, atom_embedding_dim)

    def forward(self, atom_embeddings, edges = None):
        src_embeddings = self.Q_w(atom_embeddings)
        dst_embeddings = self.K_w(atom_embeddings)
        value_embeddings = self.V_w(atom_embeddings)
        # Calculate attention scores
        attention_scores = self.edge_attention(src_embeddings, dst_embeddings, edges)
        # Softmax
        attention_scores = torch.nn.functional.softmax(attention_scores, dim=-1)
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
    """
    def __init__(self, atom_embedding_dim: int, num_atom_types: int, num_bond_types: int, num_heads: int = 8, global_num_heads = 8, hidden_dim: int = 64, output_negative_slope: float = 0.2, dropout: int = 0.1, geat_num_layers: int = 3):
        super(GeATNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.atom_embedding = AtomEmbedding(atom_embedding_dim=atom_embedding_dim, num_atom_types=num_atom_types)
        self.geat_layers = nn.ModuleList([GeATLayer(atom_embedding_dim=atom_embedding_dim, num_bond_types=num_bond_types, num_heads=num_heads, output_negative_slope=output_negative_slope) for _ in range(geat_num_layers)])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(atom_embedding_dim) for _ in range(geat_num_layers)])
        self.global_attention = nn.MultiheadAttention(embed_dim=atom_embedding_dim*global_num_heads, num_heads=global_num_heads, dropout=dropout)
        self.Q_w_global = nn.Linear(atom_embedding_dim, atom_embedding_dim*global_num_heads)
        self.K_w_global = nn.Linear(atom_embedding_dim, atom_embedding_dim*global_num_heads)
        self.V_w_global = nn.Linear(atom_embedding_dim, atom_embedding_dim*global_num_heads)
        self.fc1 = nn.Linear(atom_embedding_dim*global_num_heads, self.hidden_dim)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=output_negative_slope)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=output_negative_slope)
        self.output = nn.Linear(self.hidden_dim, 1)
        
    def forward(self, atoms, edges):
        """
        Forward pass of the GeATNet.
        :param atoms: Atom type indices of shape (batch_size, num_atoms)
        :param edges: Edge indices of shape (batch_size, num_atoms, num_atoms)
        :return: Predicted property of shape (batch_size, 1)
        """
        atom_embeddings = self.atom_embedding(atoms)
        for i, layer in enumerate(self.geat_layers):
            atom_embeddings = atom_embeddings + layer(atom_embeddings, edges)
            atom_embeddings = self.norm_layers[i](atom_embeddings)
        atom_embeddings = atom_embeddings.reshape(atom_embeddings.size(0), -1, atom_embeddings.size(-1))
        global_q = self.Q_w_global(atom_embeddings)
        global_k = self.K_w_global(atom_embeddings)
        global_v = self.V_w_global(atom_embeddings)
        global_attention_output, _ = self.global_attention(global_q, global_k, global_v)
        global_attention_output = global_attention_output.mean(dim=1)
        x = self.fc1(global_attention_output)
        x = self.leaky_relu1(x)
        x = self.fc2(x)
        x = self.leaky_relu2(x)
        x = self.output(x)
        return x