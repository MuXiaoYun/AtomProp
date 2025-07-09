import torch
import torch.nn as nn

bond_types = [
    "UNSPECIFIED", "SINGLE", "DOUBLE", "TRIPLE",
    "QUADRUPLE", "QUINTUPLE", "HEXTUPLE", "ONEANDAHALF",
    "TWOANDAHALF", "THREEANDAHALF", "FOURANDAHALF", "FIVEANDAHALF",
    "AROMATIC", "IONIC", "HYDROGEN", "THREECENTER",
    "DATIVEONE", "DATIVE", "DATIVEL", "DATIVER",
    "OTHER", "ZERO"
]
"""
Records the bond types used in the dataset
"""

class AtomEmbedding(nn.Module):
    """
    An :class:`AtomEmbedding` is a module for embedding atom types into a continuous vector space.
    """

    def __init__(self, atom_embedding_dim: int, num_atom_types: int):
        super(AtomEmbedding, self).__init__()
        self.atom_embedding_dim = atom_embedding_dim
        self.num_atom_types = num_atom_types
        self.embedding = nn.Embedding(num_embeddings=num_atom_types, embedding_dim=atom_embedding_dim)

    def forward(self, atom_type_indices):
        return self.embedding(atom_type_indices)