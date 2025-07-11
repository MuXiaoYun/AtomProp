import torch
import torch.nn as nn
import rdkit.Chem as Chem

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

class SMILESToInputs:
    """
    A utility class to convert SMILES strings to atom type indices and edges.
    """
    @staticmethod
    def convert(smiles: str, context_length_node: int = 420, context_length_edge: int = 420):
        atom_type_indices = []
        edges = []
        # Create nodes with atom types
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        for atom in mol.GetAtoms():
            atom_type_indices.append(atom.GetAtomicNum())
        # Pad atom_type_indices to context_length
        if len(atom_type_indices) < context_length_node:
            atom_type_indices += [0] * (context_length_node - len(atom_type_indices))
        elif len(atom_type_indices) > context_length_node:
            atom_type_indices = atom_type_indices[:context_length_node]
        # Create edges with bond types
        for bond in mol.GetBonds():
            src = bond.GetBeginAtomIdx()
            dst = bond.GetEndAtomIdx()
            # Get the bond type and map it to an index in `bond_types`
            bond_type = str(bond.GetBondType())
            if bond_type not in bond_types:
                raise ValueError(f"Unknown bond type: {bond_type}")
            bond_type_index = bond_types.index(bond_type)
            edges.append(((src, dst), bond_type_index))
        # Pad edges to context_length
        if len(edges) < context_length_edge:
            edges += [((-1, -1), -1)] * (context_length_edge - len(edges))
        elif len(edges) > context_length_edge:
            edges = edges[:context_length_edge]
        # Convert all to tensors
        atom_type_indices = torch.tensor(atom_type_indices, dtype=torch.long)
        edges = [((torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)), torch.tensor(bond_type_index, dtype=torch.long)) for ((src, dst), bond_type_index) in edges]
        return (atom_type_indices, edges), mol


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