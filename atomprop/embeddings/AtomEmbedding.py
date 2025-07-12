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
    
class BondTypeNamesToIndices:
    """
    A utility class to convert bond type names to indices.
    """
    @staticmethod
    def convert(bond_type_names: list[str]):
        """
        Convert a list of bond type names to their corresponding indices.
        """
        bond_type_indices = []
        for bond_type in bond_type_names:
            if bond_type not in bond_types:
                raise ValueError(f"Unknown bond type: {bond_type}")
            bond_type_indices.append(bond_types.index(bond_type))
        return torch.tensor(bond_type_indices, dtype=torch.long)
    
class SMILESToInputs:
    """
    A utility class to convert SMILES strings to atom type indices and edges.
    """
    @staticmethod
    def convert(smiles: str, context_length: int = 420):
        """
        Convert a SMILES string to atom type indices and edges.
        :param smiles: The SMILES string to convert.
        :param context_length: The maximum number of atoms in a molecule for padding.
        :return: A tuple containing the atom embeddings and edge (adj matrix, value is bond type index, -1 for no edge), and the RDKit molecule object.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        
        # Get atom type indices
        atom_type_indices = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        atom_type_indices = torch.tensor(atom_type_indices, dtype=torch.long)
        # Pad atom type indices to context length with zeros
        if len(atom_type_indices) < context_length:
            atom_type_indices = torch.cat([atom_type_indices, torch.zeros(context_length - len(atom_type_indices), dtype=torch.long)])
        else:
            atom_type_indices = atom_type_indices[:context_length] 

        # Get edges (bond type indices)
        edges = []
        for bond in mol.GetBonds():
            start_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            bond_type_index = bond_types.index(str(bond.GetBondType()))
            edges.append((start_idx, end_idx, bond_type_index))
        
        # Create adjacency matrix
        adj_matrix = torch.zeros((context_length, context_length), dtype=torch.long) - 1
        for start_idx, end_idx, bond_type_index in edges:
            adj_matrix[start_idx, end_idx] = bond_type_index
            adj_matrix[end_idx, start_idx] = bond_type_index
        # Pad atom type indices and adjacency matrix
        if len(atom_type_indices) < context_length:
            atom_type_indices = torch.cat([atom_type_indices, torch.zeros(context_length - len(atom_type_indices), dtype=torch.long)])
        else:
            atom_type_indices = atom_type_indices[:context_length]
        if adj_matrix.size(0) < context_length:
            adj_matrix = torch.cat([adj_matrix, torch.zeros(context_length - adj_matrix.size(0), context_length, dtype=torch.long) - 1], dim=0)
            adj_matrix = torch.cat([adj_matrix, torch.zeros(context_length, context_length - adj_matrix.size(1), dtype=torch.long) - 1], dim=1)
        else:
            adj_matrix = adj_matrix[:context_length, :context_length]
        return atom_type_indices, adj_matrix, mol