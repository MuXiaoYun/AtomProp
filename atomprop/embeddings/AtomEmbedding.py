"""
Library module for embedding atom types.
"""

import torch
import torch.nn as nn
import rdkit.Chem as Chem

"""
Records the bond types used in the dataset
"""

class BondTypes:
    """
    A utility class to handle bond types.
    """
    # bond_types = [
    #     "UNSPECIFIED", "SINGLE", "DOUBLE", "TRIPLE",
    #     "QUADRUPLE", "QUINTUPLE", "HEXTUPLE", "ONEANDAHALF",
    #     "TWOANDAHALF", "THREEANDAHALF", "FOURANDAHALF", "FIVEANDAHALF",
    #     "AROMATIC", "IONIC", "HYDROGEN", "THREECENTER",
    #     "DATIVEONE", "DATIVE", "DATIVEL", "DATIVER",
    #     "OTHER", "ZERO"
    # ]
    bond_types = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]

    @staticmethod
    def get_bond_types():
        return BondTypes.bond_types

    @staticmethod
    def set_bond_types(new_bond_types: list[str]):
        BondTypes.bond_types = new_bond_types
        
    @staticmethod
    def get_index(input: any):
        return BondTypes.bond_types.index(input) if input in BondTypes.bond_types else len(BondTypes.bond_types)

class BondDirections:
    """
    A utility class to handle bond directions.
    """
    # bond_directions = [
    #     "NONE",
    #     "BEGINWEDGE",
    #     "BEGINDASH",
    #     "ENDDOWNRIGHT",
    #     "ENDUPRIGHT",
    #     "EITHERDOUBLE",
    #     "UNKNOWN"
    # ]
    bond_directions = [
        "NONE",
        "ENDDOWNRIGHT",
        "ENDUPRIGHT",
    ]
    
    @staticmethod
    def get_bond_directions():
        return BondDirections.bond_directions
    
    @staticmethod
    def set_bond_directions(new_bond_directions: list[str]):
        BondDirections.bond_directions = new_bond_directions

    @staticmethod
    def get_index(input: any):
        return BondDirections.bond_directions.index(input) if input in BondDirections.bond_directions else len(BondDirections.bond_directions)

class AtomChirals:
    """
    A utility class to handle atom chirals.
    """
    # atom_chirals = [
    #     Chem.rdchem.ChiralType.CHI_ALLENE,
    #     Chem.rdchem.ChiralType.CHI_OCTAHEDRAL,
    #     Chem.rdchem.ChiralType.CHI_SQUAREPLANAR,
    #     Chem.rdchem.ChiralType.CHI_TETRAHEDRAL,
    #     Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    #     Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    #     Chem.rdchem.ChiralType.CHI_TRIGONALBIPYRAMIDAL
    # ]
    atom_chirals = [
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW
    ]

    @staticmethod
    def get_atom_chirals():
        return AtomChirals.atom_chirals
    
    @staticmethod
    def set_atom_chirals(new_atom_chirals: list):
        AtomChirals.atom_chirals = new_atom_chirals
        
    @staticmethod
    def get_index(input: any):
        return AtomChirals.atom_chirals.index(input) if input in AtomChirals.atom_chirals else len(AtomChirals.atom_chirals)

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
            if bond_type not in BondTypes.get_bond_types():
                raise ValueError(f"Unknown bond type: {bond_type}")
            bond_type_indices.append(BondTypes.get_bond_types().index(bond_type))
        return torch.tensor(bond_type_indices, dtype=torch.long)