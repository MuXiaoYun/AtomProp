"""
Library module for embedding atom types.
"""

import torch
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