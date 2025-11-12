"""
Module for generating A-B-C triplet or A-B-C-D quadruplet atom groups from molecules.
These groups can be used for various sub-structure level prediction tasks.
"""

from rdkit import Chem

class TripletGroup:
    """
    Generate A-B-C triplet atom groups from a molecule.
    1st atom -- center atom -- 3rd atom
    """

    @staticmethod
    def generate(mol):
        """
        Generate triplet groups from the given molecule.
        :param mol: RDKit molecule object.
        :return: List of triplet groups, each represented as a tuple of atom indices (a_idx, b_idx, c_idx).
        """
        triplet_groups = []
        for bond in mol.GetBonds():
            a_atom = bond.GetBeginAtom()
            c_atom = bond.GetEndAtom()
            a_idx = a_atom.GetIdx()
            c_idx = c_atom.GetIdx()
            # Find neighbors of a_atom excluding c_atom
            for neighbor in a_atom.GetNeighbors():
                if neighbor.GetIdx() != c_idx:
                    b_idx = neighbor.GetIdx()
                    triplet_groups.append([b_idx, a_idx, c_idx])
            # Find neighbors of c_atom excluding a_atom
            for neighbor in c_atom.GetNeighbors():
                if neighbor.GetIdx() != a_idx:
                    b_idx = neighbor.GetIdx()
                    triplet_groups.append([b_idx, c_idx, a_idx])
        return triplet_groups

class QuadrupletGroup:
    """
    Generate A-B-C-D quadruplet atom groups from a molecule.
    1st atom -- center atom1 -- center atom2 -- 4th atom
    """

    @staticmethod
    def generate(mol):
        """
        Generate quadruplet groups from the given molecule.
        :param mol: RDKit molecule object.
        :return: List of quadruplet groups, each represented as a tuple of atom indices (a_idx, b_idx, c_idx, d_idx).
        """
        quadruplet_groups = []
        for bond in mol.GetBonds():
            b_atom = bond.GetBeginAtom()
            c_atom = bond.GetEndAtom()
            b_idx = b_atom.GetIdx()
            c_idx = c_atom.GetIdx()
            # Find neighbors of b_atom excluding c_atom
            for a_neighbor in b_atom.GetNeighbors():
                if a_neighbor.GetIdx() != c_idx:
                    a_idx = a_neighbor.GetIdx()
                    # Find neighbors of c_atom excluding b_atom
                    for d_neighbor in c_atom.GetNeighbors():
                        if d_neighbor.GetIdx() != b_idx:
                            d_idx = d_neighbor.GetIdx()
                            quadruplet_groups.append([a_idx, b_idx, c_idx, d_idx])
        return quadruplet_groups