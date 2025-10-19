"""
Module for hand-made features for atoms and molecules.
"""

import rdkit
from rdkit import Chem
import torch
import numpy as np

class atomFeaturize:
    """
    A class to featurize atoms in a molecule.
    Used featrues include:
    - Atom type (one-hot encoding for atomic numbers 0-118, 0 for unknown or masked)
    - Degree (one-hot encoding for degrees 0-7, 7 for unknown)
    - Formal charge (one-hot encoding for charges -3 to +3, 4 for unknown)
    - Hybridization (one-hot encoding for S, SP, SP2, SP3, SP2D, SP3D, SP3D2, UnknownOrOther)
    - Chirality (one-hot encoding for ALLENE, OCTAHEDRAL, SQUAREPLANAR, TETRAHEDRAL, TETRAHEDRAL_CCW, TETRAHEDRAL_CW, TRIGONALBIPYRAMIDAL, UnknownOrOther)
    - Number of hydrogens (one-hot encoding for 0-5, 5 for unknown)
    - Mass (scaled by 0.01)
    """
    @staticmethod
    def atom_type_onehot(atom, allowable_set=range(0, 119), include_unknown=True):
        """
        One-hot encoding for atom types.
        :param atom: RDKit atom object.
        :param allowable_set: List of allowable atomic numbers.
        :param include_unknown: Whether to include an 'unknown' category.
        :return: One-hot encoded list for the atom type.
        """
        atom_type = atom.GetAtomicNum()
        if atom_type not in allowable_set:
            if include_unknown:
                atom_type = allowable_set[-1]
            else:
                raise ValueError(f"Atom type {atom_type} not in allowable set {allowable_set}")
        return [int(atom_type == s) for s in allowable_set]

    @staticmethod
    def atom_degree_onehot(atom, allowable_set=range(0, 8), include_unknown=True):
        """
        One-hot encoding for atom degree.
        :param atom: RDKit atom object.
        :param allowable_set: List of allowable degrees.
        :param include_unknown: Whether to include an 'unknown' category.
        :return: One-hot encoded list for the atom degree.
        """
        degree = atom.GetDegree()
        if degree not in allowable_set:
            if include_unknown:
                degree = allowable_set[-1]
            else:
                raise ValueError(f"Atom degree {degree} not in allowable set {allowable_set}")
        return [int(degree == s) for s in allowable_set]

    @staticmethod
    def atom_formal_charge_onehot(atom, allowable_set=range(-3, 4), include_unknown=True):
        """
        One-hot encoding for atom formal charge.
        :param atom: RDKit atom object.
        :param allowable_set: List of allowable formal charges.
        :param include_unknown: Whether to include an 'unknown' category.
        :return: One-hot encoded list for the atom formal charge.
        """
        formal_charge = atom.GetFormalCharge()
        if formal_charge not in allowable_set:
            if include_unknown:
                formal_charge = allowable_set[-1]
            else:
                raise ValueError(f"Atom formal charge {formal_charge} not in allowable set {allowable_set}")
        return [int(formal_charge == s) for s in allowable_set]

    @staticmethod
    def atom_hybridization_onehot(atom, allowable_set=(Chem.rdchem.HybridizationType.S,
                                                       Chem.rdchem.HybridizationType.SP,
                                                       Chem.rdchem.HybridizationType.SP2,
                                                       Chem.rdchem.HybridizationType.SP3,
                                                       Chem.rdchem.HybridizationType.SP2D,
                                                       Chem.rdchem.HybridizationType.SP3D,
                                                       Chem.rdchem.HybridizationType.SP3D2), include_unknown=True):
        """
        One-hot encoding for atom hybridization.
        :param atom: RDKit atom object.
        :param allowable_set: List of allowable hybridizations.
        :param include_unknown: Whether to include an 'unknown' category.
        :return: One-hot encoded list for the atom hybridization.
        """
        hybridization = atom.GetHybridization()
        if hybridization not in allowable_set:
            if include_unknown:
                return [0] * (len(allowable_set)) + [1]
            else:
                raise ValueError(f"Atom hybridization {hybridization} not in allowable set {allowable_set}")
        return [int(hybridization == s) for s in allowable_set] + [0]

    @staticmethod
    def atom_chirality_onehot(atom, allowable_set=(Chem.rdchem.ChiralType.CHI_ALLENE,
                                                    Chem.rdchem.ChiralType.CHI_OCTAHEDRAL,
                                                    Chem.rdchem.ChiralType.CHI_SQUAREPLANAR,
                                                    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL,
                                                    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                                                    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                                                    Chem.rdchem.ChiralType.CHI_TRIGONALBIPYRAMIDAL), include_unknown=True):
        """
        One-hot encoding for atom chirality.
        :param atom: RDKit atom object.
        :param allowable_set: List of allowable chiralities.
        :param include_unknown: Whether to include an 'unknown' category.
        :return: One-hot encoded list for the atom chirality.
        """
        chirality = atom.GetChiralTag()
        if chirality not in allowable_set:
            if include_unknown:
                return [0] * (len(allowable_set)) + [1]
            else:
                raise ValueError(f"Atom chirality {chirality} not in allowable set {allowable_set}")
        return [int(chirality == s) for s in allowable_set] + [0]

    @staticmethod
    def atom_num_hydrogens_onehot(atom, allowable_set=range(0, 6), include_unknown=True):
        """
        One-hot encoding for number of hydrogens on the atom.
        :param atom: RDKit atom object.
        :param allowable_set: List of allowable number of hydrogens.
        :param include_unknown: Whether to include an 'unknown' category.
        :return: One-hot encoded list for the number of hydrogens.
        """
        num_hydrogens = atom.GetTotalNumHs()
        if num_hydrogens not in allowable_set:
            if include_unknown:
                num_hydrogens = allowable_set[-1]
            else:
                raise ValueError(f"Number of hydrogens {num_hydrogens} not in allowable set {allowable_set}")
        return [int(num_hydrogens == s) for s in allowable_set]

    @staticmethod
    def atom_mass(atom):
        """
        Scaled atom mass.
        :param atom: RDKit atom object.
        :return: Scaled mass of the atom.
        """
        mass = atom.GetMass() * 0.01
        return [mass]

    @staticmethod
    def featurize(mol):
        """
        Featurize all atoms in a molecule.
        :param mol: RDKit molecule object.
        :return: A list of atom features for all atoms in the molecule.
        """
        atom_features = []
        for atom in mol.GetAtoms():
            features = []
            features += atomFeaturize.atom_type_onehot(atom)
            features += atomFeaturize.atom_degree_onehot(atom)
            features += atomFeaturize.atom_formal_charge_onehot(atom)
            features += atomFeaturize.atom_hybridization_onehot(atom)
            features += atomFeaturize.atom_chirality_onehot(atom)
            features += atomFeaturize.atom_num_hydrogens_onehot(atom)
            features += atomFeaturize.atom_mass(atom)
            atom_features.append(features)
        atom_features_np = torch.tensor(atom_features, dtype=torch.float32)
        return atom_features_np # dimension of each atom feature is 119+8+7+8+8+6+1=157