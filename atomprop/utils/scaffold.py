"""
Module for handling molecule scaffold.
"""

import torch
import numpy as np
from collections import defaultdict
from typing import List
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

class ScaffoldSimilarityMatrix:
    """Generate scaffold similarity matrix for a list of molecules.
    
    This class computes a binary matrix where element (i, j) = 1 if and 
    only if molecule i and molecule j share the same Murcko scaffold.
    
    Attributes:
        include_chirality (bool): Whether to include chirality in scaffold comparison.
        verbose (bool): Whether to print progress information.
    """
    
    def __init__(self, include_chirality: bool = False, verbose: bool = False):
        """
        Initialize the ScaffoldSimilarityMatrix.
        
        Args:
            include_chirality: If True, consider chirality when comparing scaffolds.
            verbose: If True, print progress information during computation.
        """
        self.include_chirality = include_chirality
        self.verbose = verbose
    
    def get_scaffold_smiles(self, mol: Chem.rdchem.Mol) -> str:
        """
        Extract Murcko scaffold SMILES from a molecule object.
        
        Args:
            mol: RDKit molecule object.
            
        Returns:
            SMILES string of the Murcko scaffold, or None if extraction fails.
        """
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold, isomericSmiles=self.include_chirality)
        except Exception:
            return None
    
    def compute_similarity_matrix(self, mol_list: List[Chem.rdchem.Mol]) -> torch.Tensor:
        """
        Compute scaffold similarity matrix for a list of molecules.
        
        This method creates an N x N binary matrix where N is the number of molecules.
        Element (i, j) = 1 if molecule i and molecule j share the same Murcko scaffold.
        
        Args:
            mol_list: List of RDKit molecule objects.
            
        Returns:
            torch.Tensor of shape [N, N] with dtype torch.float32.
            Diagonal elements are always 1 (a molecule shares scaffold with itself).
            Invalid molecules (None or scaffold extraction fails) are treated as unique.
            
        Raises:
            ValueError: If mol_list is empty.
        """
        if not mol_list:
            raise ValueError("mol_list cannot be empty")
        
        N = len(mol_list)
        
        # Step 1: Assign scaffold IDs to each molecule
        scaffold_to_id = {}
        molecule_scaffold_ids = np.zeros(N, dtype=int)
        
        if self.verbose:
            print(f"Processing {N} molecules...")
        
        for idx, mol in enumerate(mol_list):
            if mol is None:
                # Assign unique negative ID for invalid molecules
                molecule_scaffold_ids[idx] = -idx - 1
                if self.verbose:
                    print(f"Warning: None molecule at index {idx}")
                continue
            
            scaffold_smiles = self.get_scaffold_smiles(mol)
            
            if scaffold_smiles is None:
                # Assign unique negative ID for failed scaffold extraction
                molecule_scaffold_ids[idx] = -idx - 1
                if self.verbose:
                    print(f"Warning: Failed to extract scaffold at index {idx}")
            elif scaffold_smiles not in scaffold_to_id:
                # New scaffold found, assign new ID
                new_id = len(scaffold_to_id)
                scaffold_to_id[scaffold_smiles] = new_id
                molecule_scaffold_ids[idx] = new_id
            else:
                # Existing scaffold, use assigned ID
                molecule_scaffold_ids[idx] = scaffold_to_id[scaffold_smiles]
        
        if self.verbose:
            print(f"Found {len(scaffold_to_id)} unique scaffolds")
        
        # Step 2: Compute similarity matrix using broadcasting
        scaffold_ids = torch.tensor(molecule_scaffold_ids, dtype=torch.long).unsqueeze(1)  # [N, 1]
        
        # Compare scaffold IDs for all pairs using broadcasting
        # (scaffold_ids == scaffold_ids.T) creates an N x N boolean matrix
        similarity_matrix = (scaffold_ids == scaffold_ids.T).float()
        
        # Step 3: Handle invalid molecules
        # For molecules with negative scaffold IDs (invalid), set all comparisons to 0
        # except self-comparison (diagonal)
        invalid_indices = np.where(molecule_scaffold_ids < 0)[0]
        
        if len(invalid_indices) > 0:
            if self.verbose:
                print(f"Found {len(invalid_indices)} invalid molecules")
            
            # Create a mask for invalid molecules
            invalid_mask = torch.zeros(N, dtype=torch.bool)
            invalid_mask[invalid_indices] = True
            
            # Set rows and columns for invalid molecules to 0
            similarity_matrix[invalid_mask, :] = 0
            similarity_matrix[:, invalid_mask] = 0
            
            # Restore diagonal for invalid molecules (self-similarity = 1)
            for idx in invalid_indices:
                similarity_matrix[idx, idx] = 1
        
        return similarity_matrix
    
    def get_scaffold_groups(self, mol_list: List[Chem.rdchem.Mol]) -> List[List[int]]:
        """
        Group molecules by their scaffolds.
        
        Args:
            mol_list: List of RDKit molecule objects.
            
        Returns:
            List of lists, where each inner list contains indices of molecules
            sharing the same scaffold.
        """
        scaffolds = defaultdict(list)
        
        for idx, mol in enumerate(mol_list):
            if mol is None:
                continue
            
            scaffold_smiles = self.get_scaffold_smiles(mol)
            if scaffold_smiles is not None:
                scaffolds[scaffold_smiles].append(idx)
        
        # Return only non-empty groups
        return [indices for indices in scaffolds.values() if len(indices) > 1]
    