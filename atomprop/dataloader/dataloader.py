"""
Library module for processing SMILES and sdf files to create datasets and dataloaders.
"""
import torch
import torch.nn as nn
import rdkit.Chem as Chem
from atomprop.embeddings.AtomEmbedding import BondTypes

class SMILESToInputs:
    """
    A utility class to convert SMILES strings to atom type indices and edges.
    """
    @staticmethod
    def convert(smiles: str, context_length: int = 420, edge_output_type = 'edge_list', padding = False):
        """
        Convert a SMILES string to atom type indices and edges.
        :param smiles: The SMILES string to convert.
        :param context_length: The maximum number of atoms in a molecule for padding.
        :param edge_output_type: The type of edge representation to return ('adj_matrix' or 'edge_list').
        :return: A tuple containing the atom indices and edge (adj matrix, value is bond type index, -1 for no edge), and the RDKit molecule object.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        
        mol = Chem.RemoveHs(mol)  # Remove all H atoms
        # Get atom type indices
        atom_type_indices = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        atom_type_indices = torch.tensor(atom_type_indices, dtype=torch.long)

        if padding:
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
            bond_type_index = BondTypes.get_bond_types().index(str(bond.GetBondType()))
            edges.append((start_idx, end_idx, bond_type_index))
        
        if edge_output_type == 'adj_matrix':
            # Create adjacency matrix
            adj_matrix = torch.zeros((context_length, context_length), dtype=torch.long) - 1
            for start_idx, end_idx, bond_type_index in edges:
                adj_matrix[start_idx, end_idx] = bond_type_index
                adj_matrix[end_idx, start_idx] = bond_type_index
            if padding:
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
        elif edge_output_type == 'edge_list':
            if padding:
                # Pad atom type indices to context length with zeros
                if len(atom_type_indices) < context_length:
                    atom_type_indices = torch.cat([atom_type_indices, torch.zeros(context_length - len(atom_type_indices), dtype=torch.long)])
                else:
                    atom_type_indices = atom_type_indices[:context_length]
            return atom_type_indices, torch.tensor(edges, dtype=torch.long), mol
        else:
            raise ValueError(f"Invalid edge_output_type: {edge_output_type}. Must be 'adj_matrix' or 'edge_list'.")

class SDFToInputs:
    """
    A utility class to convert SDF files to atom type indices and edges.
    """
    @staticmethod
    def convert(sdf_path: str, context_length: int = 420, edge_output_type = 'edge_list', padding = False):
        """
        Convert an SDF file to atom type indices and edges.
        :param sdf_path: The path to the SDF file.
        :param context_length: The maximum number of atoms in a molecule for padding.
        :param edge_output_type: The type of edge representation to return ('adj_matrix' or 'edge_list').
        :return: A list of tuples, each containing the atom indices and edge (adj matrix, value is bond type index, -1 for no edge), and the RDKit molecule object.
        """
        suppl = Chem.SDMolSupplier(sdf_path, sanitize=True)
        if suppl is None:
            raise ValueError(f"Invalid SDF file: {sdf_path}")
        
        results = []
        for mol in suppl:
            if mol is None:
                continue
            mol = Chem.RemoveHs(mol)  # Remove all H atoms
            # Get atom type indices
            atom_type_indices = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            atom_type_indices = torch.tensor(atom_type_indices, dtype=torch.long)
            if padding:
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
                bond_type_index = BondTypes.get_bond_types().index(str(bond.GetBondType()))
                edges.append((start_idx, end_idx, bond_type_index))
            
            if edge_output_type == 'adj_matrix':
                # Create adjacency matrix
                adj_matrix = torch.zeros((context_length, context_length), dtype=torch.long) - 1
                for start_idx, end_idx, bond_type_index in edges:
                    adj_matrix[start_idx, end_idx] = bond_type_index
                    adj_matrix[end_idx, start_idx] = bond_type_index
                if padding:
                    # Pad adjacency matrix
                    if adj_matrix.size(0) < context_length:
                        adj_matrix = torch.cat([adj_matrix, torch.zeros(context_length - adj_matrix.size(0), context_length, dtype=torch.long) - 1], dim=0)
                        adj_matrix = torch.cat([adj_matrix, torch.zeros(context_length, context_length - adj_matrix.size(1), dtype=torch.long) - 1], dim=1)
                    else:
                        adj_matrix = adj_matrix[:context_length, :context_length]
                results.append((atom_type_indices, adj_matrix, mol))
            elif edge_output_type == 'edge_list':
                results.append((atom_type_indices, torch.tensor(edges, dtype=torch.long), mol))
            else:
                raise ValueError(f"Invalid edge_output_type: {edge_output_type}. Must be 'adj_matrix' or 'edge_list'.")
        return results

class MoleculeDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for molecules.
    :param molecules: A list of tuples, each containing the atom embeddings, edges (adj matrix), and label.
    :param min_label: The minimum label value for normalization (optional).
    :param max_label: The maximum label value for normalization (optional).
    """
    def __init__(self, molecules, min_label=None, max_label=None):
        self.molecules = molecules
        self.min_label = min_label
        self.max_label = max_label

    def __len__(self):
        return len(self.molecules)

    def __getitem__(self, idx):
        if self.min_label is not None and self.max_label is not None:
            label = (self.molecules[idx][2] - self.min_label) / (self.max_label - self.min_label)
            return self.molecules[idx][0], self.molecules[idx][1], label
        else:
            return self.molecules[idx][0], self.molecules[idx][1], self.molecules[idx][2]