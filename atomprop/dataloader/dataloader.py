"""
Library module for processing SMILES and sdf files to create datasets and dataloaders.
"""
import torch
import torch.nn as nn
import rdkit.Chem as Chem
from atomprop.embeddings.atom_embedding import BondTypes, BondDirections, AtomChirals
from torch_geometric.data import Data, Batch
import numpy as np
import pandas as pd

class SMILESToInputs:
    """
    A utility class to convert SMILES strings to atom type indices and edges.
    """
    @staticmethod
    def convert(smiles: str, removehs = True, sanitize = True, skipUnkekulizable = False):
        """
        Convert a SMILES string to atom type indices and edges.
        :param smiles: The SMILES string to convert.
        :param sanitize: Decide whether to sanitize input molecules.
        :param skipUnkekulizable: Decide whether to abandon molecules which could not be removeHs().
        :return: A tuple containing the atom info, edge info, and the RDKit molecule object.
        """
        mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
        if mol is None:
            print(f"[SMILES TO INPUTS] Invalid SMILES string: {smiles}")
            return None, None, None
        if removehs:
            try:
                mol = Chem.RemoveHs(mol=mol, sanitize=sanitize)  # Remove all H atoms
            except:
                # This is usually because the molecule cannot be kekulized.
                # Just do not remove Hs in this case.
                print(f"[SMILES TO INPUTS] SMILES string {smiles} convert error: removeHs failed")
                if skipUnkekulizable:
                    return None, None, None
        # Get atom type indices
        atoms = [[atom.GetAtomicNum(), atom.GetChiralTag()] for atom in mol.GetAtoms()]

        # Get edges (bond type indices)
        edges = []
        for bond in mol.GetBonds():
            start_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            bond_type_index = BondTypes.get_index(str(bond.GetBondType()))
            bond_direction_index = BondDirections.get_index(str(bond.GetBondDir()))
            edges.append((start_idx, end_idx, bond_type_index, bond_direction_index))
        
        return torch.tensor(atoms, dtype=torch.long), torch.tensor(edges, dtype=torch.long), mol

def smiles_to_pyg_data(smiles):
    atom_info, edge_info, mol = SMILESToInputs.convert(
        smiles=smiles,
    )
    if mol is None:
        return None
    
    num_atoms = len(mol.GetAtoms())
    x = atom_info[:num_atoms]
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    
    if edge_info.dim() == 2 and edge_info.size(1) == 4:
        edge_index = edge_info[:, :2].t().contiguous()
        edge_attr = edge_info[:, 2:]
        
    else:
        edge_index = torch.tensor([[], []], dtype=torch.long)
        edge_attr = torch.tensor([], dtype=torch.long).view(0, 2)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles, mol=mol)

class PyGChunkDataListLoader:
    def __init__(self, data_path, split_indices, chunk_size=65536, batch_size=32,
                 device=None, file_type='csv', sampler=None):
        self.data_path = data_path
        self.split_indices = split_indices  # full list of global indices to consider
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.device = device
        self.file_type = file_type
        self.sampler = sampler

        if self.file_type == 'csv':
            self.headers = pd.read_csv(data_path, nrows=0).columns.tolist()
        else:
            self.headers = ['SMILES']

        self._base_indices = list(range(len(self.split_indices)))

        self._index_iter = None
        self.current_chunk_data = None
        self.current_chunk_start = -1
        
    def _get_effective_num_samples(self):
        if self.sampler is not None:
            return len(self.sampler)
        else:
            return len(self.split_indices)
        
    @property
    def total_batches(self):
        num_samples = self._get_effective_num_samples()
        return (num_samples + self.batch_size - 1) // self.batch_size
    
    def __len__(self):
        return self.total_batches

    def __iter__(self):
        self.current_chunk_data = None
        self.current_chunk_start = -1

        if self.sampler is not None:
            self._index_iter = iter(self.sampler)
        else:
            self._index_iter = iter(self._base_indices)

        return self

    def __next__(self):
        data_list = []
        mols_list = []

        while len(data_list) < self.batch_size:
            try:
                pos_in_split = next(self._index_iter)  # position in split_indices
            except StopIteration:
                if len(data_list) > 0:
                    return data_list, mols_list
                else:
                    raise

            target_idx = self.split_indices[pos_in_split]

            chunk_num = target_idx // self.chunk_size
            chunk_start = chunk_num * self.chunk_size

            if self.current_chunk_data is None or chunk_start != self.current_chunk_start:
                if self.file_type == 'csv':
                    self.current_chunk_data = pd.read_csv(
                        self.data_path,
                        skiprows=chunk_start + 1,
                        nrows=self.chunk_size,
                        header=None,
                        names=self.headers,
                        usecols=['SMILES']
                    )
                else:
                    self.current_chunk_data = pd.read_csv(
                        self.data_path,
                        skiprows=chunk_start,
                        nrows=self.chunk_size,
                        header=None,
                        names=self.headers
                    )
                self.current_chunk_start = chunk_start

            local_idx = target_idx % self.chunk_size
            smiles = self.current_chunk_data.iloc[local_idx]['SMILES']

            data = smiles_to_pyg_data(smiles)

            if data is None:
                raise ValueError(f"Invalid SMILES at global index {target_idx}: {smiles}")

            if self.device is not None:
                data = data.to(self.device)

            data_list.append(data)
            mols_list.append(data.mol)

        return data_list, mols_list