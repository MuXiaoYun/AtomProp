"""
Library module for processing SMILES and sdf files to create datasets and dataloaders.
"""
import torch
import torch.nn as nn
import rdkit.Chem as Chem
from atomprop.embeddings.AtomEmbedding import BondTypes, BondDirections, AtomChirals
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
    
class xyzBatchLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.current_position = 0
        self.total_molecules = 0
        self.file_handle = None
        self._count_total_molecules()
        self._open_file()  # Open file handle on initialization
    
    def _open_file(self):
        """Open file handle and keep it open"""
        if self.file_handle is None or self.file_handle.closed:
            self.file_handle = open(self.data_path, 'r')
    
    def _count_total_molecules(self):
        """Count total number of molecules in file"""
        count = 0
        with open(self.data_path, 'r') as f:
            for line in f:
                if line.strip() == '':
                    count += 1
            if count > 0 or self._has_any_molecules():
                count += 1
        self.total_molecules = count
    
    def _has_any_molecules(self):
        """Check if file contains any molecules"""
        with open(self.data_path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('ERROR'):
                    return True
        return False
    
    def _read_next_molecule(self):
        """Read next molecule data from the open file stream"""
        if self.file_handle is None or self.file_handle.closed:
            self._open_file()
        
        lines = []
        molecule_data = None
        
        while True:
            line = self.file_handle.readline()
            # End of file
            if not line:
                break
            line = line.strip()
            # Empty line indicates end of molecule
            if not line and lines:
                break
            if line:  # Non-empty line
                lines.append(line)
        
        # Process collected lines
        if lines:
            # Check if it's an ERROR molecule
            if lines[0].startswith('ERROR'):
                error_str = lines[0]
                num_atoms = int(error_str.split('(')[1].split(')')[0])
                molecule_data = {
                    'type': 'error',
                    'num_atoms': num_atoms,
                    'coords': None
                }
            else:
                # Parse normal molecule coordinates
                coords = []
                for line in lines:
                    try:
                        x, y, z = map(float, line.split())
                        coords.append([x, y, z])
                    except ValueError:
                        continue  # Skip malformed lines
                
                molecule_data = {
                    'type': 'normal',
                    'num_atoms': len(coords),
                    'coords': coords
                }
        
        return molecule_data
    
    def reset(self):
        """Reset loader to start from beginning"""
        self.current_position = 0
        if self.file_handle and not self.file_handle.closed:
            self.file_handle.seek(0)  # Reset file pointer to beginning
    
    def __iter__(self):
        self.reset()
        return self
    
    def __next__(self):
        """Get next batch using iterator protocol"""
        if not hasattr(self, 'batch_size'):
            raise ValueError("batch_size must be set before iteration")
        return self.get_batch(self.batch_size)

    def download_head(self, batch_size, file_path):
        """Copy the lines in xyz file of the first batch_size molecules to a new file"""
        if self.file_handle is None or self.file_handle.closed:
            self._open_file()
        
        output_lines = []
        molecules_copied = 0
        
        while molecules_copied < batch_size:
            lines = []
            while True:
                line = self.file_handle.readline()
                # End of file
                if not line:
                    break
                line = line.strip()
                # Empty line indicates end of molecule
                if not line and lines:
                    break
                if line:  # Non-empty line
                    lines.append(line)
            
            if lines:
                output_lines.extend(lines)
                output_lines.append('')  # Add empty line between molecules
                molecules_copied += 1
            else:
                break  # No more molecules
        
        with open(file_path, 'w') as out_file:
            out_file.write('\n'.join(output_lines).strip() + '\n')
    
    def get_batch(self, batch_size):
        """Get next batch of molecules from current position"""
        if self.file_handle is None or self.file_handle.closed:
            self._open_file()
        
        molecules = []
        total_atoms = 0
        
        # Read batch_size molecules
        for _ in range(batch_size):
            molecule = self._read_next_molecule()
            if molecule is None:  # End of file
                break
                
            molecules.append(molecule)
            total_atoms += molecule['num_atoms']
            self.current_position += 1
        
        # Return empty tensor if no molecules found
        if not molecules:
            self.close()  # Close file when done
            raise StopIteration
        
        # Create result tensor filled with NaN
        batch_tensor = torch.full((total_atoms, 3), float('nan'))
        
        # Fill with actual coordinates for normal molecules
        current_pos = 0
        for mol in molecules:
            if mol['type'] == 'normal' and mol['coords'] is not None:
                coords_tensor = torch.tensor(mol['coords'], dtype=torch.float32)
                batch_tensor[current_pos:current_pos + mol['num_atoms']] = coords_tensor
            current_pos += mol['num_atoms']
        
        return batch_tensor
    
    def close(self):
        """Close the file handle"""
        if self.file_handle and not self.file_handle.closed:
            self.file_handle.close()
    
    def __del__(self):
        """Destructor to ensure file is closed"""
        self.close()
    
    def __len__(self):
        """Return total number of molecules"""
        return self.total_molecules
    
    def get_current_position(self):
        """Get current reading position"""
        return self.current_position
    
    def set_position(self, position):
        """Set current reading position (not efficient for large jumps)"""
        if position < 0 or position > self.total_molecules:
            raise ValueError(f"Position must be between 0 and {self.total_molecules}")
        
        # For large backward jumps, it's better to reset and skip
        if position < self.current_position:
            self.reset()
        
        # Skip to desired position
        while self.current_position < position:
            molecule = self._read_next_molecule()
            if molecule is None:
                break
            self.current_position += 1
    
    def has_next(self):
        """Check if there are more molecules to read"""
        return self.current_position < self.total_molecules

class xyzBatchLoaderContext:
    """
    Context manager for xyzBatchLoader to ensure proper resource management.
    """
    def __init__(self, data_path):
        self.data_path = data_path
        self.loader = None
    
    def __enter__(self):
        self.loader = xyzBatchLoader(self.data_path)
        return self.loader
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.loader:
            self.loader.close()
            