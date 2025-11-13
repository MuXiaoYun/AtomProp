"""
Library module for processing SMILES and sdf files to create datasets and dataloaders.
"""
import torch
import torch.nn as nn
import rdkit.Chem as Chem
from atomprop.embeddings.AtomEmbedding import BondTypes
from torch_geometric.data import Data, Batch

class SMILESToInputs:
    """
    A utility class to convert SMILES strings to atom type indices and edges.
    """
    @staticmethod
    def convert(smiles: str, context_length: int = 420, edge_output_type = 'edge_list', padding = False, sanitize = True):
        """
        Convert a SMILES string to atom type indices and edges.
        :param smiles: The SMILES string to convert.
        :param context_length: The maximum number of atoms in a molecule for padding.
        :param edge_output_type: The type of edge representation to return ('adj_matrix' or 'edge_list').
        :return: A tuple containing the atom indices and edge (adj matrix, value is bond type index, -1 for no edge), and the RDKit molecule object.
        """
        mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
        if mol is None:
            print(f"[SMILES TO INPUTS] Invalid SMILES string: {smiles}")
            return None, None, None
        
        mol = Chem.RemoveHs(mol=mol, sanitize=sanitize)  # Remove all H atoms
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

def smiles_to_pyg_data(smiles, max_atom_num=None):
    atom_indices, edges, mol = SMILESToInputs.convert(
        smiles=smiles,
        context_length=max_atom_num
    )

    if mol is None:
        return None
    
    num_atoms = len(mol.GetAtoms())
    x = atom_indices[:num_atoms]
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    
    if edges.dim() == 2 and edges.size(1) == 3:
        edge_index = edges[:, :2].t().contiguous()
        edge_attr = edges[:, 2].unsqueeze(-1)
    else:
        edge_index = edges
        edge_attr = torch.ones(edges.size(1), 1) if edges.dim() == 2 else torch.ones(1, 1)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles, mol=mol)

class PyGChunkDataListLoader:
    def __init__(self, data_path, split_indices, chunk_size=65536, max_atom_num=128, batch_size=32, device=None, file_type='csv'):
        self.data_path = data_path
        self.split_indices = split_indices
        self.chunk_size = chunk_size
        self.max_atom_num = max_atom_num
        self.batch_size = batch_size
        self.current_chunk_idx = 0
        self.current_chunk_data = None
        self.current_chunk_start = 0
        self.device = device
        self.file_type = file_type
        
        if self.file_type == 'csv':
            self.headers = pd.read_csv(data_path, nrows=0).columns.tolist()
        else:
            self.headers = ['SMILES']
            
        self.sorted_indices = np.sort(split_indices)
        self.total_batches = len(self.sorted_indices) // self.batch_size
        if len(self.sorted_indices) % self.batch_size != 0:
            self.total_batches += 1

    def __iter__(self):
        self.current_chunk_idx = 0
        self.current_chunk_data = None
        return self

    def __next__(self):
        data_list = []
        mols_list = []

        while len(data_list) < self.batch_size:
            if self.current_chunk_idx >= len(self.sorted_indices):
                if len(data_list) > 0:
                    return data_list, mols_list
                else:
                    raise StopIteration

            target_idx = self.sorted_indices[self.current_chunk_idx]
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

            data = smiles_to_pyg_data(smiles, self.max_atom_num)

            if data is None:
                print(f"Invalid SMILES at index {target_idx}: {smiles}")
                self.current_chunk_idx += 1
                continue

            if self.device is not None:
                data = data.to(self.device)

            data_list.append(data)
            mols_list.append(data.mol)
            self.current_chunk_idx += 1

        return data_list, mols_list

class XYZChunckDataLoader:
    def __init__(self, data_path, split_indices, chunk_size=65536, max_atom_num=128, batch_size=32, device=None, file_type='txt'):
        self.data_path = data_path
        self.split_indices = split_indices
        self.chunk_size = chunk_size
        self.max_atom_num = max_atom_num
        self.batch_size = batch_size
        self.current_chunk_idx = 0
        self.current_chunk_data = None
        self.current_chunk_start = 0
        self.device = device
        self.file_type = file_type
        
        if self.file_type == 'txt':
            self.headers = pd.read_csv(data_path, nrows=0).columns.tolist()
        else:
            pass
            
        self.sorted_indices = np.sort(split_indices)
        self.total_batches = len(self.sorted_indices) // self.batch_size
        if len(self.sorted_indices) % self.batch_size != 0:
            self.total_batches += 1

    def __iter__(self):
        self.current_chunk_idx = 0
        self.current_chunk_data = None
        return self

    def __next__(self):
        data_list = []
        mols_list = []

        while len(data_list) < self.batch_size:
            if self.current_chunk_idx >= len(self.sorted_indices):
                if len(data_list) > 0:
                    return data_list, mols_list
                else:
                    raise StopIteration

            target_idx = self.sorted_indices[self.current_chunk_idx]
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

            data = smiles_to_pyg_data(smiles, self.max_atom_num)

            if data is None:
                print(f"Invalid SMILES at index {target_idx}: {smiles}")
                self.current_chunk_idx += 1
                continue

            if self.device is not None:
                data = data.to(self.device)

            data_list.append(data)
            mols_list.append(data.mol)
            self.current_chunk_idx += 1

        return data_list, mols_list