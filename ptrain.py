from atomprop.tasks.tasks import NodeAttrPrediction, MaskedNodePrediction, GraphMaskContrast, BatchContrast, BondAnglePrediction, DihedralAnglePrediction, FunctionalGroupsPrediction
from atomprop.dataloader.dataloader import SMILESToInputs, PyGChunkDataListLoader, xyzBatchLoader, xyzBatchLoaderContext
from atomprop.models.GNNs import Embedder, GNN, GNNAggr
from atomprop.utils.mlp import MLP
from atomprop.utils.mask import MolGraphMask
from atomprop.utils.groups import TripletGroup, QuadrupletGroup
from atomprop.utils.features import FunctionalGroupUtils
from atomprop.utils.weights import WeightStratergy, EqualWeightStratergy, HardSwitch, SoftSwitch, GradNorm, ParetoOpt
from atomprop.utils.scaffold import ScaffoldSimilarityMatrix
from atomprop.embeddings.AtomEmbedding import BondTypes
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data, Batch
from torch.utils.tensorboard import SummaryWriter
from rdkit.Chem import FunctionalGroups
from deepchem.splits.splitters import ScaffoldSplitter
from deepchem.data import NumpyDataset
import os

record_freq = 100
dataset_size = -1
num_epochs = 8
batch_size = 1024

# data_path = "data/zinc15/dataset/zinc_standard_agent/processed/smiles.csv"
data_path = "data/pubchem/pubchem-10m.txt"
pretrain_file_type = 'txt'

xyz_path = "data/pubchem/pubchem-xyzs.txt"
xyz_type = 'txt'

chunk_size = 65536
max_atom_num = 128

less_rate = 0.1
more_rate = 0.3
embed_dim = 384

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def get_dataset_info(data_path):
    total_rows = sum(1 for _ in open(data_path)) - 1
    sample_chunk = pd.read_csv(data_path, nrows=10)
    return total_rows, sample_chunk.columns.tolist()

def create_data_splits(total_size):
    indices = np.arange(total_size)
    train_size = int(0.85 * total_size)
    val_size = int(0.10 * total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    return train_indices, val_indices, test_indices

if __name__ == "__main__":
    with xyzBatchLoaderContext(xyz_path) as xyz_loader:
        total_rows, columns = get_dataset_info(data_path)
        print(f"Total rows in dataset: {total_rows}")

        if dataset_size > 0:
            total_rows = min(total_rows, dataset_size)

        train_indices, val_indices, test_indices = create_data_splits(total_rows)
        print(f"Train set size: {len(train_indices)}, Val set size: {len(val_indices)}, Test set size: {len(test_indices)}")

        BondTypes.set_bond_types(["SINGLE", "DOUBLE", "TRIPLE", 'AROMATIC'])

        print(f"Using computing device: {device}")
        
        train_loader = PyGChunkDataListLoader(
            data_path=data_path,
            split_indices=train_indices,
            chunk_size=chunk_size,
            batch_size=batch_size,
            file_type=pretrain_file_type
        )
        val_loader = PyGChunkDataListLoader(
            data_path=data_path,
            split_indices=val_indices,
            chunk_size=chunk_size,
            batch_size=batch_size,
            file_type=pretrain_file_type
        )
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        eps = 1e-8
        
        for epoch in range(num_epochs):
            try:
                xyz_loader.reset()

                total_train_loss = 0.0
                train_sample_count = 0
                
                train_pbar = tqdm(enumerate(train_loader), 
                                total=train_loader.total_batches, 
                                desc=f"Epoch {epoch+1}/{num_epochs} - Training")
                
                for batch_idx, (data_list, mols) in train_pbar:
                    # calc = ScaffoldSimilarityMatrix()
                    # mat = calc.compute_similarity_matrix(mols)
                    # print("shape: ", mat.shape)
                    # print("first 5: ", mat[:5,:5])
                    # print("total: ", torch.sum(mat))
                    pass
            except:
                pass