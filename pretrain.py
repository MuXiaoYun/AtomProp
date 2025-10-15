from atomprop.tasks.tasks import NodeAttrPrediction
from atomprop.dataloader.dataloader import SMILESToInputs
from atomprop.models.GNNs import Embedder, GNN
from atomprop.utils.mlp import MLP
from atomprop.embeddings.AtomEmbedding import BondTypes
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data, Batch, DataLoader

backbone = Embedder(num_atom_types=120, embed_dim=128)
neck = GNN(num_layers=3, embed_dim=128, gnn_type='gcn', JK='last', dropout=0.1)
head = MLP(input_dim=128, hidden_dim=256, output_dim=15, num_layers=2, dropout=0.1, output_activation=None)
task = NodeAttrPrediction(criterion=torch.nn.CrossEntropyLoss())

data_path = "data/nabladft/summary.csv"
dataset_size = 100000
chunk_size = 65536
max_atom_num = 128
batch_size = 32

def get_dataset_info(data_path):
    """
    Get basic dataset information without loading all data
    """
    total_rows = sum(1 for _ in open(data_path)) - 1
    
    sample_chunk = pd.read_csv(data_path, nrows=10)
    
    return total_rows, sample_chunk.columns.tolist()

def create_data_splits(total_size):
    """
    Create indices for train/validation/test splits with random permutation
    """
    indices = np.random.permutation(total_size)
    train_size = int(0.85 * total_size)
    val_size = int(0.10 * total_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    return train_indices, val_indices, test_indices

def smiles_to_pyg_data(smiles, max_atom_num=None):
    """
    Convert SMILES to PyG Data object
    """
    atom_indices, edges, mol = SMILESToInputs.convert(
        smiles=smiles,
        context_length=max_atom_num
    )

    if mol is None:
        return None
    
    num_atoms = len(mol.GetAtoms())
    
    x = atom_indices[:num_atoms]
    
    return Data(x=x, edge_index=edges, smiles=smiles, mol=mol)


class PyGChunkDataLoader:
    """
    Custom data loader that processes data in chunks on-the-fly and yields PyG Data objects
    """
    def __init__(self, data_path, split_indices, chunk_size=65536, max_atom_num=128, batch_size=32):
        self.data_path = data_path
        self.split_indices = split_indices
        self.chunk_size = chunk_size
        self.max_atom_num = max_atom_num
        self.batch_size = batch_size
        self.current_chunk_idx = 0
        self.current_chunk_data = None
        self.current_chunk_start = 0
        self.headers = pd.read_csv(data_path, nrows=0).columns.tolist()
        
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
                    batch = Batch.from_data_list(data_list)
                    return batch, mols_list
                else:
                    raise StopIteration

            target_idx = self.sorted_indices[self.current_chunk_idx]
            chunk_num = target_idx // self.chunk_size
            chunk_start = chunk_num * self.chunk_size

            if self.current_chunk_data is None or chunk_start != self.current_chunk_start:
                self.current_chunk_data = pd.read_csv(
                    self.data_path,
                    skiprows=chunk_start + 1,
                    nrows=self.chunk_size,
                    header=None,
                    names=self.headers,
                    usecols=['SMILES']
                )
                self.current_chunk_start = chunk_start

            local_idx = target_idx % self.chunk_size
            smiles = self.current_chunk_data.iloc[local_idx]['SMILES']

            data = smiles_to_pyg_data(smiles, self.max_atom_num)

            if data is None:
                print(f"Invalid SMILES at index {target_idx}: {smiles}")
                self.current_chunk_idx += 1
                continue

            data_list.append(data)
            mols_list.append(data.mol)
            self.current_chunk_idx += 1

        batch = Batch.from_data_list(data_list)
        return batch, mols_list

if __name__ == "__main__":
    total_rows, columns = get_dataset_info(data_path)
    print(f"Total rows in dataset: {total_rows}")
    print(f"Dataset columns: {columns}")
    
    if dataset_size > 0:
        total_rows = min(total_rows, dataset_size)
    
    train_indices, val_indices, test_indices = create_data_splits(total_rows)
    print(f"Train set size: {len(train_indices)}, Val set size: {len(val_indices)}, Test set size: {len(test_indices)}")
    
    BondTypes.set_bond_types(["SINGLE", "DOUBLE", "TRIPLE", 'AROMATIC'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using computing device: {device}")
    backbone.to(device)
    neck.to(device)
    head.to(device)

    model_parameters = list(backbone.parameters()) + list(neck.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(model_parameters, lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3)
    criterion = nn.MSELoss()
    
    train_loader = PyGChunkDataLoader(
        data_path=data_path,
        split_indices=train_indices,
        chunk_size=chunk_size,
        max_atom_num=max_atom_num,
        batch_size=batch_size
    )
    val_loader = PyGChunkDataLoader(
        data_path=data_path,
        split_indices=val_indices,
        chunk_size=chunk_size,
        max_atom_num=max_atom_num,
        batch_size=batch_size
    )
    
    num_epochs = 100
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        try:
            backbone.train()
            neck.train()
            head.train()
            total_train_loss = 0.0
            train_sample_count = 0
            
            train_pbar = tqdm(enumerate(train_loader), 
                             total=train_loader.total_batches, 
                             desc=f"Epoch {epoch+1}/{num_epochs} - Training")
            
            for batch_idx, (batch_data, mols) in train_pbar:
                batch_data = batch_data.to(device)
                
                optimizer.zero_grad()
                
                atom_emb = backbone(batch_data.x, batch_data.edge_index, batch_data.edge_attr)
                graph_emb = neck(atom_emb, batch_data.edge_index, batch_data.edge_attr)
                outputs = head(graph_emb)
                
                task.set_pred(outputs)
                task.run_label(mols)
                loss = task.compute_loss()
                
                loss.backward()
                optimizer.step()
                
                if batch_idx == train_loader.total_batches - 1:
                    metrics = task.get_metrics()
                    print(f"Batch {batch_idx+1}/{train_loader.total_batches} Metrics: {metrics}")
                
                batch_size_current = len(mols)
                total_train_loss += loss.item() * batch_size_current
                train_sample_count += batch_size_current
                
                train_pbar.set_postfix({"Batch Loss": f"{loss.item():.6f}"})
            
            avg_train_loss = total_train_loss / train_sample_count
            train_losses.append(avg_train_loss)

            backbone.eval()
            neck.eval()
            head.eval()

            total_val_loss = 0.0
            val_sample_count = 0
            
            val_pbar = tqdm(enumerate(val_loader),
                                total=val_loader.total_batches, 
                                desc=f"Epoch {epoch+1}/{num_epochs} - Validation")
            
            with torch.no_grad():
                for batch_idx, (batch_data, mols) in val_pbar:
                    batch_data = batch_data.to(device)
                    
                    atom_emb = backbone(batch_data.x, batch_data.edge_index, batch_data.edge_attr)
                    graph_emb = neck(atom_emb, batch_data.edge_index, batch_data.edge_attr)
                    outputs = head(graph_emb)
                    
                    task.set_pred(outputs)
                    task.run_label(mols)
                    loss = task.compute_loss()
                    
                    if batch_idx == val_loader.total_batches - 1:
                        metrics = task.get_metrics()
                        print(f"Batch {batch_idx+1}/{val_loader.total_batches} Metrics: {metrics}")
                    
                    batch_size_current = len(mols)
                    total_val_loss += loss.item() * batch_size_current
                    val_sample_count += batch_size_current
                    
                    val_pbar.set_postfix({"Batch Loss": f"{loss.item():.6f}"})
            
            avg_val_loss = total_val_loss / val_sample_count
            val_losses.append(avg_val_loss)
            
            scheduler.step(avg_val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs} Summary: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'backbone_state_dict': backbone.state_dict(),
                    'neck_state_dict': neck.state_dict(),
                    'head_state_dict': head.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_val_loss': best_val_loss
                }, 'best_model.pth')
                print(f"Best model saved at epoch {epoch+1} with Val Loss = {best_val_loss:.6f}")

        except KeyboardInterrupt:
            torch.save({
                'backbone_state_dict': backbone.state_dict(),
                'neck_state_dict': neck.state_dict(),
                'head_state_dict': head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }, 'interrupted_model.pth')
            print("Training interrupted. Model state saved to 'interrupted_model.pth'.")
            break
            
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', linewidth=2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.yscale('log')
    plt.title('Training & Validation Loss Curves', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
    plt.show()