"""
Training script for dft total energy prediction using GeATNet.
"""

from atomprop.models.GeAT import GeATNet
from atomprop.dataloader.dataloader import SMILESToInputs
from atomprop.embeddings.AtomEmbedding import BondTypes
import configs.config as config
import rdkit.Chem as Chem
import rdkit.Chem.Descriptors as Descriptors
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data_path = "data/nabladft/summary.csv"

max_label = -766.9929885  # Maximum dft total energy for normalization
min_label = -6060.656994  # Minimum dft total energy for normalization 
max_atom_num = config.context_length  # Maximum number of atoms in a molecule for padding
dataset_size = 1000  # Use only a subset of the data for quick testing
chunk_size = 65536  # Smaller chunk size for memory efficiency during training

def print_grad_norm(module: nn.Module, prefix: str = ""):
    """
    Recursively print the L2-norm of every trainable parameter's gradient
    inside the given module.
    """
    for name, param in module.named_parameters(recurse=False):
        if param.grad is not None:
            grad_norm = param.grad.detach().norm(2).item()
            print(f"{prefix}.{name}: {grad_norm:.6f}")
    for child_name, child in module.named_children():
        child_prefix = f"{prefix}.{child_name}" if prefix else child_name
        print_grad_norm(child, prefix=child_prefix)

def get_dataset_info(data_path):
    """
    Get basic dataset information without loading all data
    """
    # Get total number of rows
    total_rows = sum(1 for _ in open(data_path)) - 1  # Subtract header
    
    # Read first chunk to get column names and sample data
    sample_chunk = pd.read_csv(data_path, nrows=10)
    
    return total_rows, sample_chunk.columns.tolist()

def process_smiles_chunk(chunk_data, max_atom_num):
    """
    Process a chunk of SMILES data and yield valid molecules one by one
    """
    smiles_list = chunk_data['SMILES'].tolist()
    target_label_list = chunk_data['DFT TOTAL ENERGY'].tolist()
    
    for i, (smiles, target_label) in enumerate(zip(smiles_list, target_label_list)):
        # Convert SMILES to GeAT inputs
        atom_embeddings, edges, mol = SMILESToInputs.convert(smiles=smiles, context_length=max_atom_num)
        if mol is not None:
            yield (atom_embeddings, edges, target_label)
        else:
            print(f"Invalid SMILES string: {smiles}")
            continue
            

def create_data_splits(total_size):
    """
    Create indices for train/val/test splits
    """
    indices = np.random.permutation(total_size)
    train_size = int(0.85 * total_size)
    val_size = int(0.10 * total_size)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    return train_indices, val_indices, test_indices

class ChunkDataLoader:
    """
    Custom data loader that processes data in chunks on-the-fly
    """
    def __init__(self, data_path, split_indices, chunk_size=1000, max_atom_num=128):
        self.data_path = data_path
        self.split_indices = split_indices
        self.chunk_size = chunk_size
        self.max_atom_num = max_atom_num
        self.current_chunk_idx = 0
        self.current_chunk_data = None
        
        # Sort indices for efficient chunk-based access
        self.sorted_indices = np.sort(split_indices)
        
    def __iter__(self):
        self.current_chunk_idx = 0
        self.current_chunk_data = None
        return self
    
    def __next__(self):
        if self.current_chunk_idx >= len(self.sorted_indices):
            raise StopIteration
        
        # Find which chunk contains the next index
        target_idx = self.sorted_indices[self.current_chunk_idx]
        chunk_num = target_idx // self.chunk_size
        chunk_start = chunk_num * self.chunk_size
        
        # Load chunk if not loaded or different chunk
        if self.current_chunk_data is None or chunk_start != self.current_chunk_start:
            self.current_chunk_data = pd.read_csv(
                self.data_path, 
                skiprows=chunk_start + 1,  # +1 for header
                nrows=self.chunk_size,
                header=None,
                names=['SMILES', 'DFT TOTAL ENERGY']
            )
            self.current_chunk_start = chunk_start
        
        # Get data from current chunk
        local_idx = target_idx % self.chunk_size
        smiles = self.current_chunk_data.iloc[local_idx]['SMILES']
        target_label = self.current_chunk_data.iloc[local_idx]['DFT TOTAL ENERGY']

        # Process SMILES
        atom_embeddings, edges, mol = SMILESToInputs.convert(
            smiles=smiles, 
            context_length=self.max_atom_num
        )
        
        if mol is None:
            print(f"Invalid SMILES at index {target_idx}: {smiles}")
            self.current_chunk_idx += 1
            return self.__next__()  # Skip to next valid sample
        
        self.current_chunk_idx += 1
        return atom_embeddings, edges, target_label

if __name__ == "__main__":
    # Get dataset information
    total_rows, columns = get_dataset_info(data_path)
    print(f"Total rows in dataset: {total_rows}")
    print(f"Columns: {columns}")
    
    if dataset_size > 0:
        total_rows = min(total_rows, dataset_size)
    
    # Create data splits
    train_indices, val_indices, test_indices = create_data_splits(total_rows)
    print(f"Train indices: {len(train_indices)}, Val indices: {len(val_indices)}, Test indices: {len(test_indices)}")
    
    BondTypes.set_bond_types(["SINGLE", "DOUBLE", "TRIPLE", 'AROMATIC'])

    # Create the model
    geatnet = GeATNet(atom_embedding_dim=config.atom_embedding_dim,
                     num_atom_types=config.num_atom_types,
                     num_bond_types=config.num_bond_types,
                     num_heads=config.num_heads,
                     global_num_heads=config.global_num_heads,
                     backbone_dropout=config.backbone_dropout,
                     neck_dropout=config.neck_dropout,
                     head_dropout=config.head_dropout,
                     mlp_hidden_dim=config.geatnet_hidden_dim,
                     output_negative_slope=config.edge_attetion_output_negative_slope,
                     parallel_between_bondtypes=config.parallel_between_bondtypes,
                     )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    geatnet.to(device)

    # Create the optimizer
    optimizer = torch.optim.Adam(geatnet.parameters(), lr=1e-4, weight_decay=1e-5)
    # Create the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3)
    # Create the loss function
    criterion = nn.MSELoss()
    
    # Create data loaders
    train_loader = ChunkDataLoader(data_path, train_indices, chunk_size, max_atom_num)
    val_loader = ChunkDataLoader(data_path, val_indices, chunk_size, max_atom_num)
    
    # Training loop
    num_epochs = 100
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        try:
            # Training phase
            geatnet.train()
            total_train_loss = 0.0
            train_samples = 0
            
            for atom_embeddings, edges, target_label in train_loader:
                atom_embeddings = atom_embeddings.to(device)
                edges = edges.to(device)
                target_label = torch.tensor(
                    [(target_label - min_label) / (max_label - min_label)], 
                    dtype=torch.float32
                ).to(device)
                
                optimizer.zero_grad()
                output = geatnet(atom_embeddings.unsqueeze(0), edges.unsqueeze(0))
                loss = criterion(output, target_label.unsqueeze(-1))
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                train_samples += 1
            
            avg_train_loss = total_train_loss / train_samples
            train_losses.append(avg_train_loss)

            # Validation phase
            geatnet.eval()
            total_val_loss = 0.0
            val_samples = 0
            
            with torch.no_grad():
                for atom_embeddings, edges, target_label in val_loader:
                    atom_embeddings = atom_embeddings.to(device)
                    edges = edges.to(device)
                    target_label = torch.tensor(
                        [(target_label - min_label) / (max_label - min_label)], 
                        dtype=torch.float32
                    ).to(device)
                    
                    output = geatnet(atom_embeddings.unsqueeze(0), edges.unsqueeze(0))
                    loss = criterion(output, target_label.unsqueeze(-1))
                    total_val_loss += loss.item()
                    val_samples += 1
            
            avg_val_loss = total_val_loss / val_samples
            val_losses.append(avg_val_loss)

            scheduler.step(avg_val_loss)

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

            # Save the model if the validation loss is the best we've seen so far.
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(geatnet.state_dict(), "trained_models/best_geatnet_model.pth")
                print(f"Epoch {epoch+1}: Saved the best model.")

            # Save every 5 epochs
            if (epoch + 1) % 5 == 0:
                torch.save(geatnet.state_dict(), f"trained_models/geatnet_model_epoch_{epoch+1}.pth")
                print(f"Saved model at epoch {epoch+1}.")

        except KeyboardInterrupt:
            print("Training interrupted. Saving the current model...")
            torch.save(geatnet.state_dict(), "trained_models/interrupted_geatnet_model.pth")
            break
    
    # Plot the training and validation loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig('loss_curve.png')
    plt.show()