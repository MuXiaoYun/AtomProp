"""
Training script for DFT total energy prediction using GeATNet.
"""

from atomprop.models.GeAT import GeATNet
from atomprop.dataloader.dataloader import SMILESToInputs
from atomprop.embeddings.AtomEmbedding import BondTypes
import configs.config as config
import rdkit.Chem as Chem
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar

data_path = "data/nabladft/summary.csv"

max_label = -766.9929885  # Maximum DFT total energy for normalization
min_label = -6060.656994  # Minimum DFT total energy for normalization 
max_atom_num = config.context_length  # Maximum number of atoms in a molecule for padding
dataset_size = -1  # Use only a subset of the data for quick testing
chunk_size = 65536  # Smaller chunk size for memory efficiency during training
batch_size = config.batch_size  # Batch size loaded from config, used for batching samples


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
    # Calculate total number of data rows (subtract 1 to exclude header)
    total_rows = sum(1 for _ in open(data_path)) - 1
    
    # Read first 10 rows to get column names and sample data structure
    sample_chunk = pd.read_csv(data_path, nrows=10)
    
    return total_rows, sample_chunk.columns.tolist()


def process_smiles_chunk(chunk_data, max_atom_num):
    """
    Process a chunk of SMILES data and yield valid molecules one by one
    """
    smiles_list = chunk_data['SMILES'].tolist()
    target_label_list = chunk_data['DFT TOTAL ENERGY'].tolist()
    
    for i, (smiles, target_label) in enumerate(zip(smiles_list, target_label_list)):
        # Convert SMILES string to GeATNet input format (atom embeddings + edges)
        atom_indices, edges, mol = SMILESToInputs.convert(smiles=smiles, context_length=max_atom_num)
        if mol is not None:  # Only yield if SMILES conversion is successful
            yield (atom_indices, edges, target_label)
        else:
            print(f"Invalid SMILES string: {smiles}")
            continue


def create_data_splits(total_size):
    """
    Create indices for train/validation/test splits with random permutation
    """
    # Shuffle all data indices to ensure random split
    indices = np.random.permutation(total_size)
    train_size = int(0.85 * total_size)  # 85% for training
    val_size = int(0.10 * total_size)    # 10% for validation
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]  # Remaining 5% for testing
    
    return train_indices, val_indices, test_indices


class ChunkDataLoader:
    """
    Custom data loader that processes data in chunks on-the-fly and supports batching
    """
    def __init__(self, data_path, split_indices, chunk_size=65536, max_atom_num=128, batch_size=32):
        self.data_path = data_path
        self.split_indices = split_indices
        self.chunk_size = chunk_size
        self.max_atom_num = max_atom_num
        self.batch_size = batch_size  # Add batch size parameter to control batch generation
        self.current_chunk_idx = 0    # Track current position in sorted split indices
        self.current_chunk_data = None# Store loaded chunk data temporarily
        self.current_chunk_start = 0  # Track start index of the currently loaded chunk
        # Load actual column names from CSV header (avoids mismatch with multi-column files)
        self.headers = pd.read_csv(data_path, nrows=0).columns.tolist()
        
        # Sort indices to enable efficient chunk-based loading (avoid repeated chunk reloads)
        self.sorted_indices = np.sort(split_indices)
        
        # Calculate total number of batches (for progress bar)
        self.total_batches = len(self.sorted_indices) // self.batch_size
        if len(self.sorted_indices) % self.batch_size != 0:
            self.total_batches += 1

    def __iter__(self):
        """Reset state when starting a new iteration (e.g., new epoch)"""
        self.current_chunk_idx = 0
        self.current_chunk_data = None
        return self

    def __next__(self):
        """Collect and return a batch of valid samples; raise StopIteration when all data is processed"""
        # Initialize lists to collect batch data (atom embeddings, edges, labels)
        batch_atom_emb = []
        batch_edges = []
        batch_labels = []

        # Collect samples until batch size is reached or all indices are processed
        while len(batch_atom_emb) < self.batch_size:
            # Stop iteration if all split indices have been processed
            if self.current_chunk_idx >= len(self.sorted_indices):
                # Return partial batch if there are remaining samples (avoid discarding data)
                if len(batch_atom_emb) > 0:
                    return self._collate_batch(batch_atom_emb, batch_edges, batch_labels)
                else:
                    raise StopIteration

            # Get the target data index to load next
            target_idx = self.sorted_indices[self.current_chunk_idx]
            # Calculate which chunk the target index belongs to
            chunk_num = target_idx // self.chunk_size
            chunk_start = chunk_num * self.chunk_size

            # Load new chunk if current chunk is unloaded or target index is in a different chunk
            if self.current_chunk_data is None or chunk_start != self.current_chunk_start:
                self.current_chunk_data = pd.read_csv(
                    self.data_path,
                    skiprows=chunk_start + 1,  # Skip header row (+1) and rows before chunk start
                    nrows=self.chunk_size,     # Load only one chunk of data
                    header=None,               # Disable default header parsing (we use custom names)
                    names=self.headers,        # Use actual CSV headers loaded in __init__
                    usecols=['SMILES', 'DFT TOTAL ENERGY']  # Load only required columns
                )
                self.current_chunk_start = chunk_start  # Update current chunk's start index

            # Calculate local index within the currently loaded chunk
            local_idx = target_idx % self.chunk_size
            # Extract SMILES and target label from the current chunk
            smiles = self.current_chunk_data.iloc[local_idx]['SMILES']
            target_label = self.current_chunk_data.iloc[local_idx]['DFT TOTAL ENERGY']

            # Convert SMILES to GeATNet input format
            atom_indices, edges, mol = SMILESToInputs.convert(
                smiles=smiles,
                context_length=self.max_atom_num
            )

            # Skip invalid SMILES and move to next index
            if mol is None:
                print(f"Invalid SMILES at index {target_idx}: {smiles}")
                self.current_chunk_idx += 1
                continue

            # Add valid sample to batch collections
            batch_atom_emb.append(atom_indices)
            batch_edges.append(edges)
            batch_labels.append(target_label)
            # Move to next index in split indices
            self.current_chunk_idx += 1

        # Collate collected samples into a batch of tensors and return
        return self._collate_batch(batch_atom_emb, batch_edges, batch_labels)

    def _collate_batch(self, batch_atom_emb, batch_edges, batch_labels):
        """
        Helper method to collate list of samples into a batch of PyTorch tensors
        (handles consistent tensor shaping for model input)
        """
        # Convert lists to tensors (assumes all samples have same shape due to max_atom_num padding)
        atom_emb_tensor = torch.stack(batch_atom_emb, dim=0)  # Shape: [batch_size, max_atom_num, atom_emb_dim]
        edges_tensor = torch.stack(batch_edges, dim=0)        # Shape: [batch_size, max_atom_num, max_atom_num, bond_emb_dim]
        labels_tensor = torch.tensor(batch_labels, dtype=torch.float32)  # Shape: [batch_size]

        return atom_emb_tensor, edges_tensor, labels_tensor


if __name__ == "__main__":
    # Get basic dataset information (total rows + column names)
    total_rows, columns = get_dataset_info(data_path)
    print(f"Total rows in dataset: {total_rows}")
    print(f"Dataset columns: {columns}")
    
    # Use subset of data if dataset_size is specified (for quick testing)
    if dataset_size > 0:
        total_rows = min(total_rows, dataset_size)
    
    # Create train/validation/test index splits
    train_indices, val_indices, test_indices = create_data_splits(total_rows)
    print(f"Train set size: {len(train_indices)}, Val set size: {len(val_indices)}, Test set size: {len(test_indices)}")
    
    # Define bond types supported by the model (matches GeATNet's input requirements)
    BondTypes.set_bond_types(["SINGLE", "DOUBLE", "TRIPLE", 'AROMATIC'])

    # Initialize GeATNet model with parameters from config
    geatnet = GeATNet(
        atom_embedding_dim=config.atom_embedding_dim,
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

    # Select computing device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using computing device: {device}")
    geatnet.to(device)  # Move model to the selected device

    # Initialize optimizer (Adam with weight decay for regularization)
    optimizer = torch.optim.Adam(geatnet.parameters(), lr=1e-4, weight_decay=1e-5)
    # Initialize learning rate scheduler (reduce LR when validation loss plateaus)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=3)
    # Initialize loss function (MSE for regression task of energy prediction)
    criterion = nn.MSELoss()
    
    # Create data loaders with batching support (use batch_size from config)
    train_loader = ChunkDataLoader(
        data_path=data_path,
        split_indices=train_indices,
        chunk_size=chunk_size,
        max_atom_num=max_atom_num,
        batch_size=batch_size  # Pass batch size to enable batching
    )
    val_loader = ChunkDataLoader(
        data_path=data_path,
        split_indices=val_indices,
        chunk_size=chunk_size,
        max_atom_num=max_atom_num,
        batch_size=batch_size  # Consistent batch size for validation
    )
    
    # Training loop configuration
    num_epochs = 100
    best_val_loss = float('inf')  # Track best validation loss for model saving
    train_losses = []  # Store training loss per epoch for plotting
    val_losses = []    # Store validation loss per epoch for plotting
    
    for epoch in range(num_epochs):
        try:
            # -------------------------- Training Phase --------------------------
            geatnet.train()  # Set model to training mode (enable dropout)
            total_train_loss = 0.0
            train_sample_count = 0  # Count total samples processed
            
            # Create progress bar for training batches
            train_pbar = tqdm(enumerate(train_loader), 
                             total=train_loader.total_batches, 
                             desc=f"Epoch {epoch+1}/{num_epochs} - Training")
            
            # Iterate over batches from train loader with progress bar
            for batch_idx, (atom_indices, edges, target_labels) in train_pbar:
                # Move batch data to computing device
                atom_indices = atom_indices.to(device)
                edges = edges.to(device)
                # Normalize target labels (scale to [0,1] range)
                target_labels = (target_labels - min_label) / (max_label - min_label)
                target_labels = target_labels.to(device)
                
                # Reset gradients from previous iteration
                optimizer.zero_grad()
                # Forward pass: model predicts energy from atom embeddings and edges
                outputs = geatnet(atom_indices, edges)
                # Calculate loss between predictions and normalized labels
                loss = criterion(outputs.squeeze(), target_labels)
                # Backward pass: compute gradients
                loss.backward()
                # Update model parameters using optimizer
                optimizer.step()
                
                # Accumulate total training loss and sample count
                batch_size_current = atom_indices.size(0)
                total_train_loss += loss.item() * batch_size_current
                train_sample_count += batch_size_current
                
                # Update progress bar with current batch loss
                train_pbar.set_postfix({"Batch Loss": f"{loss.item():.6f}"})
            
            # Calculate average training loss for the epoch
            avg_train_loss = total_train_loss / train_sample_count
            train_losses.append(avg_train_loss)

            # -------------------------- Validation Phase --------------------------
            geatnet.eval()  # Set model to evaluation mode (disable dropout)
            total_val_loss = 0.0
            val_sample_count = 0  # Count total validation samples processed
            
            # Create progress bar for validation batches
            val_pbar = tqdm(enumerate(val_loader), 
                           total=val_loader.total_batches, 
                           desc=f"Epoch {epoch+1}/{num_epochs} - Validation")
            
            # Disable gradient computation for validation (save memory and speed up)
            with torch.no_grad():
                # Iterate over batches from validation loader with progress bar
                for batch_idx, (atom_indices, edges, target_labels) in val_pbar:
                    # Move batch data to computing device
                    atom_indices = atom_indices.to(device)
                    edges = edges.to(device)
                    # Normalize target labels (same scaling as training)
                    target_labels = (target_labels - min_label) / (max_label - min_label)
                    target_labels = target_labels.to(device)
                    
                    # Forward pass: model prediction
                    outputs = geatnet(atom_indices, edges)
                    # Calculate validation loss
                    loss = criterion(outputs.squeeze(), target_labels)
                    
                    # Accumulate total validation loss and sample count
                    batch_size_current = atom_indices.size(0)
                    total_val_loss += loss.item() * batch_size_current
                    val_sample_count += batch_size_current
                    
                    # Update progress bar with current batch loss
                    val_pbar.set_postfix({"Batch Loss": f"{loss.item():.6f}"})
            
            # Calculate average validation loss for the epoch
            avg_val_loss = total_val_loss / val_sample_count
            val_losses.append(avg_val_loss)

            # Update learning rate based on validation loss (plateau detection)
            scheduler.step(avg_val_loss)

            # Print epoch summary
            print(f"Epoch {epoch+1}/{num_epochs} | Average Train Loss: {avg_train_loss:.6f} | Average Val Loss: {avg_val_loss:.6f}")

            # Save model if current validation loss is the best so far
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(geatnet.state_dict(), "trained_models/best_geatnet_model.pth")
                print(f"Epoch {epoch+1}: Saved best model (Best Val Loss: {best_val_loss:.6f})")

            # Save model checkpoint every 5 epochs (for resuming training)
            if (epoch + 1) % 5 == 0:
                torch.save(geatnet.state_dict(), f"trained_models/geatnet_model_epoch_{epoch+1}.pth")
                print(f"Epoch {epoch+1}: Saved model checkpoint")

        # Handle manual training interruption (e.g., Ctrl+C)
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving current model state...")
            torch.save(geatnet.state_dict(), "trained_models/interrupted_geatnet_model.pth")
            print("Interrupted model saved successfully.")
            break
    
    # -------------------------- Post-Training Visualization --------------------------
    # Plot training and validation loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', linewidth=2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.yscale('log')  # Log scale to better visualize loss reduction
    plt.title('GeATNet Training & Validation Loss Curves', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')  # Save plot with high resolution
    plt.show()
