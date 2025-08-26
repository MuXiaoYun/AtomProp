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

data_path = "data/nabladft/summary.csv"

max_label = -766.9929885  # Maximum dft total energy for normalization
min_label = -6060.656994  # Minimum dft total energy for normalization 
max_atom_num = config.context_length  # Maximum number of atoms in a molecule for padding

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

if __name__ == "__main__":
    # load the data from the csv file, each line is a SMILES string and its dft total energy
    import pandas as pd
    df = pd.read_csv(data_path)
    smiles_list = df['SMILES'].tolist()
    target_label_list = float(df['DFT TOTAL ENERGY'].tolist())
    print("Number of SMILES strings:", len(smiles_list))
    BondTypes.set_bond_types(["SINGLE", "DOUBLE", "TRIPLE", "ONEANDAHALF"])

    molecules = []
    for i, smiles in enumerate(smiles_list):
        if i % 25000 == 0:
            print(f"Processing {i}th SMILES: {smiles}")
        # Convert SMILES to GeAT inputs
        atom_embeddings, edges, mol = SMILESToInputs.convert(smiles=smiles, context_length=max_atom_num)
        if mol is not None:
            # Append the molecule and its mol mass with Hs
            molecules.append((atom_embeddings, edges, target_label_list[i]))
        else:
            print(f"Invalid SMILES string at index {i}: {smiles}")
            continue
        
    # split the data into train(85%), val(10%), test sets(5%)
    train_size = int(0.85 * len(molecules))
    val_size = int(0.10 * len(molecules))
    test_size = len(molecules) - train_size - val_size
    # split the data, but keep the order
    train_molecules = molecules[:train_size]
    val_molecules = molecules[train_size:train_size+val_size]
    test_molecules = molecules[train_size+val_size:]
    print(f"Train size: {len(train_molecules)}, Val size: {len(val_molecules)}, Test size: {len(test_molecules)}")
    # Create the model
    geatnet = GeATNet(
        atom_embedding_dim=config.atom_embedding_dim,
        num_atom_types=config.num_atom_types,
        bond_embedding_dim=config.bond_embedding_dim,
        num_bond_types=len(BondTypes.get_bond_types()),
        edge_attetion_output_negative_slope=config.edge_attetion_output_negative_slope,
        num_heads=config.num_heads,
        global_num_heads=config.global_num_heads,
        backbone_dropout=config.backbone_dropout,
        neck_dropout=config.neck_dropout,
        head_dropout=config.head_dropout,
        geatnet_hidden_dim=config.geatnet_hidden_dim,
        geatnet_layers=config.geatnet_layers,
        parallel_between_bondtypes=config.parallel_between_bondtypes
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    geatnet.to(device)

    # Create the optimizer
    optimizer = torch.optim.Adam(geatnet.parameters(), lr=1e-4, weight_decay=1e-5)
    # Create the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5, verbose=True)
    # Create the loss function
    criterion = nn.MSELoss()
    # Training loop
    num_epochs = 100
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        try:
            geatnet.train()
            total_loss = 0.0
            for i, (atom_embeddings, edges, target_label) in enumerate(train_molecules):
                atom_embeddings = atom_embeddings.to(device)
                edges = edges.to(device)
                target_label = torch.tensor([(target_label - min_label) / (max_label - min_label)], dtype=torch.float32).to(device)
                optimizer.zero_grad()
                output = geatnet(atom_embeddings.unsqueeze(0), edges.unsqueeze(0))
                loss = criterion(output, target_label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_train_loss = total_loss / len(train_molecules)
            train_losses.append(avg_train_loss)

            geatnet.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for i, (atom_embeddings, edges, target_label) in enumerate(val_molecules):
                    atom_embeddings = atom_embeddings.to(device)
                    edges = edges.to(device)
                    target_label = torch.tensor([(target_label - min_label) / (max_label - min_label)], dtype=torch.float32).to(device)
                    output = geatnet(atom_embeddings.unsqueeze(0), edges.unsqueeze(0))
                    loss = criterion(output, target_label)
                    total_val_loss += loss.item()
            avg_val_loss = total_val_loss / len(val_molecules)
            val_losses.append(avg_val_loss)

            scheduler.step(avg_val_loss)

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

            # Save the model if the validation loss is the best we've seen so far.
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(geatnet.state_dict(), "best_geatnet_model.pth")
                print("Saved the best model.")

        except KeyboardInterrupt:
            print("Training interrupted. Saving the current model...")
            torch.save(geatnet.state_dict(), "interrupted_geatnet_model.pth")
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
    # Load the best model and evaluate on the test set
    geatnet.load_state_dict(torch.load("best_geatnet_model.pth"))
    geatnet.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for i, (atom_embeddings, edges, target_label) in enumerate(test_molecules):
            atom_embeddings = atom_embeddings.to(device)
            edges = edges.to(device)
            target_label = torch.tensor([(target_label - min_label) / (max_label - min_label)], dtype=torch.float32).to(device)
            output = geatnet(atom_embeddings.unsqueeze(0), edges.unsqueeze(0))
            loss = criterion(output, target_label)
            total_test_loss += loss.item()
    avg_test_loss = total_test_loss / len(test_molecules)
    print(f"Test Loss: {avg_test_loss:.6f}")
    print("Testing completed.")