from atomprop.models.GeAT import GeATNet
from atomprop.embeddings.AtomEmbedding import SMILESToInputs
import configs.config as config
import rdkit.Chem as Chem
import rdkit.Chem.Descriptors as Descriptors
import torch
import torch.nn as nn

data_path = "data/pubchem/pubchem-10m.txt"

if __name__ == "__main__":
    # load the data from the text file, each line is a SMILES string
    with open(data_path, 'r') as f:
        lines = f.readlines()
    smiles_list = [line.strip() for line in lines if line.strip()]
    print("Number of SMILES strings:", len(smiles_list))
    molecules = []
    atom_types = set()
    bond_types = set()
    for i, smiles in enumerate(smiles_list):
        if i % 100 == 0:
            print(f"Processing {i}th SMILES: {smiles}")
        if i > 10000:
            break
        # Convert SMILES to GeAT inputs
        mol_input, mol = SMILESToInputs.convert(smiles)
        if mol is not None:
            # Append the molecule and its relative mol mass
            molecules.append((mol_input, Descriptors.MolWt(mol)))

    # model instance
    geatnet = GeATNet(atom_embedding_dim=config.atom_embedding_dim,
                     num_atom_types=config.num_atom_types,
                     num_bond_types=config.num_bond_types)
    
    # create a pytorch dataset with converted mol inputs and their relative mol mass
    class MoleculeDataset(torch.utils.data.Dataset):
        def __init__(self, molecules):
            self.molecules = molecules
        def __len__(self):
            return len(self.molecules)
        def __getitem__(self, idx):
            # Normalize relative mass to a range of 0-1
            max_mass = max([m[1] for m in self.molecules])
            print(f"Max mass in dataset: {max_mass}")
            rel_mass = self.molecules[idx][1] / 6000.0
            return self.molecules[idx][0], rel_mass
        
    dataset = MoleculeDataset(molecules)

    # train geatnet with the dataset by using a 5-fold cross-validation and draw 2 loss curves
    from sklearn.model_selection import KFold
    import matplotlib.pyplot as plt

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(geatnet.parameters(), lr=0.001)
    epochs = 10
    all_losses = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f"Fold {fold + 1}")
        train_loader = torch.utils.data.DataLoader(
            [dataset[i] for i in train_idx], batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            [dataset[i] for i in val_idx], batch_size=32, shuffle=False)

        for epoch in range(epochs):
            geatnet.train()
            total_loss = 0

            for atom_type_indices, edges, rel_mass in train_loader:
                optimizer.zero_grad()
                outputs = geatnet(atom_type_indices, edges)
                loss = loss_fn(outputs.squeeze(), rel_mass.float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
            all_losses.append(avg_loss)

        # save the model for each epoch
        save_path = f"model_fold_{fold + 1}.pth"
        torch.save(geatnet.state_dict(), save_path)
    print("Training complete. Models saved for each fold.")
    
    # Plot the loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(all_losses, label='Training Loss')
    plt.title('Training Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
