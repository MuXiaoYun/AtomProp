from atomprop.models.GeAT import GeATNet
from atomprop.embeddings.AtomEmbedding import SMILESToInputs
import configs.config as config
import rdkit.Chem as Chem
import rdkit.Chem.Descriptors as Descriptors
import torch
import torch.nn as nn

data_path = "data/pubchem/pubchem-10m.txt"

max_mass = 6000.0  # Maximum relative mass for normalization
max_atom_num = 420  # Maximum number of atoms in a molecule for padding
max_edge_size = 420  # Maximum number of edges in a molecule for padding

if __name__ == "__main__":
    # load the data from the text file, each line is a SMILES string
    with open(data_path, 'r') as f:
        lines = f.readlines()
    smiles_list = [line.strip() for line in lines if line.strip()]
    print("Number of SMILES strings:", len(smiles_list))
    molecules = []
    # atom_types = set()
    # bond_types = set()
    for i, smiles in enumerate(smiles_list):
        if i % 100 == 0:
            print(f"Processing {i}th SMILES: {smiles}")
        if i > 1000:
            break
        # Convert SMILES to GeAT inputs
        mol_input, mol = SMILESToInputs.convert(smiles=smiles, context_length_node=max_atom_num, context_length_edge=max_edge_size)
        # # Update maximum atomic number
        # for atom_type_index in mol_input[0]:
        #     if atom_type_index > max_atomic_num:
        #         max_atomic_num = atom_type_index
        if mol is not None:
            # Append the molecule and its mol mass
            molecules.append((mol_input, Descriptors.MolWt(mol)))
    # print(f"Max atomic number in dataset: {max_atomic_num}")

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
            # max_atom_num = max(len(mol[0][0]) for mol in self.molecules)
            # max_edge_size = max(len(mol[0][1]) for mol in self.molecules)
            # print(f"Max atom number in dataset: {max_atom_num}")
            # print(f"Max edge size in dataset: {max_edge_size}")
            rel_mass = self.molecules[idx][1] / max_mass
            return self.molecules[idx][0], rel_mass
        
    # Convert inputs to tensor and create the dataset 
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

            for (atom_type_indices, edges), rel_mass in train_loader:
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
