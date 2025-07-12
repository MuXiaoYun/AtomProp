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
        atom_embeddings, edges, mol = SMILESToInputs.convert(smiles=smiles, context_length=max_atom_num)
        # # Update maximum atomic number
        # for atom_type_index in mol_input[0]:
        #     if atom_type_index > max_atomic_num:
        #         max_atomic_num = atom_type_index
        if mol is not None:
            # Append the molecule and its mol mass
            molecules.append((atom_embeddings, edges, Descriptors.MolWt(mol)))
    # print(f"Max atomic number in dataset: {max_atomic_num}")

    # model instance
    geatnet = GeATNet(atom_embedding_dim=config.atom_embedding_dim,
                     num_atom_types=config.num_atom_types,
                     num_bond_types=config.num_bond_types)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    geatnet.to(device)
    
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
            rel_mass = self.molecules[idx][2] / max_mass
            return self.molecules[idx][0], self.molecules[idx][1], rel_mass
        
    # Convert inputs to tensor and create the dataset 
    dataset = MoleculeDataset(molecules)

    # train geatnet with the dataset and draw training loss curve
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(geatnet.parameters(), lr=0.001)
    num_epochs = 10
    for epoch in range(num_epochs):
        geatnet.train()
        running_loss = 0.0
        for i, (atom_embeddings, edges, rel_mass) in enumerate(dataloader):
            atom_embeddings = atom_embeddings.to(device)
            edges = edges.to(device)
            rel_mass = rel_mass.to(device)

            optimizer.zero_grad()
            outputs = geatnet(atom_embeddings, edges)
            loss = criterion(outputs, rel_mass)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {running_loss / len(dataloader):.4f}")

