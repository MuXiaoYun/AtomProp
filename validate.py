from atomprop.models.GeAT import GeATNet
from atomprop.embeddings.AtomEmbedding import SMILESToInputs
import configs.config as config
import rdkit.Chem as Chem
import rdkit.Chem.Descriptors as Descriptors
import torch
import torch.nn as nn

#load geatnet model from model_path
model_path = "trained_models/geatnet_epoch_10.pth"

max_mass = 6000.0  # Maximum relative mass for normalization
#load model from dict
geatnet = GeATNet(atom_embedding_dim=config.atom_embedding_dim,
                     num_atom_types=config.num_atom_types,
                     num_bond_types=config.num_bond_types,
                     num_heads=config.num_heads,
                     global_num_heads=config.global_num_heads,
                     )
device = torch.device("cpu")
print(f"Using device: {device}")
geatnet.to(device)
print("geatnet:", geatnet)
total_params = sum(p.numel() for p in geatnet.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params}")
# Load the model state dict
geatnet.load_state_dict(torch.load(model_path, map_location=device))

# Set the model to evaluation mode
geatnet.eval()
# Atoms to embed
atom_list = [
    6,7,8,9,14,16,17,35
]
# Predict 14 SMILES strings
smiles_list = [
    "CCO",  # Ethanol
    "C1=CC=CC=C1",  # Benzene
    "Oc1cc(c2ccccc2)c(N=Nc2ccccc2)cc1n1cnc2ccccc21",  # A complex aromatic compound
    "CCCCCC=CCC=CCC=CCC=CCC=CCCC(=O)OCC(COC(=O)CCCCCCCCCC=CCCCCCCCC)OC(=O)CCCCCCCCCC=CCC=CCCCCC",
    "CCc1ccc(c2ccc(c3ccc(OC)cc3)c(c3ccc(CC)cc3)[s+]2)cc1",
    "COc1ccc(CNc2ccc(C)c(F)c2)c(O)c1",
    "O=C1NCc2c(Cl)ccc(c3cc4cc(C(=O)N5CCCCC5CCO)ccc4[nH]3)c21",
    "C=CCOc1cc(C)ccc1C(=O)C=CC[NH+](C)C",
    "CCOc1cc(C(=O)Nc2nc(c3ccc(I)cc3)cs2)cc(OCC)c1OCC",
    "Cn1cnc2cc3c(cc21)CC1N(C(=O)c2ccc4c(c2)CC=N4)CCC3(C)C1(C)C",
    "CCC(C)(CC)Oc1ccc(C2(c3ccc(OC(=O)c4ccccc4)c(C(F)(F)F)c3)CCCCC2)cc1C(F)(F)F",
    "COc1cc(C(C#N)=Cc2ccccc2[N+](=O)[O-])ccc1[N+](=O)[O-]",
    "CNc1nccc(N2CCC(CC(=O)NCc3ccc(F)cc3)CC2)n1",
    "CCCCc1nc(COC)c(C[NH2+]CCC)s1",
]
molecules = []
for i, smiles in enumerate(smiles_list):
    if i % 100 == 0:
        print(f"Processing {i}th SMILES: {smiles}")
    if i > 50000:
        break
    # Convert SMILES to GeAT inputs
    atom_indices, edges, mol = SMILESToInputs.convert(smiles=smiles, context_length=config.context_length)
    if mol is not None:
        # Append the molecule and its mol mass with Hs
        molecules.append((atom_indices, edges, Descriptors.MolWt(Chem.AddHs(mol))))

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
    
# store key embeddings
fc1_input = None # 14 molecule embeddings with shape (14, 64), labels are smiles_list

def fc1_hook(module, input, output):
    global fc1_input
    fc1_input = input[0]

hook_fc1 = geatnet.fc1.register_forward_hook(fc1_hook)

# Create a dataset and dataloader
dataset = MoleculeDataset(molecules)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=False)
# Evaluate the model on the dataset
for i, (atom_indices, edges, rel_mass) in enumerate(dataloader):
    atom_indices = atom_indices.to(device)
    edges = edges.to(device)
    rel_mass = rel_mass.float().to(device)
    outputs = geatnet(atom_indices, edges)
    print("mass:", rel_mass*max_mass)
    print("predicts:", outputs.squeeze(-1)*max_mass)

hook_fc1.remove()

atom_indices_tensor = torch.tensor(atom_list).reshape(8, 1, -1)
atom_embeddings = geatnet.atom_embedding(atom_indices_tensor).reshape(8, 64) # 8 atom embeddings with shape (8, 64), labels are ["C", "N", "O", "F", "Si", "S", "Cl", "Br"]
print("atom embeddings: ", atom_embeddings) 
print("shape of atom embeddings: ", atom_embeddings.shape)
print("molecule embeddings: ", fc1_input) 
print("shape of molecule embeddings: ", fc1_input.shape)

# use t-SNE to visualize the 8 atom embeddings and the 3 molecule embeddings in one 3D plot
# label the atom symbol or molecule SMILES beside each point
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
atom_labels = ["C", "N", "O", "F", "Si", "S", "Cl", "Br"]
molecule_labels = smiles_list
tsne = TSNE(n_components=3, perplexity=5, random_state=42)
# mark the atom embeddings with blue color and molecule embeddings with red color
embeddings = torch.cat((atom_embeddings, fc1_input), dim=0)
tsne_embeddings = tsne.fit_transform(embeddings.detach().numpy())
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i, label in enumerate(atom_labels):
    ax.scatter(tsne_embeddings[i, 0], tsne_embeddings[i, 1], tsne_embeddings[i, 2], color='blue', label=label)
for i, label in enumerate(molecule_labels):
    ax.scatter(tsne_embeddings[i + len(atom_labels), 0], tsne_embeddings[i + len(atom_labels), 1], tsne_embeddings[i + len(atom_labels), 2], color='red', label=label)
# Add labels to each point
for i, label in enumerate(atom_labels):
    ax.text(tsne_embeddings[i, 0], tsne_embeddings[i, 1], tsne_embeddings[i, 2], label, color='blue')
for i, label in enumerate(molecule_labels):
    ax.text(tsne_embeddings[i + len(atom_labels), 0], tsne_embeddings[i + len(atom_labels), 1], tsne_embeddings[i + len(atom_labels), 2], label, color='red')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('t-SNE Visualization of Atom and Molecule Embeddings')
plt.legend()
plt.show()
