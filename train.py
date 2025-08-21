from atomprop.models.GeAT import GeATNet
from atomprop.embeddings.AtomEmbedding import SMILESToInputs
import configs.config as config
import rdkit.Chem as Chem
import rdkit.Chem.Descriptors as Descriptors
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

data_path = "data/pubchem/pubchem-10m.txt"

max_mass = 6000.0  # Maximum relative mass for normalization
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
    # load the data from the text file, each line is a SMILES string
    with open(data_path, 'r') as f:
        lines = f.readlines()
    smiles_list = [line.strip() for line in lines if line.strip()]
    print("Number of SMILES strings:", len(smiles_list))
    molecules = []
    # atom_types = set()
    # bond_types = set()
    # safe train num: 117800
    for i, smiles in enumerate(smiles_list):
        if i % 100 == 0:
            print(f"Processing {i}th SMILES: {smiles}")
        if i > 117800:
            break
        # Convert SMILES to GeAT inputs
        atom_embeddings, edges, mol = SMILESToInputs.convert(smiles=smiles, context_length=max_atom_num)
        # # Update maximum atomic number
        # for atom_type_index in mol_input[0]:
        #     if atom_type_index > max_atomic_num:
        #         max_atomic_num = atom_type_index
        if mol is not None:
            # Append the molecule and its mol mass with Hs
            molecules.append((atom_embeddings, edges, Descriptors.MolWt(Chem.AddHs(mol))))
    # print(f"Max atomic number in dataset: {max_atomic_num}")

    # model instance
    geatnet = GeATNet(atom_embedding_dim=config.atom_embedding_dim,
                     num_atom_types=config.num_atom_types,
                     num_bond_types=config.num_bond_types,
                     num_heads=config.num_heads,
                     global_num_heads=config.global_num_heads,
                     )
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
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    # log-mse loss
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(geatnet.parameters(), lr=0.001)
    num_epochs = 350
    losses = []
    
    for epoch in range(num_epochs):
        try:
            geatnet.train()
            running_loss = 0.0
            for i, (atom_embeddings, edges, rel_mass) in enumerate(dataloader):
                atom_embeddings = atom_embeddings.to(device)
                edges = edges.to(device)
                rel_mass = rel_mass.float().to(device)

                optimizer.zero_grad()
                outputs = geatnet(atom_embeddings, edges)

                # for last step of each epoch, calculate relative error between predicted and true mass
                if i == len(dataloader) - 1:
                    with torch.no_grad():
                        pred_mass = outputs.squeeze(-1) * max_mass
                        true_mass = rel_mass * max_mass
                        relative_errors = torch.abs(pred_mass - true_mass) / true_mass
                        avg_relative_error = torch.mean(relative_errors).item()
                        print(f"Epoch [{epoch + 1}/{num_epochs}], Last Batch Average Relative Error: {avg_relative_error:.4f}")

                loss = criterion(outputs, rel_mass.unsqueeze(-1))  # Log-MSE loss
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if (i + 1) % 10 == 0:
                    print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")
                    # print_grad_norm(geatnet, prefix=f"Epoch {epoch + 1}, Step {i + 1}")

            print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {running_loss / len(dataloader):.4f}")
            losses.append(running_loss / len(dataloader))
        except Exception as e:
            # if it's not the first epoch and the first batch, save the model
            if epoch > 0 or i > 0:
                print(f"Error occurred: {e}. Saving model state.")
                torch.save(geatnet.state_dict(), f"trained_models/geatnet_epoch_{epoch + 1}_step_{i + 1}.pth")
                # plot the loss curve
                plt.plot(range(1, len(losses) + 1), losses)
                plt.xlabel("Epoch")
                plt.ylabel("Average Loss")
                plt.title("Training Loss Curve")
                plt.savefig("training_loss_curve.png")
            raise e
        # Save the model state after each epoch
        torch.save(geatnet.state_dict(), f"trained_models/geatnet_epoch_{epoch + 1}.pth")
    print("Training complete. Model saved.")
    # plot the loss curve
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss Curve")
    plt.savefig("training_loss_curve.png")
