from atomprop.models.GeAT import GeATNet
import configs.config as config
import rdkit.Chem as Chem
import torch

# data_path = "data/chemprop/regression/mol_multitask.csv"
data_path = "data/pubchem/pubchem-10m.txt"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    # # Use the default configurations
    # geatnet = GeATNet(atom_embedding_dim=config.atom_embedding_dim,
    #                  num_atom_types=config.num_atom_types,
    #                  num_bond_types=config.num_bond_types)
    # print(geatnet)
    # # Print param numbers
    # total_params = sum(p.numel() for p in geatnet.parameters() if p.requires_grad)
    # print(f"Total trainable parameters: {total_params}")

    # # Load data from the CSV file
    # # headers are [smiles	mu	alpha	homo	lumo	gap	r2	zpve	cv	u0	u298	h298	g298]
    # # smiles for input and the rest for regression targets
    
    # import pandas as pd
    # df = pd.read_csv(data_path)
    # print(df.head())
    # # Print the columns
    # print("Columns in the dataset:", df.columns.tolist())
    # # Use rdkit.chem to handle SMILES strings. Count how many kinds of atom types and bond types are in the dataset
    # smiles_list = df['smiles'].tolist()
    # print("Number of SMILES strings:", len(smiles_list))
    # molecules = []
    # for i, smiles in enumerate(smiles_list):
    #     if i % 100 == 0:
    #         print(f"Processing {i}th SMILES: {smiles}")
    #     molecules.append(Chem.MolFromSmiles(smiles))
    # atom_types = set()
    # bond_types = set()
    # for mol in molecules:
    #     if mol is not None:
    #         for atom in mol.GetAtoms():
    #             atom_types.add(atom.GetSymbol())
    #         for bond in mol.GetBonds():
    #             bond_types.add(bond.GetBondTypeAsDouble())
    # print("Number of atom types:", len(atom_types))
    # print("Atom types:", atom_types)
    # print("Number of bond types:", len(bond_types))
    # print("Bond types:", bond_types)

    # # load the data from the text file, each line is a SMILES string
    # with open(data_path, 'r') as f:
    #     lines = f.readlines()
    # smiles_list = [line.strip() for line in lines if line.strip()]
    # print("Number of SMILES strings:", len(smiles_list))
    # # Use rdkit.chem to handle SMILES strings. Count how many kinds of atom types and bond types are in the dataset
    # molecules = []
    # atom_types = set()
    # bond_types = set()
    # for i, smiles in enumerate(smiles_list):
    #     if i % 100 == 0:
    #         print(f"Processing {i}th SMILES: {smiles}")
    #     if i > 200000:
    #         break
    #     mol = Chem.MolFromSmiles(smiles)
    #     if mol is not None:
    #         molecules.append(mol)
    #         for atom in mol.GetAtoms():
    #             atom_types.add(atom.GetSymbol())
    #         for bond in mol.GetBonds():
    #             bond_types.add(bond.GetBondTypeAsDouble())
    # print("Number of molecules:", len(molecules))
    # print("Number of atom types:", len(atom_types))
    # print("Atom types:", atom_types)
    # print("Number of bond types:", len(bond_types))
    # print("Bond types:", bond_types)
    