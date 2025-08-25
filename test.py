from atomprop.models.GeAT import GeATNet
import configs.config as config
import rdkit.Chem as Chem
import torch

data_path = "data/nabladft/summary.csv"

if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")

    # Use the default configurations
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
    print(geatnet)
    # Print param numbers
    total_params = sum(p.numel() for p in geatnet.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # Load data from the CSV file
    # smiles for input and the rest for regression targets
    
    import pandas as pd
    df = pd.read_csv(data_path)
    print(df.head())
    # Print the columns
    print("Columns in the dataset:", df.columns.tolist())
    # Use rdkit.chem to handle SMILES strings. Count how many kinds of atom types and bond types are in the dataset
    smiles_list = df['SMILES'].tolist()
    print("Number of SMILES strings:", len(smiles_list))

    atom_types = set()
    bond_types = set()

    max_num = -1

    for i, smiles in enumerate(smiles_list):
        if i % 100 == 0:
            print(f"Processing {i}th SMILES: {smiles}")
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            for atom in mol.GetAtoms():
                atom_types.add(atom.GetSymbol())
            for bond in mol.GetBonds():
                bond_types.add(bond.GetBondTypeAsDouble())
            atom_num = mol.GetNumAtoms()
            if atom_num > max_num:
                max_num = atom_num
        else:
            print(f"Invalid SMILES string at index {i}: {smiles}")
    
    for mol in molecules:
        if mol is not None:
            for atom in mol.GetAtoms():
                atom_types.add(atom.GetSymbol())
            for bond in mol.GetBonds():
                bond_types.add(bond.GetBondTypeAsDouble())
    print("Number of atom types:", len(atom_types))
    print("Atom types:", atom_types)
    print("Number of bond types:", len(bond_types))
    print("Bond types:", bond_types)
    print("Maximum number of atoms in a molecule:", max_num)
    