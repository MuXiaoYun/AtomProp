# Preprocessing script
# It reads SMILES from a txt and calculates coordinates of atoms

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import time
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

molecule_num = -1  # -1 means load full dataset
data_path = "data/tests/pubchem_unkekulized_test.txt" #each row is a SMILES
xyz_path = "data/tests/pubchem_unkekulized_test_xyzs.txt"
# Count how many molecules have been processed from output file
processed_count = 0
try:
    with open(xyz_path, "r") as f:
        for line in f:
            if line.strip() == "":
                processed_count += 1
    print(f"Found {processed_count} molecules already processed in output file")
except FileNotFoundError:
    print("Output file does not exist, starting from scratch")
    processed_count = 0

# open files
input_f = open(data_path, "r")
out_f = open(xyz_path, "a")

start_time = time.time()

count = 0

# Process line by line
while True:
    # Read one line
    line = input_f.readline()
    if not line:  # End of file
        break
    
    # Check if we've processed enough molecules
    if molecule_num != -1 and count >= molecule_num:
        break
    
    count += 1
    
    # Skip already processed molecules
    if count <= processed_count:
        if count % 100000 == 0:
            print(f"Skipping molecule {count}")
        continue
    
    if count % 100 == 0:
        print(f"Processing molecule {count}" + (f"/{molecule_num}" if molecule_num != -1 else ""))
    
    smiles = line.strip()
    if not smiles:  # Skip empty lines
        continue
    
    # Convert SMILES to molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        continue
    
    # Remove Hs with error handling
    try:
        mol = Chem.RemoveHs(mol)
    except Exception as e:
        print(f"Failed to remove Hs for molecule {count}, SMILES: {smiles}")
        print(f"Error: {e}")
        # If molecule cannot be kekulized, it cannot be embedded, so skip it
        out_f.write(f"ERROR({mol.GetNumAtoms()}): REMOVE_HS\n\n")
        continue
    
    num_atoms = mol.GetNumAtoms()
    
    # try embedding molecules
    embed_result = AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
    
    if embed_result >= 0:  # 0 for success, 1 for random init success
        try:
            # try optimizing
            opt_result = AllChem.UFFOptimizeMolecule(mol)
            conf = mol.GetConformer()
            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                out_f.write(f"{pos.x} {pos.y} {pos.z}\n")
            out_f.write("\n")
        except:
            out_f.write(f"ERROR({num_atoms}): OPTIMIZE\n\n")
            continue
    else:
        out_f.write(f"ERROR({num_atoms}): EMBED\n\n")
        continue

# Close files
input_f.close()
out_f.close()

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Processing completed!")
print(f"Total molecules processed: {count}")
print(f"Used {elapsed_time:.2f} seconds.")