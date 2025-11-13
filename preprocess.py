# This is a test script
# It reads SMILES from a txt and calculates coordinates of atoms

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import time
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

molecule_num = -1  # -1 means load full dataset
data_path = "data/pubchem/pubchem-10m.txt" #each row is a SMILES

# open files
input_f = open(data_path, "r")
out_f = open("data/pubchem/pubchem-xyzs.txt", "w")

start_time = time.time()

successful_conformers = 0
count = 0
error_embed_count = 0
error_optimize_count = 0
invalid_smiles_count = 0

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
    if count % 100 == 0:
        print(f"Processing molecule {count}" + (f"/{molecule_num}" if molecule_num != -1 else ""))
    
    smiles = line.strip()
    if not smiles:  # Skip empty lines
        invalid_smiles_count += 1
        continue
    
    # Convert SMILES to molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        invalid_smiles_count += 1
        continue
    # remove Hs
    mol = Chem.RemoveHs(mol)
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
            successful_conformers += 1
            out_f.write("\n")
        except:
            out_f.write(f"ERROR({num_atoms}): OPTIMIZE\n\n")
            error_optimize_count += 1
            continue
    else:
        out_f.write(f"ERROR({num_atoms}): EMBED\n\n")
        error_embed_count += 1
        continue

# Close files
input_f.close()
out_f.close()

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Processing completed!")
print(f"Total molecules processed: {count}")
print(f"Used {elapsed_time:.2f} seconds.")
print(f"Successful conformers: {successful_conformers}")
print(f"Statistics:")
print(f"  - Invalid SMILES: {invalid_smiles_count}")
print(f"  - Embed errors: {error_embed_count}")
print(f"  - Optimize errors: {error_optimize_count}")
print(f"  - Success rate: {successful_conformers/count*100:.2f}%" if count > 0 else "N/A")