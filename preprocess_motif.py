# Functional group detection script
# It reads SMILES from a txt and detects functional groups

import rdkit
from rdkit import Chem
from rdkit.Chem import FunctionalGroups
import time
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

molecule_num = -1  # -1 means load full dataset
data_path = "data/pubchem/pubchem-10m.txt"  # each row is a SMILES
output_path = "data/pubchem/pubchem-functional-groups.txt"  # output file path

# Count how many molecules have been processed from output file
processed_count = 0
try:
    with open(output_path, "r") as f:
        processed_count = sum(1 for line in f if line.strip())
    print(f"Found {processed_count} molecules already processed in output file")
except FileNotFoundError:
    print("Output file does not exist, starting from scratch")
    processed_count = 0

# Build functional group hierarchy
print("Building functional group hierarchy...")
fg_hierarchy = FunctionalGroups.BuildFuncGroupHierarchy()

# Extract all functional group patterns
functional_groups = []
def extract_functional_groups(fg_list):
    for fg in fg_list:
        functional_groups.append(fg)
        if hasattr(fg, 'children') and fg.children:
            extract_functional_groups(fg.children)

extract_functional_groups(fg_hierarchy)
print(f"Found {len(functional_groups)} functional groups in hierarchy")

# We'll use the 12 main functional groups you mentioned
# If you want to use all, remove the slicing below
selected_fg = functional_groups[:12]  # Use first 12 functional groups
print(f"Using {len(selected_fg)} functional groups for detection")

# open files
input_f = open(data_path, "r")
out_f = open(output_path, "a")

start_time = time.time()

count = 0

def detect_functional_groups(mol, functional_groups_list):
    """Detect presence of functional groups in a molecule"""
    presence_vector = []
    
    for fg in functional_groups_list:
        pattern = fg.pattern
        matches = mol.GetSubstructMatches(pattern)
        # 1 if present, 0 if absent
        presence_vector.append(1 if matches else 0)
    
    return presence_vector

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
    
    try:
        # Detect functional groups
        fg_vector = detect_functional_groups(mol, selected_fg)
        
        # Convert to string and write to file
        fg_str = " ".join(str(x) for x in fg_vector)
        out_f.write(fg_str + "\n")
        
    except Exception as e:
        # In case of any error, write all zeros
        print(f"Error processing molecule {count}, SMILES: {smiles}")
        print(f"Error: {e}")
        out_f.write(" ".join(["0"] * len(selected_fg)) + "\n")
        continue

# Close files
input_f.close()
out_f.close()

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Processing completed!")
print(f"Total molecules processed: {count}")
print(f"Used {elapsed_time:.2f} seconds.")

# Print functional group names for reference
print("\nFunctional groups used (in order):")
for i, fg in enumerate(selected_fg):
    fg_name = getattr(fg, 'label', f'Group_{i}')
    print(f"{i}: {fg_name}")