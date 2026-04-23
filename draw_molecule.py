from rdkit import Chem
from rdkit.Chem import Draw
import sys

def draw_molecule_from_smiles(smiles):
    """
    This function takes a SMILES string as an input,
    converts it into a molecule object, draws its structure,
    and saves the image in the current directory.
    
    :param smiles: str, input SMILES string of the molecule
    """
    # Convert SMILES to RDKit molecule object
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Invalid SMILES string.")
        return
    
    # Generate image of the molecule
    img = Draw.MolToImage(mol)
    
    # Save the image with a filename based on the SMILES (sanitized)
    filename = Chem.MolToSmiles(mol, isomericSmiles=True).replace("/", "_").replace("\\", "_") + ".png"
    img.save(filename)
    print(f"Molecule image saved as {filename}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python draw_molecule.py <SMILES>")
    else:
        smiles_input = sys.argv[1]
        draw_molecule_from_smiles(smiles_input)