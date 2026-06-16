"""Draw a molecule structure from a SMILES string."""

import argparse
import sys

from rdkit import Chem
from rdkit.Chem import Draw

from atomprop.paths import FIGURES_DIR, ensure_output_dirs


def draw_molecule_from_smiles(smiles: str, output_dir=FIGURES_DIR) -> str | None:
    """Convert SMILES to a PNG structure image."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Invalid SMILES string.")
        return None

    ensure_output_dirs()
    img = Draw.MolToImage(mol)
    filename = (
        Chem.MolToSmiles(mol, isomericSmiles=True).replace("/", "_").replace("\\", "_")
        + ".png"
    )
    output_path = output_dir / filename
    img.save(output_path)
    print(f"Molecule image saved as {output_path}")
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw a molecule from SMILES")
    parser.add_argument("smiles", help="Input SMILES string")
    args = parser.parse_args()
    draw_molecule_from_smiles(args.smiles)
