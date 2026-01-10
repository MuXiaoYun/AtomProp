"""
This script reads a file containing one SMILES string per line,
validates each SMILES using RDKit, and prints all invalid SMILES
along with their line numbers (1-indexed).
"""

import sys
from rdkit import Chem
from rdkit import RDLogger

# Suppress RDKit warnings (e.g., valence issues) to avoid cluttering output
RDLogger.DisableLog('rdApp.*')


def is_valid_smiles(smiles: str) -> bool:
    """
    Check if a SMILES string is valid using RDKit.
    
    Args:
        smiles (str): The SMILES string to validate.
    
    Returns:
        bool: True if valid, False otherwise.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return False
    mol = Chem.MolFromSmiles(smiles.strip())
    return mol is not None


def main(input_file: str):
    """
    Main function to scan the input file and report invalid SMILES.
    
    Args:
        input_file (str): Path to the input file containing SMILES (one per line).
    """
    print(f"Scanning file: {input_file}")
    invalid_count = 0

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, start=1):
                smiles = line.strip()
                # Skip empty lines
                if not smiles:
                    continue
                if not is_valid_smiles(smiles):
                    print(f"Invalid SMILES at line {line_num}: {smiles}")
                    invalid_count += 1
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nTotal invalid SMILES found: {invalid_count}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python datawash.py <input_smiles_file.txt>", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    main(input_path)