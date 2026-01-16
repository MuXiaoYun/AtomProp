"""
This script reads a file containing one SMILES string per line,
validates each SMILES using RDKit, prints all invalid SMILES with line numbers,
and writes a cleaned version (only valid SMILES) to a new file in the same directory.
The new file is named <original_name>.clean.

It processes the file line-by-line to handle large files efficiently.
"""

import sys
import os
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
    Main function to scan the input file, report invalid SMILES,
    and write a cleaned file with only valid SMILES.
    
    Args:
        input_file (str): Path to the input file containing SMILES (one per line).
    """
    print(f"Scanning file: {input_file}")
    
    # Generate output filename: same dir, same name + '.clean'
    base_name = os.path.basename(input_file)
    dir_name = os.path.dirname(input_file)
    output_file = os.path.join(dir_name, base_name + ".clean")

    invalid_count = 0
    valid_count = 0

    try:
        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_file, 'w', encoding='utf-8') as fout:

            for line_num, line in enumerate(fin, start=1):
                smiles = line.strip()

                # Progress indicator
                if line_num % 10000 == 0:
                    print(f"Processed {line_num} lines...")

                # Skip empty lines
                if not smiles:
                    continue

                if is_valid_smiles(smiles):
                    fout.write(smiles + '\n')
                    valid_count += 1
                else:
                    print(f"Invalid SMILES at line {line_num}: {smiles}")
                    invalid_count += 1

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nProcessing complete.")
    print(f"Valid SMILES written to: {output_file}")
    print(f"Total valid:   {valid_count}")
    print(f"Total invalid: {invalid_count}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python datawash.py <input_smiles_file.txt>", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    main(input_path)