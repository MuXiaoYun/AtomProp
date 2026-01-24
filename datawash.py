"""
This script reads a file containing SMILES strings and validates them using RDKit.
- If file_type is "txt": one SMILES per line (original behavior).
- If file_type is "csv": expects a column named "SMILES" or "smiles"; invalid rows are removed.

Invalid SMILES (with line numbers for TXT, or row indices for CSV) are printed.
A cleaned file (<original_name>.clean) is written in the same directory.
Processes files efficiently (line-by-line for TXT, chunked for CSV if needed).
"""

import sys
import os
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

# Suppress RDKit warnings to avoid cluttering output
RDLogger.DisableLog('rdApp.*')

file_type = "csv"  # Can be "txt" or "csv"

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


def process_txt_file(input_file: str):
    """Process a .txt file with one SMILES per line."""
    print(f"Scanning TXT file: {input_file}")
    
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

                if line_num % 10000 == 0:
                    print(f"Processed {line_num} lines...")

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
        print(f"Error processing TXT file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"\nProcessing complete.")
    print(f"Valid SMILES written to: {output_file}")
    print(f"Total valid:   {valid_count}")
    print(f"Total invalid: {invalid_count}")


def process_csv_file(input_file: str):
    """Process a CSV file containing a 'SMILES' or 'smiles' column."""
    print(f"Scanning CSV file: {input_file}")
    
    base_name = os.path.basename(input_file)
    dir_name = os.path.dirname(input_file)
    output_file = os.path.join(dir_name, base_name + ".clean")

    try:
        df = pd.read_csv(input_file, dtype=str)  # Read all columns as strings to avoid parsing issues
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file: {e}", file=sys.stderr)
        sys.exit(1)

    # Find SMILES column (case-insensitive)
    smiles_col = None
    for col in df.columns:
        if col.lower() == "smiles":
            smiles_col = col
            break

    if smiles_col is None:
        print("Error: CSV file must contain a 'SMILES' or 'smiles' column.", file=sys.stderr)
        sys.exit(1)

    print(f"Found SMILES column: '{smiles_col}'")

    valid_rows = []
    invalid_count = 0
    total_rows = len(df)

    for idx, row in df.iterrows():
        smiles_val = row[smiles_col]

        # Progress indicator
        if (idx + 1) % 10000 == 0:
            print(f"Processed {idx + 1} / {total_rows} rows...")

        if pd.isna(smiles_val) or not str(smiles_val).strip():
            print(f"Invalid SMILES at row {idx + 1} (empty or NaN): {smiles_val}")
            invalid_count += 1
            continue

        smiles_str = str(smiles_val).strip()
        if is_valid_smiles(smiles_str):
            valid_rows.append(row)
        else:
            print(f"Invalid SMILES at row {idx + 1}: {smiles_str}")
            invalid_count += 1

    # Create cleaned DataFrame
    cleaned_df = pd.DataFrame(valid_rows, columns=df.columns) if valid_rows else pd.DataFrame(columns=df.columns)

    # Write cleaned CSV
    try:
        cleaned_df.to_csv(output_file, index=False)
    except Exception as e:
        print(f"Error writing cleaned CSV: {e}", file=sys.stderr)
        sys.exit(1)

    valid_count = len(cleaned_df)
    print(f"\nProcessing complete.")
    print(f"Cleaned CSV written to: {output_file}")
    print(f"Total valid:   {valid_count}")
    print(f"Total invalid: {invalid_count}")


def main(input_file: str):
    """
    Main function to dispatch processing based on file_type.
    
    Args:
        input_file (str): Path to the input file.
    """
    if file_type == "txt":
        process_txt_file(input_file)
    elif file_type == "csv":
        process_csv_file(input_file)
    else:
        print(f"Unsupported file_type: {file_type}. Use 'txt' or 'csv'.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python datawash.py <input_file>", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    if not os.path.isfile(input_path):
        print(f"Error: Input file does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)

    main(input_path)