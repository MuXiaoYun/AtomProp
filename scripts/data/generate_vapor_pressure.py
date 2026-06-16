"""Generate synthetic vapor-pressure training data from SMILES."""

import random

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

from atomprop.paths import GENERATED_DATA_DIR, ensure_output_dirs

SMILES_FILE = "data/pubchem/pubchem-10m.txt.clean"
MAX_MOLECULES = 10000
T_POINTS = 30


def main() -> None:
    ensure_output_dirs()
    output_file = GENERATED_DATA_DIR / "vapor_pressure_estimated.csv"

    with open(SMILES_FILE) as handle:
        smiles_list = [line.strip() for line in handle if line.strip()]

    random.shuffle(smiles_list)

    rows = []
    valid_mol_count = 0

    print("Starting data generation...")

    for smi in tqdm(smiles_list):
        if valid_mol_count >= MAX_MOLECULES:
            break
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            mol_weight = Descriptors.MolWt(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)

            # Empirical critical temperature / pressure estimates for small molecules
            critical_temp = 50 + 2.5 * mol_weight + 10 * hbd
            critical_pressure = 4 + 0.1 * mol_weight + 0.5 * hba

            temp_min = max(150, 0.3 * critical_temp)
            temp_max = 0.9 * critical_temp
            if temp_max <= temp_min:
                continue

            temp_list = np.linspace(temp_min, temp_max, T_POINTS)

            # Antoine equation parameters (rough estimate)
            antoine_a = 8.07131
            antoine_b = 1730.63
            antoine_c = 233.426

            mol_data = []
            for temperature in temp_list:
                temp_celsius = temperature - 273.15
                pressure_mmhg = 10 ** (antoine_a - antoine_b / (temp_celsius + antoine_c))
                if pressure_mmhg <= 0:
                    continue
                pressure_pa = pressure_mmhg * 133.322
                mol_data.append((temperature, pressure_pa))

            if len(mol_data) < 5:
                continue

            for temperature, pressure in mol_data:
                rows.append(
                    {
                        "smiles": smi,
                        "T": float(temperature),
                        "P": float(pressure),
                        "logP": float(np.log(pressure)),
                        "CAS": "",
                    }
                )

            valid_mol_count += 1
        except Exception:
            continue

    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)

    print("\nDone!")
    print(f"Valid molecules: {valid_mol_count}")
    print(f"Total samples: {len(df)}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
