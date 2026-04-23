import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm
import random

SMILES_FILE = "data/pubchem/pubchem-10m.txt.clean"
OUTPUT_FILE = "thermo_vapor_pressure_estimated.csv"
MAX_MOLECULES = 10000
T_POINTS = 30

# =========================
# 读取 SMILES
with open(SMILES_FILE) as f:
    smiles_list = [line.strip() for line in f if line.strip()]

random.shuffle(smiles_list)

data = []
valid_mol_count = 0

print("Starting data generation...")

for smi in tqdm(smiles_list):
    if valid_mol_count >= MAX_MOLECULES:
        break
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        # 基本分子描述符
        MW = Descriptors.MolWt(mol)
        HBD = Descriptors.NumHDonors(mol)
        HBA = Descriptors.NumHAcceptors(mol)
        TPSA = Descriptors.TPSA(mol)

        # 简单 Tc/Pc 经验估算（可以覆盖绝大多数小分子）
        Tc = 50 + 2.5 * MW + 10 * HBD  # K
        Pc = 4 + 0.1 * MW + 0.5 * HBA  # bar

        Tmin = max(150, 0.3 * Tc)
        Tmax = 0.9 * Tc
        if Tmax <= Tmin:
            continue

        T_list = np.linspace(Tmin, Tmax, T_POINTS)

        # Antoine 经验参数 (粗略估算)
        A = 8.07131
        B = 1730.63
        C = 233.426

        mol_data = []
        for T in T_list:
            # 转换为 Celsius 用 Antoine 方程
            T_C = T - 273.15
            P_mmHg = 10 ** (A - B / (T_C + C))
            if P_mmHg <= 0:
                continue
            P_Pa = P_mmHg * 133.322  # 转 Pa
            mol_data.append((T, P_Pa))

        if len(mol_data) < 5:
            continue

        for T, P in mol_data:
            data.append({
                "smiles": smi,
                "T": float(T),
                "P": float(P),
                "logP": float(np.log(P)),
                "CAS": ""
            })

        valid_mol_count += 1

    except:
        continue

df = pd.DataFrame(data)
df.to_csv(OUTPUT_FILE, index=False)

print("\nDone!")
print(f"Valid molecules: {valid_mol_count}")
print(f"Total samples: {len(df)}")
print(f"Saved to: {OUTPUT_FILE}")