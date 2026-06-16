"""Plot vapor-pressure curves for a molecule using a fine-tuned thermo head."""

import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.metrics import mean_squared_error, r2_score

from atomprop.models.heads import ThermoRegressionHead
from atomprop.paths import FIGURES_DIR, MODELS_DIR, ensure_output_dirs

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GEAT_DIM = 768
HIDDEN_DIM = 256


def get_molecule_properties(smiles: str) -> dict | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol_weight = Descriptors.MolWt(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    critical_temp = 50 + 2.5 * mol_weight + 10 * hbd
    temp_min = max(150, 0.3 * critical_temp)
    temp_max = 0.9 * critical_temp

    if temp_max <= temp_min:
        return None

    return {
        "MW": mol_weight,
        "HBD": hbd,
        "HBA": hba,
        "Tc": critical_temp,
        "Tmin": temp_min,
        "Tmax": temp_max,
    }


def generate_antoine_data(props: dict) -> tuple[np.ndarray, np.ndarray]:
    temp_list = np.linspace(props["Tmin"], props["Tmax"], 100)
    antoine_a, antoine_b, antoine_c = 8.07131, 1730.63, 233.426

    temperatures = []
    pressures = []
    for temperature in temp_list:
        temp_celsius = temperature - 273.15
        if (temp_celsius + antoine_c) <= 0:
            continue
        pressure_mmhg = 10 ** (antoine_a - antoine_b / (temp_celsius + antoine_c))
        if pressure_mmhg > 0:
            pressures.append(pressure_mmhg * 133.322)
            temperatures.append(temperature)

    return np.array(temperatures), np.array(pressures)


def get_geat_embedding_single(smiles: str) -> torch.Tensor:
    """Placeholder embedding until real GeAT inference is wired in."""
    del smiles
    return torch.rand(1, GEAT_DIM)


def predict_curve(model, smiles, temp_list, scaler_stats):
    geat_emb = get_geat_embedding_single(smiles).to(DEVICE)
    temp_min = scaler_stats["T_min"]
    temp_max = scaler_stats["T_max"]
    temp_norm = (temp_list - temp_min) / (temp_max - temp_min)
    temp_tensor = torch.tensor(temp_norm, dtype=torch.float, device=DEVICE).unsqueeze(1)
    geat_emb_expanded = geat_emb.expand(len(temp_list), -1)
    combined_features = torch.cat([geat_emb_expanded, temp_tensor], dim=1)

    model.eval()
    with torch.no_grad():
        log_pressure = model(combined_features)
        return torch.exp(log_pressure).cpu().numpy()


def main(args) -> None:
    ensure_output_dirs()
    model_path = MODELS_DIR / args.task_name / "best_model.pth"
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

    model = ThermoRegressionHead(input_dim=GEAT_DIM + 1, hidden_dim=HIDDEN_DIM).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    scaler_stats = checkpoint["scaler_stats"]

    props = get_molecule_properties(args.smiles)
    if not props:
        print("Error: Invalid SMILES")
        return

    print(f"Molecule: {args.smiles}, MW: {props['MW']:.2f}, Tc: {props['Tc']:.2f} K")

    temp_gt, pressure_gt = generate_antoine_data(props)
    pressure_pred = predict_curve(model, args.smiles, temp_gt, scaler_stats)

    valid_mask = (pressure_gt > 1) & (pressure_pred > 1)
    if np.sum(valid_mask) < 5:
        print("Warning: Too few valid points for metric calculation.")
        r2 = float("nan")
        rmse = float("nan")
    else:
        r2 = r2_score(pressure_gt[valid_mask], pressure_pred[valid_mask])
        rmse = math.sqrt(mean_squared_error(pressure_gt[valid_mask], pressure_pred[valid_mask]))

    print("-" * 30)
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE:     {rmse:.4f}")
    print("-" * 30)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(temp_gt, pressure_gt, label="Estimated (Antoine)", linestyle="--", color="blue")
    plt.plot(temp_gt, pressure_pred, label="GeAT Prediction", linestyle="-", color="red")
    plt.title(f"Vapor Pressure (Linear Scale)\nSMILES: {args.smiles}")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Pressure (Pa)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.semilogy(temp_gt, pressure_gt, label="Estimated (Antoine)", linestyle="--", color="blue")
    plt.semilogy(temp_gt, pressure_pred, label="GeAT Prediction", linestyle="-", color="red")
    plt.title(f"Vapor Pressure (Log Scale)\nSMILES: {args.smiles}")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Pressure (Pa) [Log Scale]")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)

    plt.tight_layout()
    safe_smiles = args.smiles.replace("/", "_").replace("\\", "_")
    output_img = FIGURES_DIR / f"vapor_pressure_{safe_smiles}.png"
    plt.savefig(output_img, dpi=300)
    print(f"Plot saved to {output_img}")
    if args.show_plot:
        plt.show()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot vapor pressure curves for one SMILES")
    parser.add_argument("--smiles", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="thermo_vp_finetune")
    parser.add_argument("--show-plot", action="store_true")
    main(parser.parse_args())
