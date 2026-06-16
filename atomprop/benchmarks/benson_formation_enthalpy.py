"""Benson group-contribution benchmark for ideal-gas formation enthalpy."""

from __future__ import annotations

import warnings

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from atomprop.benchmarks.base import PropertyBenchmark, save_scatter_plot
from atomprop.paths import FORMATION_ENTHALPY_CSV, LEGACY_FORMATION_ENTHALPY_CSV, resolve_data_path

warnings.filterwarnings("ignore")

# Benson group increments (kcal/mol). Subset covering common C, H, O, N environments.
# Reference: NIST / Benson group increment tables.
GROUP_CONTRIBUTIONS = {
    "C-(H)3(C)": -10.20,
    "C-(H)2(C)2": -4.93,
    "C-(H)(C)3": -1.90,
    "C-(C)4": 0.50,
    "C-(H)3(Cd)": -10.00,
    "C-(H)2(C)1(Cd)": -4.76,
    "C-(H)3(Cb)": -10.03,
    "Cd-(H)2": 6.26,
    "Cd-(H)(C)": 8.59,
    "Cd-(C)2": 10.34,
    "Cd-(C)(Cb)": 9.80,
    "Ct-(C)": 27.3,
    "Ct-(Ct)": 30.0,
    "Cb-H": 3.30,
    "Cb-(C)": 3.00,
    "Cb-(Cb)": 3.00,
    "O-(H)(C)": -37.90,
    "O-(C)2": -18.00,
    "CO-(H)(C)": -25.00,
    "CO-(C)2": -31.00,
    "COO-(H)(C)": -92.0,
    "COO-(C)2": -85.0,
    "N-(H)2(C)": -10.0,
    "N-(H)(C)2": -4.5,
    "N-(C)3": 0.0,
}

KCAL_TO_KJ = 4.184


class BensonGroupAnalyzer:
    """Identify Benson groups in a molecule and sum formation enthalpy contributions."""

    def analyze(self, mol) -> tuple[float | None, str]:
        if mol is None:
            return None, "Invalid Molecule"

        total_h_kcal = 0.0

        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            num_h = atom.GetTotalNumHs()
            neighbors = atom.GetNeighbors()
            num_heavy = len(neighbors)
            hybridization = atom.GetHybridization()
            group_key = None

            if symbol == "C":
                if hybridization == Chem.rdchem.HybridizationType.SP3:
                    if num_heavy == 1:
                        group_key = "C-(H)3(C)"
                    elif num_heavy == 2:
                        group_key = "C-(H)2(C)2"
                    elif num_heavy == 3:
                        group_key = "C-(H)(C)3"
                    elif num_heavy == 4:
                        group_key = "C-(C)4"
                elif hybridization == Chem.rdchem.HybridizationType.SP2:
                    if atom.GetIsAromatic():
                        group_key = "Cb-H" if num_h > 0 else "Cb-(C)"
                    else:
                        is_double = any(
                            bond.GetBondType() == Chem.rdchem.BondType.DOUBLE
                            for bond in atom.GetBonds()
                        )
                        if is_double:
                            if num_h == 2:
                                group_key = "Cd-(H)2"
                            elif num_h == 1:
                                group_key = "Cd-(H)(C)"
                            elif num_h == 0:
                                group_key = "Cd-(C)2"

            elif symbol == "O":
                if num_h > 0:
                    group_key = "O-(H)(C)"
                elif num_heavy == 2:
                    group_key = "O-(C)2"

            elif symbol == "N":
                if hybridization == Chem.rdchem.HybridizationType.SP3:
                    if num_h == 2:
                        group_key = "N-(H)2(C)"
                    elif num_h == 1:
                        group_key = "N-(H)(C)2"
                    elif num_h == 0:
                        group_key = "N-(C)3"

            if group_key and group_key in GROUP_CONTRIBUTIONS:
                total_h_kcal += GROUP_CONTRIBUTIONS[group_key]

        return total_h_kcal * KCAL_TO_KJ, "OK"


class BensonFormationEnthalpyBenchmark(PropertyBenchmark):
    """Predict ideal-gas formation enthalpy using Benson group increments."""

    name = "benson_formation_enthalpy"
    unit = "kJ/mol"
    value_column_hints = ("enthalpy", "formation", "hf", "value", "pvcvalue")

    def __init__(self) -> None:
        self._analyzer = BensonGroupAnalyzer()

    def predict(self, smiles: str) -> tuple[float | None, str | None]:
        if not isinstance(smiles, str) or not smiles.strip():
            return None, "Invalid SMILES"

        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None, "RDKit parse failed"

        value, status = self._analyzer.analyze(mol)
        if value is None:
            return None, status
        return value, None

    def plot(self, experimental, predicted, out_file) -> None:
        save_scatter_plot(
            experimental,
            predicted,
            out_file,
            xlabel=r"Experimental $\Delta H_f^\circ$ (kJ/mol)",
            ylabel=r"Predicted $\Delta H_f^\circ$ (kJ/mol)",
            title="Ideal Gas Formation Enthalpy\n(Benson Group Contribution)",
            colorbar_label="Absolute Error (kJ/mol)",
            cmap="viridis",
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Benson group-contribution formation enthalpy benchmark"
    )
    parser.add_argument(
        "--csv",
        default=str(resolve_data_path(FORMATION_ENTHALPY_CSV, LEGACY_FORMATION_ENTHALPY_CSV)),
        help="Path to CSV with SMILES and formation enthalpy values",
    )
    parser.add_argument(
        "--output-prefix",
        default="benson_formation_enthalpy",
        help="Output subdirectory name under outputs/benchmarks/",
    )
    parser.add_argument("--show-plot", action="store_true")
    args = parser.parse_args()

    benchmark = BensonFormationEnthalpyBenchmark()
    benchmark.analyze(args.csv, output_prefix=args.output_prefix, show_plot=args.show_plot)


if __name__ == "__main__":
    main()
