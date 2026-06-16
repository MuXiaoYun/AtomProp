"""Joback group-contribution benchmark for boiling point prediction."""

from __future__ import annotations

import warnings

import numpy as np
from rdkit import Chem, RDLogger
from thermo.group_contribution.joback import Joback

from atomprop.benchmarks.base import PropertyBenchmark, save_scatter_plot
from atomprop.paths import BOILING_POINT_CSV, LEGACY_BOILING_POINT_CSV, resolve_data_path

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


class JobackBoilingPointBenchmark(PropertyBenchmark):
    """Predict boiling points using the Joback group-contribution method."""

    name = "joback_boiling_point"
    unit = "K"
    value_column_hints = ("pvcvalue", "value", "boiling")

    def predict(self, smiles: str) -> tuple[float | None, str | None]:
        if not isinstance(smiles, str):
            return None, "Invalid SMILES type"

        smiles = smiles.strip()
        if smiles == "":
            return None, "Empty SMILES"

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, "RDKit parse failed"

        try:
            joback = Joback(smiles)
            if joback.status != "OK":
                return None, f"Fragmentation failed: {joback.status}"

            boiling_point = joback.estimate(callables=False).get("Tb")
            if boiling_point is None or np.isnan(boiling_point):
                return None, "Tb unavailable"

            return float(boiling_point), None
        except Exception as exc:
            return None, str(exc)

    def plot(self, experimental, predicted, out_file) -> None:
        save_scatter_plot(
            experimental,
            predicted,
            out_file,
            xlabel="Experimental BP (K)",
            ylabel="Predicted BP (K)",
            title="Joback Boiling Point Prediction",
            colorbar_label="Absolute Error (K)",
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Joback boiling point benchmark")
    parser.add_argument(
        "--csv",
        default=str(resolve_data_path(BOILING_POINT_CSV, LEGACY_BOILING_POINT_CSV)),
        help="Path to CSV with SMILES and boiling point values",
    )
    parser.add_argument(
        "--output-prefix",
        default="joback_boiling_point",
        help="Output subdirectory name under outputs/benchmarks/",
    )
    parser.add_argument("--show-plot", action="store_true")
    args = parser.parse_args()

    benchmark = JobackBoilingPointBenchmark()
    benchmark.analyze(args.csv, output_prefix=args.output_prefix, show_plot=args.show_plot)


if __name__ == "__main__":
    main()
