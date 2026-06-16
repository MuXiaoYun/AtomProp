"""Shared utilities for classical baseline benchmarks."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from atomprop.paths import BENCHMARKS_DIR, ensure_output_dirs


def load_property_dataset(
    csv_path: str | os.PathLike,
    value_column_hints: tuple[str, ...] = ("value", "pvcvalue"),
) -> pd.DataFrame:
    """Load a SMILES + experimental value dataset from CSV."""
    csv_path = str(csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = None
    for encoding in ("utf-8", "gbk", "latin1"):
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            break
        except Exception:
            continue

    if df is None:
        raise RuntimeError(f"Cannot read CSV: {csv_path}")

    smiles_col = None
    value_col = None

    for column in df.columns:
        lower = column.lower()
        if "smiles" in lower:
            smiles_col = column
        for hint in value_column_hints:
            if hint in lower or "enthalpy" in lower or "formation" in lower or "hf" in lower:
                value_col = column
                break

    if smiles_col is None or value_col is None:
        if len(df.columns) >= 2:
            smiles_col = df.columns[0]
            value_col = df.columns[1]
        else:
            raise RuntimeError("Cannot find SMILES and value columns")

    data = pd.DataFrame()
    data["smiles"] = df[smiles_col].astype(str)
    data["exp"] = pd.to_numeric(df[value_col], errors="coerce")
    data = data.dropna()
    data = data[data["smiles"].str.len() > 2].reset_index(drop=True)
    return data


def compute_regression_metrics(
    experimental: np.ndarray,
    predicted: np.ndarray,
) -> tuple[float, float, float]:
    """Return R2, MAE, and RMSE."""
    r2 = r2_score(experimental, predicted)
    mae = mean_absolute_error(experimental, predicted)
    rmse = float(np.sqrt(mean_squared_error(experimental, predicted)))
    return r2, mae, rmse


def save_scatter_plot(
    experimental: np.ndarray,
    predicted: np.ndarray,
    out_file: str | os.PathLike,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    colorbar_label: str,
    cmap: str = "coolwarm",
    show: bool = False,
) -> None:
    """Save a parity scatter plot colored by absolute error."""
    plt.figure(figsize=(8, 8))
    error = np.abs(predicted - experimental)
    scatter = plt.scatter(
        experimental,
        predicted,
        c=error,
        cmap=cmap,
        s=50,
        alpha=0.7,
        edgecolors="k",
    )

    min_v = min(experimental.min(), predicted.min())
    max_v = max(experimental.max(), predicted.max())
    margin = (max_v - min_v) * 0.1 if max_v > min_v else 1.0
    plt.plot(
        [min_v - margin, max_v + margin],
        [min_v - margin, max_v + margin],
        "r--",
        label="Ideal fit",
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar(scatter, label=colorbar_label)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    if show:
        plt.show()
    plt.close()


class PropertyBenchmark(ABC):
    """Base class for SMILES-based property baseline benchmarks."""

    name: str = "benchmark"
    unit: str = ""
    value_column_hints: tuple[str, ...] = ("value", "pvcvalue")

    @abstractmethod
    def predict(self, smiles: str) -> tuple[float | None, str | None]:
        """Return (prediction, error_message)."""

    def run_prediction(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
        """Run predictions over a dataset."""
        predictions: list[float] = []
        experimental: list[float] = []
        fail_reasons: dict[str, int] = {}

        for _, row in dataframe.iterrows():
            pred, err = self.predict(row["smiles"])
            if pred is not None:
                predictions.append(pred)
                experimental.append(row["exp"])
            elif err:
                key = err.split(":")[0]
                fail_reasons[key] = fail_reasons.get(key, 0) + 1

        return np.array(experimental), np.array(predictions), fail_reasons

    @abstractmethod
    def plot(
        self,
        experimental: np.ndarray,
        predicted: np.ndarray,
        out_file: str | os.PathLike,
    ) -> None:
        """Save benchmark-specific scatter plot."""

    def analyze(
        self,
        csv_path: str | os.PathLike,
        output_prefix: str | None = None,
        *,
        show_plot: bool = False,
    ) -> dict | None:
        """Load data, run benchmark, save results, and print metrics."""
        ensure_output_dirs()
        prefix = output_prefix or self.name
        out_dir = BENCHMARKS_DIR / prefix
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_out = out_dir / "results.csv"
        plot_out = out_dir / "scatter.png"

        print(f"Loading dataset from {csv_path}...")
        df = load_property_dataset(csv_path, self.value_column_hints)
        print(f"Total molecules: {len(df)}")
        print(f"Running {self.name} benchmark...")

        exp, pred, fail_reasons = self.run_prediction(df)
        success = len(pred)
        print("\nPrediction summary")
        print(f"Successful: {success}")
        print(f"Failed: {len(df) - success}")

        if fail_reasons:
            print("\nFailure reasons:")
            for reason, count in sorted(fail_reasons.items(), key=lambda item: -item[1]):
                print(f"  {reason}: {count}")

        if success < 5:
            print("Too few successful predictions to compute statistics.")
            return None

        r2, mae, rmse = compute_regression_metrics(exp, pred)
        print("\nStatistics")
        print(f"R2  : {round(r2, 4)}")
        print(f"MAE : {round(mae, 2)} {self.unit}")
        print(f"RMSE: {round(rmse, 2)} {self.unit}")

        result_df = pd.DataFrame(
            {"Experimental": exp, "Predicted": pred, "Error": pred - exp}
        )
        result_df.to_csv(csv_out, index=False)
        print(f"\nSaved results: {csv_out}")

        self.plot(exp, pred, plot_out)
        print(f"Saved plot: {plot_out}")

        if show_plot:
            plt.show()

        return {"r2": r2, "mae": mae, "rmse": rmse, "n": success}
