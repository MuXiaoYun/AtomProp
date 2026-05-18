import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# chemistry
from rdkit import Chem
from rdkit import RDLogger
from thermo.group_contribution.joback import Joback

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


# =========================================================
# Joback Predictor
# =========================================================

class JobackPredictor:
    """
    使用 thermo Joback 方法预测沸点
    """

    def predict(self, smiles):

        if not isinstance(smiles, str):
            return None, "Invalid SMILES type"

        smiles = smiles.strip()

        if smiles == "":
            return None, "Empty SMILES"

        # RDKit 检查
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            return None, "RDKit parse failed"

        try:
            J = Joback(smiles)

            if J.status != "OK":
                return None, f"Fragmentation failed: {J.status}"

            result = J.estimate(callables=False)

            Tb = result.get("Tb")

            if Tb is None or np.isnan(Tb):
                return None, "Tb unavailable"

            return float(Tb), None

        except Exception as e:
            return None, str(e)


# =========================================================
# CSV Loader
# =========================================================

def load_dataset(csv_path):

    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = None

    for enc in ["utf-8", "gbk", "latin1"]:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except:
            pass

    if df is None:
        raise RuntimeError("Cannot read CSV")

    # 自动寻找列
    smiles_col = None
    value_col = None

    for c in df.columns:

        lc = c.lower()

        if "smiles" in lc:
            smiles_col = c

        if "pvcvalue" in lc or "value" in lc:
            value_col = c

    if smiles_col is None or value_col is None:
        raise RuntimeError("Cannot find required columns")

    data = pd.DataFrame()

    data["smiles"] = df[smiles_col].astype(str)
    data["exp"] = pd.to_numeric(df[value_col], errors="coerce")

    data = data.dropna()

    data = data[data["smiles"].str.len() > 2]

    data = data.reset_index(drop=True)

    return data


# =========================================================
# Prediction loop
# =========================================================

def run_prediction(df):

    predictor = JobackPredictor()

    predictions = []
    experimental = []

    fail_reasons = {}

    for i, row in df.iterrows():

        smiles = row["smiles"]
        exp = row["exp"]

        pred, err = predictor.predict(smiles)

        if pred is not None:

            predictions.append(pred)
            experimental.append(exp)

        else:

            key = err.split(":")[0]

            fail_reasons[key] = fail_reasons.get(key, 0) + 1

    return np.array(experimental), np.array(predictions), fail_reasons


# =========================================================
# Statistics
# =========================================================

def compute_metrics(exp, pred):

    r2 = r2_score(exp, pred)

    mae = mean_absolute_error(exp, pred)

    rmse = np.sqrt(mean_squared_error(exp, pred))

    return r2, mae, rmse


# =========================================================
# Plot
# =========================================================

def plot_results(exp, pred, out_file):

    plt.figure(figsize=(8, 8))

    error = np.abs(pred - exp)

    sc = plt.scatter(
        exp,
        pred,
        c=error,
        cmap="coolwarm",
        s=50,
        alpha=0.7,
        edgecolor="k"
    )

    min_v = min(exp.min(), pred.min())
    max_v = max(exp.max(), pred.max())

    plt.plot([min_v, max_v], [min_v, max_v], "r--")

    plt.xlabel("Experimental BP (K)")
    plt.ylabel("Predicted BP (K)")
    plt.title("Joback Boiling Point Prediction")

    plt.colorbar(sc, label="Absolute Error (K)")

    plt.tight_layout()

    plt.savefig(out_file, dpi=300)

    plt.show()


# =========================================================
# Main pipeline
# =========================================================

def analyze(csv_path, output_prefix="joback"):

    print("Loading dataset...")
    df = load_dataset(csv_path)

    print("Total molecules:", len(df))

    print("Running Joback prediction...")

    exp, pred, fail = run_prediction(df)

    success = len(pred)

    print("\nPrediction summary")

    print("Successful:", success)
    print("Failed:", len(df) - success)

    if fail:
        print("\nFailure reasons:")
        for k, v in sorted(fail.items(), key=lambda x: -x[1]):
            print(k, v)

    if success < 5:
        print("Too few predictions")
        return

    r2, mae, rmse = compute_metrics(exp, pred)

    print("\nStatistics")
    print("R2  :", round(r2, 4))
    print("MAE :", round(mae, 2), "K")
    print("RMSE:", round(rmse, 2), "K")

    # 保存结果

    result_df = pd.DataFrame(
        {
            "Experimental": exp,
            "Predicted": pred,
            "Error": pred - exp
        }
    )

    out_csv = output_prefix + "_results.csv"

    result_df.to_csv(out_csv, index=False)

    print("\nSaved:", out_csv)

    # plot

    plot_file = output_prefix + "_scatter.png"

    plot_results(exp, pred, plot_file)

    print("Saved:", plot_file)


# =========================================================
# Entry
# =========================================================

if __name__ == "__main__":

    csv_file = "./data/data/沸点.csv"

    analyze(csv_file)