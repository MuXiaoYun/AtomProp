"""Fine-tune a thermodynamic regression head on vapor-pressure data."""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

from atomprop.models.heads import ThermoRegressionHead
from atomprop.paths import GENERATED_DATA_DIR, MODELS_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GEAT_DIM = 768
HIDDEN_DIM = 256
DEFAULT_TASK_NAME = "thermo_vp_finetune"


class ThermoDataset(Dataset):
    """Dataset for temperature-conditioned vapor-pressure regression."""

    def __init__(self, dataframe, t_min=None, t_max=None):
        self.data = dataframe
        self.T_min = t_min if t_min is not None else self.data["T"].min()
        self.T_max = t_max if t_max is not None else self.data["T"].max()
        if self.T_max == self.T_min:
            self.T_max += 1e-6
        self.targets = np.log(self.data["P"].values)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        temp_raw = self.data.iloc[index]["T"]
        temp_norm = (temp_raw - self.T_min) / (self.T_max - self.T_min)
        return {
            "smiles": self.data.iloc[index]["smiles"],
            "T_norm": torch.tensor(temp_norm, dtype=torch.float),
            "target": torch.tensor(self.targets[index], dtype=torch.float),
        }


def get_geat_embedding(smiles_list, batch_size=32):
    """Placeholder GeAT embeddings until real inference is wired in."""
    print(f"Generating embeddings for {len(smiles_list)} molecules...")
    embeddings = []
    for start in range(0, len(smiles_list), batch_size):
        batch_size_actual = min(batch_size, len(smiles_list) - start)
        embeddings.append(torch.rand(batch_size_actual, GEAT_DIM))
    return torch.cat(embeddings, dim=0)


def create_data_loaders(args, train_df, val_df, test_df):
    all_smiles = list(
        set(
            train_df["smiles"].tolist()
            + val_df["smiles"].tolist()
            + test_df["smiles"].tolist()
        )
    )
    smiles_to_idx = {smi: idx for idx, smi in enumerate(all_smiles)}
    geat_embeddings = get_geat_embedding(all_smiles)

    class ThermoDatasetWithEmbed(ThermoDataset):
        def __init__(self, dataframe, geat_embs, mapping, t_min=None, t_max=None):
            super().__init__(dataframe, t_min, t_max)
            self.geat_embs = geat_embs
            self.smiles_to_idx = mapping

        def __getitem__(self, index):
            item = super().__getitem__(index)
            smiles = self.data.iloc[index]["smiles"]
            item["geat_emb"] = self.geat_embs[self.smiles_to_idx[smiles]]
            return item

    global_t_min = min(train_df["T"].min(), val_df["T"].min(), test_df["T"].min())
    global_t_max = max(train_df["T"].max(), val_df["T"].max(), test_df["T"].max())

    train_dataset = ThermoDatasetWithEmbed(
        train_df, geat_embeddings, smiles_to_idx, global_t_min, global_t_max
    )
    val_dataset = ThermoDatasetWithEmbed(
        val_df, geat_embeddings, smiles_to_idx, global_t_min, global_t_max
    )
    test_dataset = ThermoDatasetWithEmbed(
        test_df, geat_embeddings, smiles_to_idx, global_t_min, global_t_max
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def train_and_evaluate(args):
    print(f"Loading data from {args.data_file}...")
    df = pd.read_csv(args.data_file)

    if 0 < args.max_samples < len(df):
        print(f"Debug mode: sampling {args.max_samples} rows...")
        df = df.sample(n=args.max_samples, random_state=42).reset_index(drop=True)

    train_df = df.sample(frac=0.8, random_state=42)
    temp_df = df.drop(train_df.index)
    val_df = temp_df.sample(frac=0.5, random_state=42)
    test_df = temp_df.drop(val_df.index)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_loader, val_loader, test_loader = create_data_loaders(args, train_df, val_df, test_df)

    model = ThermoRegressionHead(input_dim=GEAT_DIM + 1, hidden_dim=HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    task_name = args.task_name or DEFAULT_TASK_NAME
    save_dir = MODELS_DIR / task_name
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_metrics = {}
    print(f"Starting training on {DEVICE}...")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            geat_emb = batch["geat_emb"].to(DEVICE)
            temp_norm = batch["T_norm"].to(DEVICE)
            targets = batch["target"].to(DEVICE)
            features = torch.cat([geat_emb, temp_norm.unsqueeze(1)], dim=1)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for batch in val_loader:
                geat_emb = batch["geat_emb"].to(DEVICE)
                temp_norm = batch["T_norm"].to(DEVICE)
                targets = batch["target"].to(DEVICE)
                features = torch.cat([geat_emb, temp_norm.unsqueeze(1)], dim=1)
                preds = model(features)
                val_loss += criterion(preds, targets).item()
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        r2 = r2_score(all_targets, all_preds)
        rmse = math.sqrt(mean_squared_error(all_targets, all_preds))

        print(
            f"Epoch {epoch + 1:2d} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val R²: {r2:.4f} | Val RMSE: {rmse:.4f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_metrics = {
                "epoch": epoch + 1,
                "val_loss": avg_val_loss,
                "val_r2": r2,
                "val_rmse": rmse,
            }
            save_path = save_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "val_r2": r2,
                    "scaler_stats": {
                        "T_min": train_loader.dataset.T_min,
                        "T_max": train_loader.dataset.T_max,
                    },
                    "args": vars(args),
                },
                save_path,
            )
            print("   -> Best model saved (validation loss improved)")

    print("\n" + "=" * 50)
    print("Training completed. Best validation metrics:")
    print(json.dumps(best_metrics, indent=2))
    del test_loader


if __name__ == "__main__":
    default_data = GENERATED_DATA_DIR / "vapor_pressure_estimated.csv"
    parser = argparse.ArgumentParser(
        description="Fine-tune a thermodynamic head for vapor-pressure prediction"
    )
    parser.add_argument("--data_file", type=str, default=str(default_data))
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--task_name", type=str, default=None)
    train_and_evaluate(parser.parse_args())
