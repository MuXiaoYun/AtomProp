#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
finetune_sei.py — Finetune AtomProp on SEI datasets (偏心因子 & 熔点).

Usage:
    python finetune_sei.py                          # finetune on both datasets
    python finetune_sei.py --dataset eccentric      # only 偏心因子
    python finetune_sei.py --dataset melting        # only 熔点
    python finetune_sei.py --epochs 200 --bs 64     # custom params

Output:
    trained_models/sei_eccentric_factor/best_model.pth   ← Web UI compatible
    trained_models/sei_melting_point/best_model.pth      ← Web UI compatible
"""

from __future__ import annotations

import os
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# ── project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import configs.config_reg as cfg  # noqa: E402

from atomprop.dataloader.dataloader import SMILESToInputs  # noqa: E402
from atomprop.models.gnns import Embedder, GNNAggr, MaskedMSELoss  # noqa: E402
from atomprop.models.geat import GeATNet  # noqa: E402
from atomprop.utils.mlp import MLP  # noqa: E402
from atomprop.utils.utils import remove_module_prefix  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

# ── helpers ───────────────────────────────────────────────────────────────────

def _read_csv_safe(path: str | Path) -> pd.DataFrame:
    """Read CSV with automatic encoding detection (gbk → utf-8 → latin-1)."""
    for enc in ("gbk", "gb2312", "utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Cannot decode CSV: {path}")


def create_dataset(smiles_list: list[str], labels: np.ndarray) -> list[Data]:
    """Convert SMILES + labels → PyG Data list."""
    dataset = []
    for smi, label in zip(smiles_list, labels):
        atom_info, edge_info, _ = SMILESToInputs.convert(smi, sanitize=False)
        if atom_info is None or edge_info is None:
            continue
        if edge_info.dim() == 2 and edge_info.size(1) == 4:
            edge_index = edge_info[:, :2].t().contiguous()
            edge_attr = edge_info[:, 2:]
        else:
            edge_index = torch.tensor([[], []], dtype=torch.long)
            edge_attr = torch.tensor([], dtype=torch.long).view(0, 2)
        y = torch.tensor(label, dtype=torch.float)
        dataset.append(Data(x=atom_info, edge_index=edge_index, edge_attr=edge_attr, y=y))
    if not dataset:
        raise RuntimeError("No valid molecules from SMILES list.")
    return dataset


def compute_scaler_stats(labels: np.ndarray) -> dict:
    """Compute per-task mean & std, ignoring missing values (-1)."""
    n_tasks = labels.shape[1]
    means, scales = [], []
    for col in range(n_tasks):
        col_vals = labels[:, col]
        valid = col_vals[col_vals != -1]
        if len(valid) == 0:
            mean_v, scale_v = 0.0, 1.0
        elif len(valid) == 1:
            mean_v, scale_v = float(valid[0]), 1.0
        else:
            mean_v = float(np.mean(valid))
            scale_v = float(np.std(valid)) or 1.0
        means.append(mean_v)
        scales.append(scale_v)
    return {"mean": np.array(means, dtype=np.float32),
            "scale": np.array(scales, dtype=np.float32),
            "n_tasks": n_tasks}


def transform(labels: np.ndarray, stats: dict) -> np.ndarray:
    """Standardisation: (x - mean) / scale.  Missing (-1) is kept as -1."""
    labels = labels.copy()
    mean, scale = stats["mean"], stats["scale"]
    scale = np.where(scale == 0.0, 1.0, scale)
    valid = labels != -1
    return np.where(valid, (labels - mean) / scale, labels).astype(np.float32)


def inverse_transform(scaled: np.ndarray, stats: dict) -> np.ndarray:
    """Reverse standardisation."""
    return scaled * stats["scale"] + stats["mean"]


def build_checkpoint(
    embedding_layer: nn.Module,
    backbone: GeATNet,
    head: nn.Module,
    aggrmodel: nn.Module | None,
    scaler_stats: dict,
    target_name: str,
    epoch: int,
    val_loss: float | None = None,
) -> dict:
    """Build a Web‑UI compatible checkpoint dict."""
    ckpt: dict = {
        "epoch": epoch,
        "embedding_layer_state_dict": embedding_layer.state_dict(),
        "head_state_dict": head.state_dict(),
        "aggr_state_dict": aggrmodel.state_dict() if aggrmodel is not None else None,
        "scaler_stats": scaler_stats,
        "target_column": target_name,
        "task_type": "regression",
    }
    if val_loss is not None:
        ckpt["val_loss"] = val_loss

    if cfg.use_lora:
        ckpt["lora_state_dict"] = backbone.get_lora_state_dict()
        ckpt["lora_config"] = {
            "rank": cfg.lora_rank,
            "alpha": cfg.lora_alpha,
            "dropout": cfg.lora_dropout,
            "include_ffn": cfg.lora_include_ffn,
            "include_global_attn": cfg.lora_include_global_attn,
        }
    else:
        ckpt["backbone_state_dict"] = backbone.state_dict()
    return ckpt


# ── main training ─────────────────────────────────────────────────────────────

def finetune_dataset(
    csv_path: str,
    output_dir: str,
    target_name: str,
    smiles_col: str = "smi",
    value_col: str = "pvcValue",
    num_epochs: int | None = None,
    batch_size: int | None = None,
    device: str = "cuda:0",
) -> str:
    """
    Full‑dataset finetune, save a Web‑UI compatible .pth.

    Returns
    -------
    str : path to the saved checkpoint.
    """
    # ── overrides ─────────────────────────────────────────────────────────
    if num_epochs is not None:
        cfg.num_epochs = num_epochs
    if batch_size is not None:
        cfg.batch_size = batch_size

    os.makedirs(output_dir, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}")

    # ── load data ─────────────────────────────────────────────────────────
    df = _read_csv_safe(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"  Columns: {df.columns.tolist()}")

    smiles_list = df[smiles_col].astype(str).tolist()
    raw_labels = df[[value_col]].fillna(-1).values.astype(np.float32)

    scaler_stats = compute_scaler_stats(raw_labels)
    labels_scaled = transform(raw_labels, scaler_stats)

    print(f"  Target: {value_col}  mean={scaler_stats['mean'][0]:.4f}  "
          f"std={scaler_stats['scale'][0]:.4f}")

    dataset = create_dataset(smiles_list, labels_scaled)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=Batch.from_data_list,
        num_workers=0,
    )
    print(f"  Valid molecules: {len(dataset)}  Batches: {len(loader)}")

    # ── build model ───────────────────────────────────────────────────────
    embedding_layer = Embedder(num_atom_types=120, embed_dim=cfg.embed_dim).to(dev)
    backbone = GeATNet(
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        global_num_heads=cfg.global_num_heads,
        output_negative_slope=cfg.output_negative_slope,
        dropout=cfg.geat_dropout,
        geat_num_layers=cfg.geat_num_layers,
        aggr_num_layers=cfg.aggr_num_layers,
        FFN_type=cfg.FFN_type,
        FFN_hidden_dim=cfg.FFN_hidden_dim,
        FFN_num_experts=cfg.FFN_num_experts,
        FFN_num_layers=cfg.FFN_num_layers,
        FFN_top_k=cfg.FFN_top_k,
        use_edge_embedding=cfg.use_edge_embedding,
    ).to(dev)

    aggrmodel = GNNAggr(embed_dim=cfg.embed_dim, aggr=cfg.aggr, layers=1).to(dev)
    head = MLP(
        input_dim=cfg.embed_dim,
        hidden_dim=cfg.head_hidden_dim,
        output_dim=1,  # single target value
        num_layers=cfg.head_layers,
        dropout=cfg.head_dropout,
        batch_norm=True,
        output_activation=None,
    ).to(dev)

    # ── load pretrained weights ───────────────────────────────────────────
    if not cfg.no_pretrain:
        pretrained_path = PROJECT_ROOT / cfg.pretrained_path
        if pretrained_path.is_file():
            ckpt = torch.load(pretrained_path, map_location=dev, weights_only=False)
            embedding_layer.load_state_dict(
                remove_module_prefix(ckpt["embedding_layer_state_dict"]))
            backbone_state = remove_module_prefix(ckpt["backbone_state_dict"])
            try:
                backbone.load_state_dict(backbone_state, strict=True)
            except RuntimeError:
                print("[WARNING] Strict load failed, using non-strict mode...")
                backbone.load_state_dict(backbone_state, strict=False)
            print(f"  Pretrained weights loaded from {pretrained_path}")
        else:
            print(f"  [WARNING] Pretrained model not found at {pretrained_path}")

    # ── LoRA ──────────────────────────────────────────────────────────────
    if cfg.use_lora:
        backbone.apply_lora(
            rank=cfg.lora_rank,
            alpha=cfg.lora_alpha,
            dropout=cfg.lora_dropout,
            include_ffn=cfg.lora_include_ffn,
            include_global_attn=cfg.lora_include_global_attn,
        )
        backbone = backbone.to(dev)
        lora_n = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        print(f"  LoRA applied (rank={cfg.lora_rank}, trainable backbone params: {lora_n:,})")

    total_params = sum(p.numel() for p in embedding_layer.parameters() if p.requires_grad)
    total_params += sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    total_params += sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"  Total trainable params: {total_params:,}")

    # ── optimiser & scheduler ─────────────────────────────────────────────
    backbone_trainable = [p for p in backbone.parameters() if p.requires_grad]
    opt_emb_backbone = torch.optim.Adam([
        {"params": embedding_layer.parameters(), "lr": cfg.lr_embedding_layer_backbone},
        {"params": backbone_trainable, "lr": cfg.lr_embedding_layer_backbone},
    ], weight_decay=cfg.wd_emb_backbone)
    opt_head = torch.optim.Adam(
        head.parameters(), lr=cfg.lr_head, weight_decay=cfg.wd_head
    )
    sched_emb_backbone = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_emb_backbone, T_max=cfg.num_epochs,
        eta_min=cfg.eta_min_embedding_layer_backbone,
    )
    sched_head = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_head, T_max=cfg.num_epochs, eta_min=cfg.eta_min_head,
    )

    criterion = MaskedMSELoss()

    # ── training loop ─────────────────────────────────────────────────────
    best_loss = float("inf")
    best_path = os.path.join(output_dir, "best_model.pth")
    final_path = os.path.join(output_dir, "final_model.pth")

    print(f"\n{'='*50}\nTraining on: {target_name}  ({len(dataset)} mols, {cfg.num_epochs} epochs)\n{'='*50}")

    for epoch in range(cfg.num_epochs):
        embedding_layer.train()
        backbone.train()
        head.train()
        if cfg.aggr == "attention":
            aggrmodel.train()

        epoch_loss = 0.0
        n_batches = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")
        for batch in pbar:
            batch = batch.to(dev)
            opt_emb_backbone.zero_grad()
            opt_head.zero_grad()

            emb = embedding_layer(batch.x.squeeze())
            emb = backbone(
                Data(x=emb, edge_index=batch.edge_index, edge_attr=batch.edge_attr),
                batch=batch.batch,
            )
            g_emb = aggrmodel(emb, batch.batch)
            preds = head(g_emb)
            loss = criterion(preds, batch.y.reshape(-1, 1))
            loss.backward()
            opt_emb_backbone.step()
            opt_head.step()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_loss = epoch_loss / max(n_batches, 1)
        sched_emb_backbone.step()
        sched_head.step()

        print(f"  Epoch {epoch+1:3d}  avg loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt = build_checkpoint(
                embedding_layer, backbone, head, aggrmodel,
                scaler_stats, target_name, epoch + 1, val_loss=avg_loss,
            )
            torch.save(ckpt, best_path)
            print(f"  → best model saved ({best_path})")

    # ── save final ────────────────────────────────────────────────────────
    ckpt_final = build_checkpoint(
        embedding_layer, backbone, head, aggrmodel,
        scaler_stats, target_name, cfg.num_epochs,
    )
    torch.save(ckpt_final, final_path)
    print(f"\nFinal model → {final_path}")

    return best_path


# ── CLI ───────────────────────────────────────────────────────────────────────

DATASETS = {
    "eccentric": {
        "csv": "data/sei/偏心因子.csv",
        "output": "trained_models/sei_eccentric_factor",
        "target": "偏心因子",
    },
    "melting": {
        "csv": "data/sei/熔点.csv",
        "output": "trained_models/sei_melting_point",
        "target": "熔点",
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune on SEI property datasets")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "eccentric", "melting"],
                        help="Which dataset to finetune (default: all)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override num_epochs from config")
    parser.add_argument("--bs", type=int, default=None,
                        help="Override batch_size from config")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID (default: 0)")
    args = parser.parse_args()

    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    if device_str == "cpu":
        print("[WARNING] CUDA not available, using CPU (slow).")

    to_run = ["eccentric", "melting"] if args.dataset == "all" else [args.dataset]

    for key in to_run:
        info = DATASETS[key]
        csv_path = PROJECT_ROOT / info["csv"]
        if not csv_path.is_file():
            print(f"[SKIP] File not found: {csv_path}")
            continue

        print(f"\n{'#'*60}")
        print(f"#  Finetuning: {info['target']}")
        print(f"#  CSV:  {csv_path}")
        print(f"#  Out:  {info['output']}")
        print(f"{'#'*60}")

        finetune_dataset(
            csv_path=str(csv_path),
            output_dir=str(PROJECT_ROOT / info["output"]),
            target_name=info["target"],
            num_epochs=args.epochs,
            batch_size=args.bs,
            device=device_str,
        )

    print("\n✓ All done. Models saved to:")
    for key in to_run:
        info = DATASETS[key]
        p = PROJECT_ROOT / info["output"] / "best_model.pth"
        print(f"  {p}  ({'✓' if p.is_file() else '✗ missing'})")
