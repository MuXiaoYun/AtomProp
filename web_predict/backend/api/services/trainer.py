"""Web training service (regression & classification finetune)."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

ATOMPROP_ROOT = Path(__file__).resolve().parents[4]
if str(ATOMPROP_ROOT) not in sys.path:
    sys.path.insert(0, str(ATOMPROP_ROOT))

from atomprop.dataloader.dataloader import SMILESToInputs  # noqa: E402
from atomprop.dataloader.splitter import ScaffoldSplitter  # noqa: E402
from atomprop.models.gnns import (  # noqa: E402
    Embedder,
    GNNAggr,
    MaskedBCELossWithLogits,
    MaskedFocalLoss,
    MaskedMSELoss,
)
from atomprop.models.geat import GeATNet  # noqa: E402
from atomprop.utils.mlp import MLP  # noqa: E402
from atomprop.utils.utils import remove_module_prefix  # noqa: E402

import configs.config_reg as cfg_reg  # noqa: E402
import configs.config_finetune as cfg_cls  # noqa: E402


ProgressCallback = Callable[[int, int, float, float], None]


@dataclass
class TrainParams:
    csv_path: str
    smiles_column: str
    target_column: str
    task_type: str  # regression | classification
    init_mode: str  # scratch | checkpoint | pretrain
    checkpoint_path: str | None
    num_epochs: int
    learning_rate: float
    output_dir: str
    job_id: str
    # LoRA options
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 8.0
    lora_dropout: float = 0.0


def _get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _build_geat(embed_dim: int, cfg_module):
    return GeATNet(
        embed_dim=embed_dim,
        num_heads=cfg_module.num_heads,
        global_num_heads=cfg_module.global_num_heads,
        output_negative_slope=cfg_module.output_negative_slope,
        dropout=cfg_module.geat_dropout,
        geat_num_layers=cfg_module.geat_num_layers,
        aggr_num_layers=cfg_module.aggr_num_layers,
        FFN_type=cfg_module.FFN_type,
        FFN_hidden_dim=cfg_module.FFN_hidden_dim,
        FFN_num_experts=cfg_module.FFN_num_experts,
        FFN_num_layers=cfg_module.FFN_num_layers,
        FFN_top_k=cfg_module.FFN_top_k,
        use_edge_embedding=cfg_module.use_edge_embedding,
    )


def create_dataset_from_smiles_labels(smiles_list, labels_list, regression: bool):
    dataset = []
    for smi, label in zip(smiles_list, labels_list):
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
        data = Data(x=atom_info, edge_index=edge_index, edge_attr=edge_attr, y=y)
        dataset.append(data)
    if not dataset:
        raise RuntimeError("无法从 SMILES 构建有效图数据，请检查输入。")
    return dataset


def compute_scaler_stats(labels: np.ndarray) -> dict:
    n_tasks = labels.shape[1]
    means, scales = [], []
    for col in range(n_tasks):
        col_vals = labels[:, col]
        valid_vals = col_vals[col_vals != -1]
        if len(valid_vals) == 0:
            mean_val, scale_val = 0.0, 1.0
        elif len(valid_vals) == 1:
            mean_val, scale_val = float(valid_vals[0]), 1.0
        else:
            mean_val = float(np.mean(valid_vals))
            scale_val = float(np.std(valid_vals)) or 1.0
        means.append(mean_val)
        scales.append(scale_val)
    return {
        "mean": np.array(means, dtype=np.float32),
        "scale": np.array(scales, dtype=np.float32),
        "n_tasks": n_tasks,
    }


def transform_with_scaler(labels: np.ndarray, scaler_stats: dict) -> np.ndarray:
    labels = labels.copy()
    mean = scaler_stats["mean"]
    scale = scaler_stats["scale"]
    scale = np.where(scale == 0.0, 1.0, scale)
    valid_mask = labels != -1
    return np.where(valid_mask, (labels - mean) / scale, labels).astype(np.float32)


def _build_checkpoint(embedding_layer, backbone, head, aggrmodel, scaler_stats,
                      use_lora, task_type, target_column, **extra):
    """Build a checkpoint dict. Uses LoRA state dict if use_lora is True."""
    ckpt = {
        "embedding_layer_state_dict": embedding_layer.state_dict(),
        "head_state_dict": head.state_dict(),
        "aggr_state_dict": aggrmodel.state_dict() if aggrmodel is not None else None,
        "scaler_stats": scaler_stats,
        "task_type": task_type,
        "target_column": target_column,
        **extra,
    }
    if use_lora:
        ckpt["lora_state_dict"] = backbone.get_lora_state_dict()
        ckpt["lora_config"] = {"rank": 8, "alpha": 8.0, "include_ffn": False, "include_global_attn": False}
    else:
        ckpt["backbone_state_dict"] = backbone.state_dict()
    return ckpt


def _load_pretrained_weights(embedding_layer, backbone, path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    embedding_layer.load_state_dict(
        remove_module_prefix(ckpt["embedding_layer_state_dict"])
    )
    backbone.load_state_dict(remove_module_prefix(ckpt["backbone_state_dict"]))


def _load_checkpoint(
    embedding_layer,
    backbone,
    head,
    aggrmodel,
    path: str,
    device: torch.device,
    aggr: str,
):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    embedding_layer.load_state_dict(ckpt["embedding_layer_state_dict"])
    backbone.load_state_dict(ckpt["backbone_state_dict"])
    if "head_state_dict" in ckpt:
        try:
            head.load_state_dict(ckpt["head_state_dict"], strict=False)
        except Exception:
            pass
    if aggr == "attention" and ckpt.get("aggr_state_dict"):
        try:
            aggrmodel.load_state_dict(ckpt["aggr_state_dict"], strict=False)
        except Exception:
            pass


def _run_regression(
    params: TrainParams,
    on_epoch: ProgressCallback,
) -> str:
    cfg = cfg_reg
    device = _get_device()
    y_cols = [params.target_column]

    df = pd.read_csv(params.csv_path)
    smiles_list = df[params.smiles_column].astype(str).tolist()
    df[y_cols] = df[y_cols].fillna(-1)
    labels = df[y_cols].values.astype(np.float32)

    # Lazy import to avoid triggering DGL at module load (DGL 2.2.1 requires PyTorch ≤ 2.3)
    from deepchem.data import NumpyDataset

    dc_dataset = NumpyDataset(X=labels, ids=smiles_list)
    splitter = ScaffoldSplitter()
    train_inds, val_inds, test_inds = splitter.split(
        dataset=dc_dataset, seed=cfg.random_state
    )

    train_smiles = [dc_dataset.ids[i] for i in train_inds]
    train_labels = dc_dataset.X[train_inds]
    val_smiles = [dc_dataset.ids[i] for i in val_inds]
    val_labels = dc_dataset.X[val_inds]

    scaler_stats = compute_scaler_stats(train_labels)
    train_labels_scaled = transform_with_scaler(train_labels, scaler_stats)
    val_labels_scaled = transform_with_scaler(val_labels, scaler_stats)

    train_dataset = create_dataset_from_smiles_labels(
        train_smiles, train_labels_scaled, regression=True
    )
    val_dataset = create_dataset_from_smiles_labels(
        val_smiles, val_labels_scaled, regression=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=Batch.from_data_list,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=Batch.from_data_list,
        num_workers=0,
    )

    embedding_layer = Embedder(num_atom_types=120, embed_dim=cfg.embed_dim).to(device)
    backbone = _build_geat(cfg.embed_dim, cfg).to(device)
    aggrmodel = GNNAggr(embed_dim=cfg.embed_dim, aggr=cfg.aggr, layers=1).to(device)
    head = MLP(
        input_dim=cfg.embed_dim,
        hidden_dim=cfg.head_hidden_dim,
        output_dim=len(y_cols),
        num_layers=cfg.head_layers,
        dropout=cfg.head_dropout,
        batch_norm=True,
        output_activation=None,
    ).to(device)

    # Apply LoRA after model creation (before optimizer so only trainable params are included)
    if params.use_lora:
        backbone.apply_lora(
            rank=params.lora_rank,
            alpha=params.lora_alpha,
            dropout=params.lora_dropout,
        )
        backbone.to(device)  # move newly-created LoRA params to device

    if params.init_mode == "pretrain" and cfg.pretrained_path:
        p = str(ATOMPROP_ROOT / cfg.pretrained_path)
        if os.path.isfile(p):
            _load_pretrained_weights(embedding_layer, backbone, p, device)
    elif params.init_mode == "checkpoint" and params.checkpoint_path:
        _load_checkpoint(
            embedding_layer,
            backbone,
            head,
            aggrmodel,
            params.checkpoint_path,
            device,
            cfg.aggr,
        )

    lr_ratio = cfg.lr_embedding_layer_backbone / cfg.lr_head
    lr_backbone = params.learning_rate * lr_ratio
    backbone_trainable = [p for p in backbone.parameters() if p.requires_grad]
    opt_emb = torch.optim.Adam(
        [
            {"params": embedding_layer.parameters(), "lr": lr_backbone},
            {"params": backbone_trainable, "lr": lr_backbone},
        ],
        weight_decay=cfg.wd_emb_backbone,
    )
    opt_head = torch.optim.Adam(
        head.parameters(), lr=params.learning_rate, weight_decay=cfg.wd_head
    )
    sched_emb = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_emb, T_max=params.num_epochs, eta_min=cfg.eta_min_embedding_layer_backbone
    )
    sched_head = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_head, T_max=params.num_epochs, eta_min=cfg.eta_min_head
    )

    criterion = MaskedMSELoss()
    best_val = float("inf")
    best_path = os.path.join(params.output_dir, "best_model.pth")
    tolerating = 0

    for epoch in range(params.num_epochs):
        embedding_layer.train()
        backbone.train()
        head.train()
        if cfg.aggr == "attention":
            aggrmodel.train()

        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            opt_emb.zero_grad()
            opt_head.zero_grad()
            emb = embedding_layer(batch.x.squeeze())
            emb = backbone(
                Data(x=emb, edge_index=batch.edge_index, edge_attr=batch.edge_attr),
                batch=batch.batch,
            )
            g_emb = aggrmodel(emb, batch.batch)
            preds = head(g_emb)
            loss = criterion(
                preds.reshape(-1, len(y_cols)), batch.y.reshape(-1, len(y_cols))
            )
            loss.backward()
            opt_emb.step()
            opt_head.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)
        sched_emb.step()
        sched_head.step()

        embedding_layer.eval()
        backbone.eval()
        head.eval()
        if cfg.aggr == "attention":
            aggrmodel.eval()

        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                emb = embedding_layer(batch.x.squeeze())
                emb = backbone(
                    Data(x=emb, edge_index=batch.edge_index, edge_attr=batch.edge_attr),
                    batch=batch.batch,
                )
                g_emb = aggrmodel(emb, batch.batch)
                preds = head(g_emb)
                loss = criterion(
                    preds.reshape(-1, len(y_cols)), batch.y.reshape(-1, len(y_cols))
                )
                val_loss_sum += loss.item() * batch.num_graphs
                val_n += batch.num_graphs
        avg_val = val_loss_sum / max(val_n, 1)

        on_epoch(epoch + 1, params.num_epochs, avg_train, avg_val)

        if avg_val < best_val:
            best_val = avg_val
            tolerating = 0
            ckpt = _build_checkpoint(
                embedding_layer, backbone, head, aggrmodel, scaler_stats,
                params.use_lora, "regression", params.target_column,
                epoch=epoch + 1, val_loss=avg_val,
            )
            torch.save(ckpt, best_path)
        else:
            tolerating += 1
            if tolerating >= cfg.tolerance:
                break

    if not os.path.isfile(best_path):
        ckpt = _build_checkpoint(
            embedding_layer, backbone, head, aggrmodel, scaler_stats,
            params.use_lora, "regression", params.target_column,
        )
        torch.save(ckpt, best_path)
    return best_path


def _run_classification(
    params: TrainParams,
    on_epoch: ProgressCallback,
) -> str:
    cfg = cfg_cls
    device = _get_device()
    y_cols = [params.target_column]

    df = pd.read_csv(params.csv_path)
    smiles_list = df[params.smiles_column].astype(str).tolist()
    df[y_cols] = df[y_cols].fillna(-1).astype(float)
    labels = df[y_cols].values

    # Lazy import to avoid triggering DGL at module load (DGL 2.2.1 requires PyTorch ≤ 2.3)
    from deepchem.data import NumpyDataset

    dc_dataset = NumpyDataset(X=labels, ids=smiles_list)
    splitter = ScaffoldSplitter()
    train_inds, val_inds, _ = splitter.split(dataset=dc_dataset, seed=cfg.random_state)

    train_smiles = [dc_dataset.ids[i] for i in train_inds]
    train_labels = dc_dataset.X[train_inds]
    val_smiles = [dc_dataset.ids[i] for i in val_inds]
    val_labels = dc_dataset.X[val_inds]

    train_dataset = create_dataset_from_smiles_labels(
        train_smiles, train_labels, regression=False
    )
    val_dataset = create_dataset_from_smiles_labels(
        val_smiles, val_labels, regression=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=Batch.from_data_list,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=Batch.from_data_list,
        num_workers=0,
    )

    embedding_layer = Embedder(num_atom_types=120, embed_dim=cfg.embed_dim).to(device)
    backbone = _build_geat(cfg.embed_dim, cfg).to(device)
    aggrmodel = GNNAggr(embed_dim=cfg.embed_dim, aggr=cfg.aggr, layers=1).to(device)
    head = MLP(
        input_dim=cfg.embed_dim,
        hidden_dim=cfg.head_hidden_dim,
        output_dim=len(y_cols),
        num_layers=cfg.head_layers,
        dropout=cfg.head_dropout,
        batch_norm=True,
        output_activation=None,
    ).to(device)

    # Apply LoRA after model creation (before optimizer so only trainable params are included)
    if params.use_lora:
        backbone.apply_lora(
            rank=params.lora_rank,
            alpha=params.lora_alpha,
            dropout=params.lora_dropout,
        )
        backbone.to(device)  # move newly-created LoRA params to device

    if params.init_mode == "pretrain" and cfg.pretrained_path:
        p = str(ATOMPROP_ROOT / cfg.pretrained_path)
        if os.path.isfile(p):
            _load_pretrained_weights(embedding_layer, backbone, p, device)
    elif params.init_mode == "checkpoint" and params.checkpoint_path:
        _load_checkpoint(
            embedding_layer,
            backbone,
            head,
            aggrmodel,
            params.checkpoint_path,
            device,
            cfg.aggr,
        )

    lr_ratio = cfg.lr_embedding_layer_backbone / cfg.lr_head
    lr_backbone = params.learning_rate * lr_ratio
    backbone_trainable = [p for p in backbone.parameters() if p.requires_grad]
    opt_emb = torch.optim.Adam(
        [
            {"params": embedding_layer.parameters(), "lr": lr_backbone},
            {"params": backbone_trainable, "lr": lr_backbone},
        ],
        weight_decay=cfg.wd_emb_backbone,
    )
    opt_head = torch.optim.Adam(
        head.parameters(), lr=params.learning_rate, weight_decay=cfg.wd_head
    )
    sched_emb = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_emb, T_max=params.num_epochs, eta_min=cfg.eta_min_embedding_layer_backbone
    )
    sched_head = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_head, T_max=params.num_epochs, eta_min=cfg.eta_min_head
    )

    eval_criterion = MaskedBCELossWithLogits()
    pos_count = torch.zeros(len(y_cols))
    neg_count = torch.zeros(len(y_cols))
    total_valid = torch.zeros(len(y_cols))
    for batch in train_loader:
        y = batch.y.reshape(-1, len(y_cols))
        valid_mask = y != -1
        pos_count += (y == 1).sum(dim=0).cpu()
        neg_count += (y == 0).sum(dim=0).cpu()
        total_valid += valid_mask.sum(dim=0).cpu()
    neg_ratio = neg_count / torch.clamp(total_valid, min=1)
    train_criterion = MaskedFocalLoss(alpha=neg_ratio, gamma=cfg.gamma, reduction="mean")

    best_val = float("inf")
    best_path = os.path.join(params.output_dir, "best_model.pth")
    tolerating = 0

    for epoch in range(params.num_epochs):
        embedding_layer.train()
        backbone.train()
        head.train()
        if cfg.aggr == "attention":
            aggrmodel.train()

        epoch_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            opt_emb.zero_grad()
            opt_head.zero_grad()
            if epoch < cfg.freeze:
                with torch.no_grad():
                    emb = embedding_layer(batch.x.squeeze())
                    emb = backbone(
                        Data(
                            x=emb,
                            edge_index=batch.edge_index,
                            edge_attr=batch.edge_attr,
                        ),
                        batch=batch.batch,
                    )
            else:
                emb = embedding_layer(batch.x.squeeze())
                emb = backbone(
                    Data(
                        x=emb,
                        edge_index=batch.edge_index,
                        edge_attr=batch.edge_attr,
                    ),
                    batch=batch.batch,
                )
            preds = aggrmodel(head(emb), batch.batch)
            loss = train_criterion(
                preds.reshape(-1, len(y_cols)), batch.y.reshape(-1, len(y_cols))
            )
            loss.backward()
            opt_emb.step()
            opt_head.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)
        sched_emb.step()
        sched_head.step()

        embedding_layer.eval()
        backbone.eval()
        head.eval()
        if cfg.aggr == "attention":
            aggrmodel.eval()

        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                emb = embedding_layer(batch.x.squeeze())
                emb = backbone(
                    Data(x=emb, edge_index=batch.edge_index, edge_attr=batch.edge_attr),
                    batch=batch.batch,
                )
                preds = aggrmodel(head(emb), batch.batch)
                loss = eval_criterion(
                    preds.reshape(-1, len(y_cols)), batch.y.reshape(-1, len(y_cols))
                )
                val_loss_sum += loss.item()
                val_n += 1
        avg_val = val_loss_sum / max(val_n, 1)

        on_epoch(epoch + 1, params.num_epochs, avg_train, avg_val)

        if avg_val < best_val:
            best_val = avg_val
            tolerating = 0
            ckpt = _build_checkpoint(
                embedding_layer, backbone, head, aggrmodel, None,
                params.use_lora, "classification", params.target_column,
                epoch=epoch + 1, val_loss=avg_val,
            )
            torch.save(ckpt, best_path)
        else:
            tolerating += 1
            if tolerating >= cfg.tolerance:
                break

    if not os.path.isfile(best_path):
        ckpt = _build_checkpoint(
            embedding_layer, backbone, head, aggrmodel, None,
            params.use_lora, "classification", params.target_column,
        )
        torch.save(ckpt, best_path)
    return best_path


def run_training(params: TrainParams, on_epoch: ProgressCallback) -> str:
    os.makedirs(params.output_dir, exist_ok=True)
    if params.task_type == "classification":
        return _run_classification(params, on_epoch)
    return _run_regression(params, on_epoch)
