# finetune_regression.py
# English Version with full_train option (all original outputs preserved)

from atomprop.dataloader.dataloader import SMILESToInputs
from atomprop.models.gnns import Embedder, GNNAggr, MaskedMSELoss
from atomprop.utils.mlp import MLP
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import argparse
from tqdm import tqdm
from torch_geometric.data import Data, Batch, DataLoader
from torch.utils.tensorboard import SummaryWriter
from atomprop.dataloader.splitter import ScaffoldSplitter
from deepchem.data import NumpyDataset
import csv
import os
import json
from datetime import datetime
from atomprop.models.geat import GeATNet
import configs.config_reg as cfg
from atomprop.utils.utils import remove_module_prefix
from atomprop.paths import FIGURES_DIR, ensure_output_dirs

# Use MSE-based loss for regression tasks
criterion = MaskedMSELoss()


def create_dataset_from_smiles_labels(smiles_list, labels_list):
    """
    Create PyTorch Geometric dataset from SMILES strings and continuous regression labels.
    Args:
        smiles_list: List of molecular SMILES strings
        labels_list: List of regression targets
    Returns:
        List of PyG Data objects for model training
    """
    dataset = []
    for smi, label in zip(smiles_list, labels_list):
        atom_info, edge_info, mol = SMILESToInputs.convert(smi, sanitize=False)
        if atom_info is None or edge_info is None:
            continue

        if edge_info.dim() == 2 and edge_info.size(1) == 4:
            edge_index = edge_info[:, :2].t().contiguous()
            edge_attr = edge_info[:, 2:]
        else:
            edge_index = torch.tensor([[], []], dtype=torch.long)
            edge_attr = torch.tensor([], dtype=torch.long).view(0, 2)
        
        data = Data(x=atom_info, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor(label, dtype=torch.float))
        dataset.append(data)

    if len(dataset) == 0:
        raise RuntimeError("No valid molecules created from SMILES list!")
    return dataset


def compute_scaler_stats(labels):
    """
    Compute mean and std per task, ignoring missing values (-1).
    Returns a dict with 'mean' and 'scale' arrays of shape (n_tasks,).
    """
    n_tasks = labels.shape[1]
    means = []
    scales = []
    for col in range(n_tasks):
        col_vals = labels[:, col]
        valid_vals = col_vals[col_vals != -1]
        if len(valid_vals) == 0:
            mean_val, scale_val = 0.0, 1.0
        elif len(valid_vals) == 1:
            mean_val, scale_val = valid_vals[0], 1.0
        else:
            mean_val = np.mean(valid_vals)
            scale_val = np.std(valid_vals)
        means.append(mean_val)
        scales.append(scale_val)
    return {
        'mean': np.array(means, dtype=np.float32),
        'scale': np.array(scales, dtype=np.float32),
        'n_tasks': n_tasks
    }


def transform_with_scaler(labels, scaler_stats):
    """
    Apply standardization: (x - mean) / scale.
    Missing values (-1) are left unchanged.
    """
    labels = labels.copy()
    mean = scaler_stats['mean']
    scale = scaler_stats['scale']
    scale = np.where(scale == 0.0, 1.0, scale)
    valid_mask = (labels != -1)
    labels_scaled = np.where(valid_mask, (labels - mean) / scale, labels)
    return labels_scaled.astype(np.float32)


def inverse_transform_with_scaler(scaled_labels, scaler_stats):
    """
    Reverse standardization: x * scale + mean.
    """
    mean = scaler_stats['mean']
    scale = scaler_stats['scale']
    return scaled_labels * scale + mean


def compute_rmse(predictions, labels):
    """
    Compute RMSE metric, ignoring NaN values.
    Returns RMSE per task and overall RMSE.
    """
    valid_mask = ~np.isnan(labels)
    if not np.any(valid_mask):
        return np.nan, np.full(predictions.shape[1], np.nan)
    
    squared_errors = (predictions - labels) ** 2
    mse_per_task = []
    for col in range(predictions.shape[1]):
        col_valid = valid_mask[:, col]
        if np.sum(col_valid) > 0:
            mse = np.mean(squared_errors[col_valid, col])
            mse_per_task.append(mse)
        else:
            mse_per_task.append(np.nan)
    
    overall_mse = np.nanmean(squared_errors[valid_mask])
    rmse_per_task = np.sqrt(np.array(mse_per_task))
    overall_rmse = np.sqrt(overall_mse)
    
    return overall_rmse, rmse_per_task


def compute_r2(predictions, labels):
    """
    Compute R² score, ignoring NaN values.
    Returns R² per task and overall R².
    """
    valid_mask = ~np.isnan(labels)
    if not np.any(valid_mask):
        return np.nan, np.full(predictions.shape[1], np.nan)
    
    r2_per_task = []
    for col in range(predictions.shape[1]):
        col_valid = valid_mask[:, col]
        if np.sum(col_valid) > 0:
            col_preds = predictions[col_valid, col]
            col_labels = labels[col_valid, col]
            ss_res = np.sum((col_labels - col_preds) ** 2)
            ss_tot = np.sum((col_labels - np.mean(col_labels)) ** 2)
            if ss_tot == 0:
                r2 = 1.0 if ss_res == 0 else 0.0
            else:
                r2 = 1 - (ss_res / ss_tot)
            r2_per_task.append(r2)
        else:
            r2_per_task.append(np.nan)
    
    all_valid_preds = predictions[valid_mask]
    all_valid_labels = labels[valid_mask]
    ss_res_total = np.sum((all_valid_labels - all_valid_preds) ** 2)
    ss_tot_total = np.sum((all_valid_labels - np.mean(all_valid_labels)) ** 2)
    if ss_tot_total == 0:
        overall_r2 = 1.0 if ss_res_total == 0 else 0.0
    else:
        overall_r2 = 1 - (ss_res_total / ss_tot_total)
    
    return overall_r2, np.array(r2_per_task)


def evaluate_model(model_components, dataloader, criterion, y_cols, device, scaler_stats=None, aggr='attention'):
    """
    Evaluate model and return metrics and predictions.
    """
    embedding_layer, backbone, head, aggrmodel = model_components
    embedding_layer.eval()
    backbone.eval()
    head.eval()
    if aggr == 'attention':
        aggrmodel.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            emb = embedding_layer(batch.x.squeeze())
            emb = backbone(Data(x=emb, edge_index=batch.edge_index, edge_attr=batch.edge_attr), batch=batch.batch)
            graph_emb = aggrmodel(emb, batch.batch)
            preds = head(graph_emb)

            loss = criterion(preds.reshape(-1, len(y_cols)), batch.y.reshape(-1, len(y_cols)))
            total_loss += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch.y.reshape(-1, len(y_cols)).cpu().numpy())

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    if len(all_preds) > 0:
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        if scaler_stats is not None:
            pred_inv = inverse_transform_with_scaler(all_preds, scaler_stats)
            label_inv = inverse_transform_with_scaler(all_labels, scaler_stats)
            pred_inv = np.where(all_labels == -1, np.nan, pred_inv)
            label_inv = np.where(all_labels == -1, np.nan, label_inv)
        else:
            pred_inv = all_preds
            label_inv = all_labels

        overall_rmse, rmse_per_task = compute_rmse(pred_inv, label_inv)
        overall_r2, r2_per_task = compute_r2(pred_inv, label_inv)
    else:
        pred_inv = np.array([])
        label_inv = np.array([])
        overall_rmse = np.nan
        rmse_per_task = np.full(len(y_cols), np.nan)
        overall_r2 = np.nan
        r2_per_task = np.full(len(y_cols), np.nan)

    return avg_loss, overall_rmse, rmse_per_task, overall_r2, r2_per_task, pred_inv, label_inv


def train(train_dataloader, val_dataloader, test_dataloader, model_components, optimizers, schedulers,
          device, num_epochs, y_cols, logdir, no_pretrain, scaler_stats, run_num, aggr='attention', full_train=False):
    """
    Train model with full_train option: no val/test, train on all data.
    """
    embedding_layer, backbone, head, aggrmodel = model_components
    writer = SummaryWriter(log_dir=f'runs/finetune_{logdir}/run{run_num}')

    best_val_rmse = float('inf')
    best_epoch = -1
    global_step = 0
    tolerating = 0

    for epoch in range(num_epochs):
        embedding_layer.train()
        backbone.train()
        head.train()
        if aggr == 'attention':
            aggrmodel.train()

        epoch_loss = 0.0
        num_batches = 0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training"):
            batch = batch.to(device)
            for opt in optimizers:
                opt.zero_grad()

            emb = embedding_layer(batch.x.squeeze())
            emb = backbone(Data(x=emb, edge_index=batch.edge_index, edge_attr=batch.edge_attr), batch=batch.batch)
            graph_emb = aggrmodel(emb, batch.batch)
            preds = head(graph_emb)

            loss = criterion(preds.reshape(-1, len(y_cols)), batch.y.reshape(-1, len(y_cols)))
            loss.backward()

            for opt in optimizers:
                opt.step()

            epoch_loss += loss.item()
            num_batches += 1
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            global_step += 1

        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.6f}")

        for scheduler in schedulers:
            scheduler.step()

        # Skip validation in full_train mode
        if full_train:
            if epoch == num_epochs - 1:
                model_suffix = "nopretrain" if no_pretrain else "pretrained"
                save_path = f"trained_models/{logdir}/final_model_{model_suffix}.pth"
                _ckpt_final = {
                    'epoch': epoch + 1,
                    'embedding_layer_state_dict': embedding_layer.state_dict(),
                    'head_state_dict': head.state_dict(),
                    'aggr_state_dict': aggrmodel.state_dict() if aggr == 'attention' else None,
                    'scaler_stats': scaler_stats,
                }
                if cfg.use_lora:
                    _ckpt_final['lora_state_dict'] = backbone.get_lora_state_dict()
                    _ckpt_final['lora_config'] = {
                        'rank': cfg.lora_rank, 'alpha': cfg.lora_alpha,
                        'include_ffn': cfg.lora_include_ffn,
                        'include_global_attn': cfg.lora_include_global_attn,
                    }
                else:
                    _ckpt_final['backbone_state_dict'] = backbone.state_dict()
                torch.save(_ckpt_final, save_path)
                print(f"[Full Train] Final model saved: {save_path}")
            continue

        # Original validation
        val_loss, val_rmse, val_rmse_per_task, val_r2, val_r2_per_task, _, _ = evaluate_model(
            model_components, val_dataloader, criterion, y_cols, device, scaler_stats=scaler_stats, aggr=aggr
        )
        print(f"Epoch {epoch+1} Val MSE: {val_loss:.6f} | Val RMSE: {val_rmse:.6f} | Val R²: {val_r2:.6f}")
        writer.add_scalar('Val/MSE', val_loss, epoch)
        writer.add_scalar('Val/RMSE', val_rmse, epoch)
        writer.add_scalar('Val/R2', val_r2, epoch)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch + 1
            model_suffix = "nopretrain" if no_pretrain else "pretrained"
            save_path = f"trained_models/{logdir}/best_model_{model_suffix}.pth"
            _ckpt_best = {
                'epoch': epoch + 1,
                'embedding_layer_state_dict': embedding_layer.state_dict(),
                'head_state_dict': head.state_dict(),
                'aggr_state_dict': aggrmodel.state_dict() if aggr == 'attention' else None,
                'val_rmse': val_rmse,
                'val_r2': val_r2,
                'val_mse': val_loss,
                'scaler_stats': scaler_stats,
            }
            if cfg.use_lora:
                _ckpt_best['lora_state_dict'] = backbone.get_lora_state_dict()
                _ckpt_best['lora_config'] = {
                    'rank': cfg.lora_rank, 'alpha': cfg.lora_alpha,
                    'include_ffn': cfg.lora_include_ffn,
                    'include_global_attn': cfg.lora_include_global_attn,
                }
            else:
                _ckpt_best['backbone_state_dict'] = backbone.state_dict()
            torch.save(_ckpt_best, save_path)
            print(f"Best model saved at epoch {best_epoch}")
            tolerating = 0
        else:
            tolerating += 1
            if tolerating >= cfg.tolerance:
                print("Early stopping triggered.")
                break

    writer.close()

    # Skip test in full_train mode
    if full_train:
        print("\n[Full Train] Training completed. No test.")
        return {
            'best_val_rmse': np.nan,
            'best_val_r2': np.nan,
            'best_val_mse': np.nan,
            'best_epoch': best_epoch,
            'test_rmse': np.nan,
            'test_r2': np.nan,
            'test_mse': np.nan,
            'test_rmse_per_task': [np.nan]*len(y_cols),
            'test_r2_per_task': [np.nan]*len(y_cols),
            'test_predictions_path': None
        }

    # Original test logic (FULLY PRESERVED)
    print("\n--- Testing ---")
    model_suffix = "nopretrain" if no_pretrain else "pretrained"
    load_path = f"trained_models/{logdir}/best_model_{model_suffix}.pth"
    checkpoint = torch.load(load_path, weights_only=False, map_location=device)

    embedding_layer.load_state_dict(checkpoint['embedding_layer_state_dict'])
    if 'lora_state_dict' in checkpoint:
        # LoRA checkpoint: backbone already has LoRA applied, just load adapter weights
        backbone.load_lora_state_dict(checkpoint['lora_state_dict'])
    else:
        backbone.load_state_dict(checkpoint['backbone_state_dict'])
    head.load_state_dict(checkpoint['head_state_dict'])
    if aggr == 'attention' and checkpoint['aggr_state_dict']:
        aggrmodel.load_state_dict(checkpoint['aggr_state_dict'])

    test_scaler_stats = checkpoint.get('scaler_stats', None)

    test_loss, test_rmse, test_rmse_per_task, test_r2, test_r2_per_task, all_test_preds, all_test_labels = evaluate_model(
        model_components, test_dataloader, criterion, y_cols, device, scaler_stats=test_scaler_stats, aggr=aggr
    )

    output_csv_path = f"trained_models/{logdir}/test_predictions_{model_suffix}.csv"
    if len(all_test_preds) > 0:
        with open(output_csv_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            header = []
            for col in y_cols:
                header.extend([f"{col}_pred", f"{col}_label"])
            csv_writer.writerow(header)
            for i in range(all_test_preds.shape[0]):
                row = []
                for j in range(len(y_cols)):
                    pred_val = all_test_preds[i, j]
                    label_val = all_test_labels[i, j]
                    pred_str = "" if np.isnan(pred_val) else f"{pred_val:.6f}"
                    label_str = "" if np.isnan(label_val) else f"{label_val:.6f}"
                    row.extend([pred_str, label_str])
                csv_writer.writerow(row)

        if cfg.draw_plot:
            for j, col in enumerate(y_cols):
                plt.figure(figsize=(8, 8))
                valid_mask = ~np.isnan(all_test_labels[:, j])
                if np.sum(valid_mask) > 0:
                    plt.scatter(all_test_labels[valid_mask, j], all_test_preds[valid_mask, j], alpha=0.6)
                    min_val = min(np.min(all_test_labels[valid_mask, j]), np.min(all_test_preds[valid_mask, j]))
                    max_val = max(np.max(all_test_labels[valid_mask, j]), np.max(all_test_preds[valid_mask, j]))
                    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
                    plt.xlabel(f'Actual {col}')
                    plt.ylabel(f'Predicted {col}')
                    plt.title(f'{col}: Actual vs Predicted (RMSE: {test_rmse_per_task[j]:.4f}, R²: {test_r2_per_task[j]:.4f})')
                    plt.tight_layout()
                    ensure_output_dirs()
                    scatter_dir = FIGURES_DIR / logdir
                    scatter_dir.mkdir(parents=True, exist_ok=True)
                    plt.savefig(scatter_dir / f"scatter_{col}_run{run_num}.png", dpi=150)
                    plt.close()

    print(f"Test MSE: {test_loss:.6f} | Test RMSE: {test_rmse:.6f} | Test R²: {test_r2:.6f}")
    for i, col in enumerate(y_cols):
        if not np.isnan(test_rmse_per_task[i]):
            print(f"  {col}: RMSE = {test_rmse_per_task[i]:.6f}, R² = {test_r2_per_task[i]:.6f}")

    return {
        'best_val_rmse': best_val_rmse,
        'best_val_r2': checkpoint.get('val_r2', None),
        'best_val_mse': checkpoint.get('val_mse', None),
        'best_epoch': best_epoch,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'test_mse': test_loss,
        'test_rmse_per_task': test_rmse_per_task.tolist(),
        'test_r2_per_task': test_r2_per_task.tolist(),
        'test_predictions_path': output_csv_path
    }


def main(ft_dataset=None):
    parser = argparse.ArgumentParser(description='GNN Regression Finetuning')
    parser.add_argument('--dataset', type=str, default='', help='Dataset name')
    parser.add_argument('--bs', type=int, default=128, help='Batch size')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--full_train', action='store_true', help='Train on full dataset without val/test')

    args = parser.parse_args()

    if ft_dataset is None:
        ft_dataset = args.dataset
        cfg.batch_size = args.bs
        cfg.device = f"cuda:{args.gpu}"

    full_train_mode = args.full_train

    if len(ft_dataset) > 0:
        cfg.set_data_path(ft_dataset)
        print(f"Finetuning on {ft_dataset}...")

    os.makedirs(f"trained_models/{cfg.logdir}", exist_ok=True)
    device = torch.device(cfg.device)

    df = pd.read_csv(cfg.data_path)
    headers = df.columns.tolist()
    y_cols = [col for col in headers if col not in cfg.exclude_list + [cfg.x_col]]
    smiles_list = df[cfg.x_col].tolist()
    df[y_cols] = df[y_cols].fillna(-1)
    labels = df[y_cols].values.astype(np.float32)

    cfg.print_all_params()
    dc_dataset = NumpyDataset(X=labels, ids=smiles_list)
    splitter = ScaffoldSplitter()
    all_results = []

    for run_num in range(cfg.num_runs):
        print(f"\nStarting run {run_num}...")

        embedding_layer = Embedder(num_atom_types=120, embed_dim=cfg.embed_dim)
        backbone = GeATNet(
            embed_dim=cfg.embed_dim, num_heads=cfg.num_heads, global_num_heads=cfg.global_num_heads,
            output_negative_slope=cfg.output_negative_slope, dropout=cfg.geat_dropout,
            geat_num_layers=cfg.geat_num_layers, aggr_num_layers=cfg.aggr_num_layers,
            FFN_type=cfg.FFN_type, FFN_hidden_dim=cfg.FFN_hidden_dim,
            FFN_num_experts=cfg.FFN_num_experts, FFN_num_layers=cfg.FFN_num_layers, FFN_top_k=cfg.FFN_top_k,
            use_edge_embedding=cfg.use_edge_embedding,
            per_layer_FFN_type=cfg.per_layer_FFN_type,
            per_layer_FFN_hidden_dim=cfg.per_layer_FFN_hidden_dim,
            per_layer_FFN_num_layers=cfg.per_layer_FFN_num_layers,
            per_layer_FFN_dropout=cfg.per_layer_FFN_dropout,
            per_layer_FFN_num_experts=cfg.per_layer_FFN_num_experts,
            per_layer_FFN_top_k=cfg.per_layer_FFN_top_k,
            attention_rank=cfg.attention_rank
        )
        aggrmodel = GNNAggr(embed_dim=cfg.embed_dim, aggr=cfg.aggr, layers=1)
        head = MLP(input_dim=cfg.embed_dim, hidden_dim=cfg.head_hidden_dim, output_dim=len(y_cols),
                   num_layers=cfg.head_layers, dropout=cfg.head_dropout, batch_norm=True, output_activation=None)

        if not cfg.no_pretrain:
            ckpt = torch.load(cfg.pretrained_path, weights_only=False, map_location=device)
            embedding_layer.load_state_dict(remove_module_prefix(ckpt['embedding_layer_state_dict']))
            backbone_state = remove_module_prefix(ckpt['backbone_state_dict'])
            try:
                backbone.load_state_dict(backbone_state, strict=True)
            except RuntimeError as e:
                print(f"[WARNING] Strict load failed (expected for architecture change): {e}")
                print("Attempting partial load with strict=False...")
                missing, unexpected = backbone.load_state_dict(backbone_state, strict=False)
                print(f"Loaded backbone with {len(missing)} missing keys and {len(unexpected)} unexpected keys")
                if missing:
                    print(f"  First 5 missing: {missing[:5]}")

        # ---- Apply LoRA if configured ----
        if cfg.use_lora:
            backbone.apply_lora(
                rank=cfg.lora_rank,
                alpha=cfg.lora_alpha,
                dropout=cfg.lora_dropout,
                include_ffn=cfg.lora_include_ffn,
                include_global_attn=cfg.lora_include_global_attn,
            )
            backbone = backbone.to(device)  # move new LoRA params to device
            print(f"[LoRA] Applied LoRA (rank={cfg.lora_rank}, alpha={cfg.lora_alpha})")
            lora_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
            print(f"[LoRA] Trainable backbone params: {lora_params:,}")

        print(f"Embedding Params: {sum(p.numel() for p in embedding_layer.parameters() if p.requires_grad)}")
        print(f"Backbone Params: {sum(p.numel() for p in backbone.parameters() if p.requires_grad)}")
        print(f"Head Params: {sum(p.numel() for p in head.parameters() if p.requires_grad)}")

        # Full train: use all data, no split
        if full_train_mode:
            train_smiles = smiles_list
            train_labels = labels
            val_smiles = []
            val_labels = np.empty((0, labels.shape[1]))
            test_smiles = []
            test_labels = np.empty((0, labels.shape[1]))
            print("=> FULL TRAIN MODE: All data used for training")
        else:
            train_inds, val_inds, test_inds = splitter.split(dataset=dc_dataset, seed=cfg.random_state + run_num)
            train_smiles = [dc_dataset.ids[i] for i in train_inds]
            train_labels = dc_dataset.X[train_inds]
            val_smiles = [dc_dataset.ids[i] for i in val_inds]
            val_labels = dc_dataset.X[val_inds]
            test_smiles = [dc_dataset.ids[i] for i in test_inds]
            test_labels = dc_dataset.X[test_inds]

        scaler_stats = compute_scaler_stats(train_labels)
        train_labels_scaled = transform_with_scaler(train_labels, scaler_stats)
        val_labels_scaled = transform_with_scaler(val_labels, scaler_stats) if not full_train_mode else np.empty((0, labels.shape[1]))
        test_labels_scaled = transform_with_scaler(test_labels, scaler_stats) if not full_train_mode else np.empty((0, labels.shape[1]))

        print(f"Run {run_num}: Train={len(train_smiles)}, Val={len(val_smiles)}, Test={len(test_smiles)}")

        train_dataset = create_dataset_from_smiles_labels(train_smiles, train_labels_scaled)
        val_dataset = create_dataset_from_smiles_labels(val_smiles, val_labels_scaled) if not full_train_mode else []
        test_dataset = create_dataset_from_smiles_labels(test_smiles, test_labels_scaled) if not full_train_mode else []

        train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=Batch.from_data_list, num_workers=0)
        val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=Batch.from_data_list, num_workers=0) if not full_train_mode else None
        test_dataloader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=False, collate_fn=Batch.from_data_list, num_workers=0) if not full_train_mode else None

        embedding_layer = embedding_layer.to(device)
        backbone = backbone.to(device)
        head = head.to(device)
        aggrmodel = aggrmodel.to(device) if cfg.aggr == 'attention' else aggrmodel

        # Collect only trainable params (LoRA adapters if use_lora, else all backbone params)
        backbone_trainable = [p for p in backbone.parameters() if p.requires_grad]
        opt_emb_backbone = torch.optim.Adam([
            {'params': embedding_layer.parameters(), 'lr': cfg.lr_embedding_layer_backbone},
            {'params': backbone_trainable, 'lr': cfg.lr_embedding_layer_backbone}
        ])
        opt_head = torch.optim.Adam(head.parameters(), lr=cfg.lr_head)
        optimizers = [opt_emb_backbone, opt_head]

        sched_emb_backbone = torch.optim.lr_scheduler.CosineAnnealingLR(opt_emb_backbone, T_max=cfg.T_max, eta_min=cfg.eta_min_embedding_layer_backbone)
        sched_head = torch.optim.lr_scheduler.CosineAnnealingLR(opt_head, T_max=cfg.T_max, eta_min=cfg.eta_min_head)
        schedulers = [sched_emb_backbone, sched_head]

        model_components = (embedding_layer, backbone, head, aggrmodel)

        try:
            result = train(
                train_dataloader, val_dataloader, test_dataloader,
                model_components, optimizers, schedulers,
                device, cfg.num_epochs, y_cols, cfg.logdir, cfg.no_pretrain, scaler_stats, run_num, cfg.aggr, full_train=full_train_mode
            )
            all_results.append(result)
        except Exception as e:
            import traceback
            print(f"Run {run_num} error: {e}")
            traceback.print_exc()

    print(f"\n{'='*60}\nREGRESSION SUMMARY\n{'='*60}")
    if all_results:
        val_rmses = [r['best_val_rmse'] for r in all_results]
        val_r2s = [r['best_val_r2'] for r in all_results]
        test_rmses = [r['test_rmse'] for r in all_results]
        test_r2s = [r['test_r2'] for r in all_results]
        for i, r in enumerate(all_results):
            print(f"Run {i}: Val RMSE={r['best_val_rmse']:.6f}, Val R²={r['best_val_r2']:.6f}, Test RMSE={r['test_rmse']:.6f}, Test R²={r['test_r2']:.6f}")
        print(f"\nMean Test RMSE: {np.mean(test_rmses):.6f} ± {np.std(test_rmses):.6f}")
        print(f"Mean Test R²: {np.mean(test_r2s):.6f} ± {np.std(test_r2s):.6f}")
        
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_to_serializable(item) for item in obj)
            else:
                return obj
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'dataset': cfg.data_path,
            'results': convert_to_serializable(all_results),
            'summary_stats': {
                'mean_test_rmse': float(np.mean(test_rmses)),
                'std_test_rmse': float(np.std(test_rmses)),
                'mean_test_r2': float(np.mean(test_r2s)),
                'std_test_r2': float(np.std(test_r2s))
            }
        }
        with open(f"trained_models/{cfg.logdir}/regression_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
    else:
        print("NO VALID RUNS COMPLETED!")


if __name__ == "__main__":
    main()
