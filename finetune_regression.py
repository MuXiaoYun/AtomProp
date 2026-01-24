# finetune_regression.py

from atomprop.dataloader.dataloader import SMILESToInputs
from atomprop.models.GNNs import Embedder, GNN, GNNAggr, MaskedMSELoss
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
from atomprop.models.GeAT import GeATNet
import configs.config_finetune as cfg
from atomprop.utils.utils import remove_module_prefix

# Use MSE-based loss for regression
criterion = MaskedMSELoss()


def create_dataset_from_smiles_labels(smiles_list, labels_list):
    """Create PyG dataset from SMILES and continuous labels."""
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
    # Avoid division by zero
    scale = np.where(scale == 0.0, 1.0, scale)
    valid_mask = (labels != -1)
    labels_scaled = np.where(valid_mask, (labels - mean) / scale, labels)
    return labels_scaled.astype(np.float32)


def inverse_transform_with_scaler(scaled_labels, scaler_stats):
    """
    Reverse standardization: x * scale + mean.
    Assumes missing values are NaN or already handled.
    """
    mean = scaler_stats['mean']
    scale = scaler_stats['scale']
    return scaled_labels * scale + mean


def evaluate_model(model_components, dataloader, criterion, y_cols, device, scaler_stats=None, aggr='attention'):
    """
    Evaluate model and return average loss and predictions.
    If scaler_stats is provided, inverse-transform predictions/labels for output.
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

        # Inverse transform if scaler provided (for final output only)
        if scaler_stats is not None:
            # Convert -1 to NaN for clean inverse transform
            pred_inv = inverse_transform_with_scaler(all_preds, scaler_stats)
            label_inv = inverse_transform_with_scaler(all_labels, scaler_stats)
            # Re-mask originally missing values as NaN
            pred_inv = np.where(all_labels == -1, np.nan, pred_inv)
            label_inv = np.where(all_labels == -1, np.nan, label_inv)
        else:
            pred_inv = all_preds
            label_inv = all_labels
    else:
        pred_inv = np.array([])
        label_inv = np.array([])

    return avg_loss, pred_inv, label_inv


def train(train_dataloader, val_dataloader, test_dataloader, model_components, optimizers, schedulers,
          device, num_epochs, y_cols, logdir, no_pretrain, scaler_stats, aggr='attention'):
    """
    Train the model on single GPU/CPU. Best model selected by lowest validation MSE.
    """
    embedding_layer, backbone, head, aggrmodel = model_components
    writer = SummaryWriter(log_dir=f'runs/finetune_{logdir}')

    best_val_mse = float('inf')
    best_epoch = -1
    global_step = 0

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

        # Evaluate on validation set
        val_loss, _, _ = evaluate_model(
            model_components, val_dataloader, criterion, y_cols, device, scaler_stats=None, aggr=aggr
        )

        print(f"Epoch {epoch+1} Validation MSE: {val_loss:.6f}")
        writer.add_scalar('Val/MSE', val_loss, epoch)

        # Save best model
        if val_loss < best_val_mse:
            best_val_mse = val_loss
            best_epoch = epoch + 1
            model_suffix = "nopretrain" if no_pretrain else "pretrained"
            save_path = f"trained_models/{logdir}/best_model_{model_suffix}.pth"

            torch.save({
                'epoch': epoch + 1,
                'embedding_layer_state_dict': embedding_layer.state_dict(),
                'backbone_state_dict': backbone.state_dict(),
                'head_state_dict': head.state_dict(),
                'aggr_state_dict': aggrmodel.state_dict() if aggr == 'attention' else None,
                'val_mse': val_loss,
                'scaler_stats': scaler_stats,
                'optimizer_embedding_layer_backbone_state_dict': optimizers[0].state_dict(),
                'optimizer_head_state_dict': optimizers[1].state_dict(),
                'optimizer_aggr_state_dict': optimizers[2].state_dict() if len(optimizers) > 2 else None,
            }, save_path)
            print(f"Best model saved at epoch {best_epoch} with validation MSE: {best_val_mse:.6f}")

    writer.close()

    # Final test phase
    print(f"\n--- Testing ---")
    model_suffix = "nopretrain" if no_pretrain else "pretrained"
    load_path = f"trained_models/{logdir}/best_model_{model_suffix}.pth"
    checkpoint = torch.load(load_path, weights_only=False, map_location=device)

    embedding_layer.load_state_dict(checkpoint['embedding_layer_state_dict'])
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    head.load_state_dict(checkpoint['head_state_dict'])
    if aggr == 'attention' and checkpoint['aggr_state_dict']:
        aggrmodel.load_state_dict(checkpoint['aggr_state_dict'])

    test_scaler_stats = checkpoint.get('scaler_stats', None)

    test_loss, all_test_preds, all_test_labels = evaluate_model(
        model_components, test_dataloader, criterion, y_cols, device, scaler_stats=test_scaler_stats, aggr=aggr
    )
    test_mse = test_loss

    # Save predictions
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

    print(f"Test MSE (scaled): {test_mse:.6f}")

    return {
        'best_val_mse': best_val_mse,
        'best_epoch': best_epoch,
        'test_mse': test_mse,
        'test_predictions_path': output_csv_path
    }


def main(ft_dataset=None):
    parser = argparse.ArgumentParser(description='Finetuning script')
    parser.add_argument('--dataset', type=str, default='', help='The downstream task for finetuning.')
    parser.add_argument('--bs', type=int, default=128, help='Batch size for finetuning.')
    parser.add_argument('--gpu', type=int, default=0, help='Use which GPU to finetune.')
    args = parser.parse_args()
    
    if ft_dataset is None:
        ft_dataset = args.dataset
        cfg.batch_size = args.bs
        cfg.device = f"cuda:{args.gpu}"

    if len(ft_dataset) > 0:
        cfg.set_data_path(ft_dataset)
        print(f"Finetuning on {ft_dataset}...")
    
    os.makedirs(f"trained_models/{cfg.logdir}", exist_ok=True)

    device = torch.device(cfg.device)

    # Load data
    df = pd.read_csv(cfg.data_path)
    headers = df.columns.tolist()
    y_cols = [col for col in headers if col not in cfg.exclude_list + [cfg.x_col]]
    smiles_list = df[cfg.x_col].tolist()
    df[y_cols] = df[y_cols].fillna(-1)
    labels = df[y_cols].values.astype(np.float32)

    cfg.print_all_params()

    # Prepare dataset
    dc_dataset = NumpyDataset(X=labels, ids=smiles_list)
    splitter = ScaffoldSplitter()
    all_results = []

    for run_num in range(cfg.num_runs):
        print(f"\nStarting regression run {run_num}...")

        # Initialize model
        embedding_layer = Embedder(num_atom_types=120, embed_dim=cfg.embed_dim)
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
            use_edge_embedding=cfg.use_edge_embedding
        )
        aggrmodel = GNNAggr(embed_dim=cfg.embed_dim, aggr=cfg.aggr, layers=1)
        
        head = MLP(
            input_dim=cfg.embed_dim,
            hidden_dim=cfg.head_hidden_dim,
            output_dim=len(y_cols),
            num_layers=2,
            dropout=cfg.head_dropout,
            batch_norm=True,
            output_activation=None
        )
        
        # head = DownstreamHead(
        #     input_dim=cfg.embed_dim,
        #     hidden_dim=cfg.head_hidden_dim,
        #     output_dim=len(y_cols),
        #     mlp_num_layers=2,
        #     attn_num_layers=cfg.downstream_head_attn_num_layers,
        #     dropout=cfg.head_dropout,
        #     batch_norm=True,
        #     output_activation=None
        # )

        if not cfg.no_pretrain:
            ckpt = torch.load(cfg.pretrained_path, weights_only=False, map_location=device)
            embedding_layer.load_state_dict(remove_module_prefix(ckpt['embedding_layer_state_dict']))
            backbone.load_state_dict(remove_module_prefix(ckpt['backbone_state_dict']))

        print(f"embedding_layer Parameters: {sum(p.numel() for p in embedding_layer.parameters() if p.requires_grad)}")
        print(f"backbone Parameters: {sum(p.numel() for p in backbone.parameters() if p.requires_grad)}")
        print(f"Head Parameters: {sum(p.numel() for p in head.parameters() if p.requires_grad)}")

        train_inds, valid_inds, test_inds = splitter.split(dataset=dc_dataset, seed=cfg.random_state + run_num)

        train_smiles = [dc_dataset.ids[i] for i in train_inds]
        train_labels = dc_dataset.X[train_inds]
        val_smiles = [dc_dataset.ids[i] for i in valid_inds]
        val_labels = dc_dataset.X[valid_inds]
        test_smiles = [dc_dataset.ids[i] for i in test_inds]
        test_labels = dc_dataset.X[test_inds]

        # Compute scaler stats on training set only
        scaler_stats = compute_scaler_stats(train_labels)

        # Transform all sets
        train_labels_scaled = transform_with_scaler(train_labels, scaler_stats)
        val_labels_scaled = transform_with_scaler(val_labels, scaler_stats)
        test_labels_scaled = transform_with_scaler(test_labels, scaler_stats)

        print(f"Run {run_num}: Train={len(train_smiles)}, Val={len(val_smiles)}, Test={len(test_smiles)}")

        # Create datasets
        train_dataset = create_dataset_from_smiles_labels(train_smiles, train_labels_scaled)
        val_dataset = create_dataset_from_smiles_labels(val_smiles, val_labels_scaled)
        test_dataset = create_dataset_from_smiles_labels(test_smiles, test_labels_scaled)

        # Standard DataLoaders (no DistributedSampler)
        train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, collate_fn=Batch.from_data_list, num_workers=0)
        val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, collate_fn=Batch.from_data_list, num_workers=0)
        test_dataloader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=False, collate_fn=Batch.from_data_list, num_workers=0)

        # Move to device
        embedding_layer = embedding_layer.to(device)
        backbone = backbone.to(device)
        head = head.to(device)
        aggrmodel = aggrmodel.to(device) if cfg.aggr == 'attention' else aggrmodel

        # Optimizers and schedulers
        opt_emb_backbone = torch.optim.Adam([
            {'params': embedding_layer.parameters(), 'lr': cfg.lr_embedding_layer_backbone},
            {'params': backbone.parameters(), 'lr': cfg.lr_embedding_layer_backbone}
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
                device, cfg.num_epochs, y_cols, cfg.logdir, cfg.no_pretrain, scaler_stats, cfg.aggr
            )
            all_results.append(result)
        except Exception as e:
            import traceback
            print(f"Error in run {run_num}: {e}")
            traceback.print_exc()

    # Final summary
    print(f"\n{'='*60}\nREGRESSION SUMMARY\n{'='*60}")
    if all_results:
        val_mses = [r['best_val_mse'] for r in all_results]
        test_mses = [r['test_mse'] for r in all_results]
        for i, r in enumerate(all_results):
            print(f"Run {i}: Val MSE={r['best_val_mse']:.6f}, Test MSE={r['test_mse']:.6f}")
        print(f"\nMean Test MSE: {np.mean(test_mses):.6f} ± {np.std(test_mses):.6f}")
        summary = {
            'timestamp': datetime.now().isoformat(),
            'dataset': cfg.data_path,
            'results': all_results,
            'summary_stats': {
                'mean_test_mse': float(np.mean(test_mses)),
                'std_test_mse': float(np.std(test_mses))
            }
        }
        with open(f"trained_models/{cfg.logdir}/regression_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
    else:
        print("NO VALID RUNS COMPLETED!")


if __name__ == "__main__":
    main()