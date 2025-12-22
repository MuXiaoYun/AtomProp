from atomprop.dataloader.dataloader import SMILESToInputs
from atomprop.models.GNNs import Embedder, GNN, GNNAggr, MaskedBCELoss, MaskedFocalLoss
from atomprop.utils.mlp import MLP
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from torch_geometric.data import Data, Batch, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, roc_curve
from atomprop.dataloader.splitter import ScaffoldKFoldSplitter
from deepchem.data import NumpyDataset
import csv
import os
import json
from datetime import datetime
from atomprop.models.GeAT import GeATNet
from configs.config_finetune import (
    data_path, x_col, exclude_list,
    no_pretrain, pretrained_path, logdir,
    batch_size, test_batch_size, num_epochs, random_state,
    k_folds, frac_test,
    embed_dim, aggr,
    lr_backbone_neck, lr_head, lr_aggr,
    T_max, eta_min_backbone_neck, eta_min_head, eta_min_aggr,
    device_id
)

criterion = MaskedBCELoss()

os.makedirs(f"trained_models/{logdir}", exist_ok=True)


def create_dataset_from_smiles_labels(smiles_list, labels_list):
    """Create PyG dataset from SMILES and labels"""
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
            
        data = Data(x=atom_info, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor(label))
        dataset.append(data)
    
    return dataset


def evaluate_model(model_components, dataloader, criterion, y_cols, device, aggr='attention'):
    """Evaluate model performance"""
    backbone, neck, head, aggrmodel = model_components
    
    backbone.eval()
    neck.eval()
    head.eval()
    if aggr == 'attention':
        aggrmodel.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            emb = backbone(batch.x.squeeze())
            emb = neck(Data(x=emb, edge_index=batch.edge_index, edge_attr=batch.edge_attr), batch=batch.batch)
            graph_emb = aggrmodel(emb, batch.batch)
            preds = head(graph_emb)
            
            loss = criterion(preds.reshape(-1, len(y_cols)), batch.y.reshape(-1, len(y_cols)))
            total_loss += loss.item()
            
            preds_np = F.sigmoid(preds).cpu().numpy()
            labels_np = batch.y.reshape(-1, len(y_cols)).cpu().numpy()
            all_preds.append(preds_np)
            all_labels.append(labels_np)
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    
    if len(all_preds) > 0:
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        task_aucs = []
        
        for col_idx in range(len(y_cols)):
            valid_mask = all_labels[:, col_idx] != -1
            if valid_mask.sum() == 0:
                continue
            
            valid_labels = all_labels[valid_mask, col_idx]
            valid_preds = all_preds[valid_mask, col_idx]
            
            if len(np.unique(valid_labels)) < 2:
                continue
            
            try:
                auc = roc_auc_score(valid_labels, valid_preds)
                task_aucs.append(auc)
            except Exception:
                pass
        
        mean_auc = np.nanmean(task_aucs) if len(task_aucs) > 0 else float('nan')
    else:
        mean_auc = float('nan')
        task_aucs = []
    
    return avg_loss, mean_auc, task_aucs, all_preds, all_labels


def train_fold(fold_idx, train_dataloader, val_dataloader, test_dataloader, 
               model_components, optimizers, schedulers, device, 
               num_epochs, y_cols, logdir, no_pretrain, aggr='attention'):
    """Train a single fold"""
    backbone, neck, head, aggrmodel = model_components
    writer = SummaryWriter(log_dir=f'runs/finetune_kfold_fold{fold_idx}')
    
    best_val_auc = 0.0
    best_epoch = -1
    fold_global_step = 0
    
    for epoch in range(num_epochs):
        backbone.train()
        neck.train()
        head.train()
        if aggr == 'attention':
            aggrmodel.train()
        
        epoch_loss = 0.0
        
        for batch in tqdm(train_dataloader, desc=f"Fold {fold_idx}, Epoch {epoch+1} Training"):
            batch = batch.to(device)
            
            for optimizer in optimizers:
                optimizer.zero_grad()
            
            emb = backbone(batch.x.squeeze())
            emb = neck(Data(x=emb, edge_index=batch.edge_index, edge_attr=batch.edge_attr), batch=batch.batch)
            graph_emb = aggrmodel(emb, batch.batch)
            preds = head(graph_emb)
            
            loss = criterion(preds.reshape(-1, len(y_cols)), batch.y.reshape(-1, len(y_cols)))
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            
            epoch_loss += loss.item()
            writer.add_scalar(f'Fold{fold_idx}/Train/Loss', loss.item(), fold_global_step)
            fold_global_step += 1
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Fold {fold_idx}, Epoch {epoch+1} Training Loss: {avg_epoch_loss:.4f}")
        
        for scheduler in schedulers:
            scheduler.step()
        
        val_loss, val_auc, _, _, _ = evaluate_model(
            model_components, val_dataloader, criterion, y_cols, device, aggr
        )
        print(f"Fold {fold_idx}, Epoch {epoch+1} Validation Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")
        writer.add_scalar(f'Fold{fold_idx}/Val/Loss', val_loss, epoch)
        writer.add_scalar(f'Fold{fold_idx}/Val/AUC', val_auc, epoch)
        
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            
            model_suffix = "nopretrain" if no_pretrain else "pretrained"
            save_path = f"trained_models/{logdir}/fold{fold_idx}_best_model_{model_suffix}.pth"
            
            torch.save({
                'epoch': epoch + 1,
                'fold': fold_idx,
                'backbone_state_dict': backbone.state_dict(),
                'neck_state_dict': neck.state_dict(),
                'head_state_dict': head.state_dict(),
                'aggr_state_dict': aggrmodel.state_dict() if aggr == 'attention' else None,
                'val_auc': val_auc,
                'val_loss': val_loss,
                'optimizer_backbone_neck_state_dict': optimizers[0].state_dict(),
                'optimizer_head_state_dict': optimizers[1].state_dict(),
                'optimizer_aggr_state_dict': optimizers[2].state_dict() if len(optimizers) > 2 else None,
            }, save_path)
            
            print(f"Fold {fold_idx}: Best model saved at epoch {best_epoch} with validation AUC: {best_val_auc:.4f}")
    
    writer.close()
    
    # Test phase
    print(f"\n--- Testing Fold {fold_idx} ---")
    model_suffix = "nopretrain" if no_pretrain else "pretrained"
    load_path = f"trained_models/{logdir}/fold{fold_idx}_best_model_{model_suffix}.pth"
    checkpoint = torch.load(load_path, weights_only=False)
    
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    neck.load_state_dict(checkpoint['neck_state_dict'])
    head.load_state_dict(checkpoint['head_state_dict'])
    if aggr == 'attention' and checkpoint['aggr_state_dict']:
        aggrmodel.load_state_dict(checkpoint['aggr_state_dict'])
    
    test_loss, test_auc, test_task_aucs, all_test_preds, all_test_labels = evaluate_model(
        model_components, test_dataloader, criterion, y_cols, device, aggr
    )
    
    print(f"Fold {fold_idx} Test Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Mean Test AUC: {test_auc:.4f}")
    
    output_csv_path = f"trained_models/{logdir}/fold{fold_idx}_test_predictions.csv"
    if len(all_test_preds) > 0:
        all_test_preds = np.vstack(all_test_preds)
        all_test_labels = np.vstack(all_test_labels)
        
        with open(output_csv_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(y_cols)
            
            for i in range(all_test_preds.shape[0]):
                row_preds = all_test_preds[i].tolist()
                row_labels = all_test_labels[i].astype(int).tolist()
                row_labels = [lbl if lbl != -1 else "" for lbl in row_labels]
                csv_writer.writerow(row_preds)
                csv_writer.writerow(row_labels)
                csv_writer.writerow([])
    
    return {
        'fold': fold_idx,
        'best_val_auc': best_val_auc,
        'best_epoch': best_epoch,
        'test_auc': test_auc,
        'test_loss': test_loss,
        'test_task_aucs': test_task_aucs,
        'test_predictions_path': output_csv_path
    }


if __name__ == "__main__":
    # 1. Load data
    df = pd.read_csv(data_path)
    headers = df.columns.tolist()
    y_cols = [col for col in headers if col not in exclude_list + [x_col]]
    
    smiles_list = df[x_col].tolist()
    df[y_cols] = df[y_cols].fillna(-1).astype(float)
    labels = df[y_cols].values.tolist()
    
    # 2. Initialize base model components (for weight loading if needed)
    backbone = Embedder(num_atom_types=120, embed_dim=embed_dim)
    neck = GeATNet(embed_dim=embed_dim, dropout=0.5)
    aggrmodel = GNNAggr(embed_dim=embed_dim, aggr=aggr, layers=1)
    
    if no_pretrain == False:
        ckpt = torch.load(pretrained_path, weights_only=False)
        backbone.load_state_dict(ckpt['backbone_state_dict'])
        neck.load_state_dict(ckpt['neck_state_dict'])
    
    head = MLP(input_dim=embed_dim, hidden_dim=384, output_dim=len(y_cols),
               num_layers=2, dropout=0.5, batch_norm=True, output_activation=None)
    head.init_params(gain=2.0)
    
    print(f"Backbone Parameters: {sum(p.numel() for p in backbone.parameters() if p.requires_grad)}")
    print(f"Neck Parameters: {sum(p.numel() for p in neck.parameters() if p.requires_grad)}")
    print(f"Head Parameters: {sum(p.numel() for p in head.parameters() if p.requires_grad)}")
    
    # 3. Prepare dataset and splitter
    dc_dataset = NumpyDataset(X=labels, ids=smiles_list)
    splitter = ScaffoldKFoldSplitter(fold=k_folds, frac_test=frac_test)
    
    # 4. Set device
    if torch.cuda.is_available() and device_id is not None and device_id >= 0:
        device = torch.device(f'cuda:{device_id}')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # 5. K-Fold CV
    all_fold_results = []
    print(f"\nStarting {k_folds}-Fold Cross-Validation...")
    print(f"Test set fraction: {frac_test}")
    print(f"Training/Validation fraction: {1 - frac_test}\n")
    
    fold_idx = 0
    for train_inds, valid_inds, test_inds in splitter.k_fold_split(dc_dataset):
        print(f"\n{'='*60}")
        print(f"Training Fold {fold_idx + 1}/{k_folds}")
        print(f"{'='*60}")
        
        train_smiles = [dc_dataset.ids[i] for i in train_inds]
        train_labels = dc_dataset.X[train_inds]
        val_smiles = [dc_dataset.ids[i] for i in valid_inds]
        val_labels = dc_dataset.X[valid_inds]
        test_smiles = [dc_dataset.ids[i] for i in test_inds]
        test_labels = dc_dataset.X[test_inds]
        
        print(f"Training set size: {len(train_smiles)}")
        print(f"Validation set size: {len(val_smiles)}")
        print(f"Test set size: {len(test_smiles)}")
        
        train_dataset = create_dataset_from_smiles_labels(train_smiles, train_labels)
        val_dataset = create_dataset_from_smiles_labels(val_smiles, val_labels)
        test_dataset = create_dataset_from_smiles_labels(test_smiles, test_labels)
        
        print(f"Valid training graphs: {len(train_dataset)}")
        print(f"Valid validation graphs: {len(val_dataset)}")
        print(f"Valid test graphs: {len(test_dataset)}")
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=Batch.from_data_list)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=Batch.from_data_list)
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=Batch.from_data_list)
        
        # Clone models per fold
        backbone_fold = Embedder(num_atom_types=120, embed_dim=embed_dim)
        neck_fold = GeATNet(embed_dim=embed_dim, dropout=0.5)
        aggrmodel_fold = GNNAggr(embed_dim=embed_dim, aggr=aggr, layers=1)
        head_fold = MLP(input_dim=embed_dim, hidden_dim=384, output_dim=len(y_cols),
                        num_layers=2, dropout=0.5, batch_norm=True, output_activation=None)
        head_fold.init_params(gain=2.0)
        
        if not no_pretrain:
            backbone_fold.load_state_dict(backbone.state_dict())
            neck_fold.load_state_dict(neck.state_dict())
        
        backbone_fold = backbone_fold.to(device)
        neck_fold = neck_fold.to(device)
        head_fold = head_fold.to(device)
        if aggr == 'attention':
            aggrmodel_fold = aggrmodel_fold.to(device)
        
        # Optimizers
        optimizer_backbone_neck = torch.optim.Adam([
            {'params': backbone_fold.parameters(), 'lr': lr_backbone_neck},
            {'params': neck_fold.parameters(), 'lr': lr_backbone_neck}
        ])
        optimizer_head = torch.optim.Adam(head_fold.parameters(), lr=lr_head)
        optimizers = [optimizer_backbone_neck, optimizer_head]
        
        if aggr == 'attention':
            optimizer_aggr = torch.optim.Adam(aggrmodel_fold.parameters(), lr=lr_aggr)
            optimizers.append(optimizer_aggr)
        
        # Schedulers
        scheduler_backbone_neck = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_backbone_neck, T_max=T_max, eta_min=eta_min_backbone_neck
        )
        scheduler_head = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_head, T_max=T_max, eta_min=eta_min_head
        )
        schedulers = [scheduler_backbone_neck, scheduler_head]
        
        if aggr == 'attention':
            scheduler_aggr = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_aggr, T_max=T_max, eta_min=eta_min_aggr
            )
            schedulers.append(scheduler_aggr)
        
        model_components = (backbone_fold, neck_fold, head_fold, aggrmodel_fold)
        
        try:
            fold_result = train_fold(
                fold_idx=fold_idx,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                test_dataloader=test_dataloader,
                model_components=model_components,
                optimizers=optimizers,
                schedulers=schedulers,
                device=device,
                num_epochs=num_epochs,
                y_cols=y_cols,
                logdir=logdir,
                no_pretrain=no_pretrain,
                aggr=aggr
            )
            all_fold_results.append(fold_result)
        except KeyboardInterrupt:
            print(f"\nTraining interrupted during Fold {fold_idx}. Saving current progress...")
            break
        except Exception as e:
            print(f"\nError during Fold {fold_idx}: {e}")
            continue
        
        fold_idx += 1
    
    # 6. Summary
    print(f"\n{'='*60}")
    print("K-FOLD CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    if len(all_fold_results) > 0:
        val_aucs = []
        test_aucs = []
        print("\nIndividual Fold Results:")
        print("-" * 40)
        for result in all_fold_results:
            print(f"Fold {result['fold']}:")
            print(f"  Best Validation AUC: {result['best_val_auc']:.4f} (epoch {result['best_epoch']})")
            print(f"  Test AUC: {result['test_auc']:.4f}")
            print(f"  Test Loss: {result['test_loss']:.4f}\n")
            val_aucs.append(result['best_val_auc'])
            test_aucs.append(result['test_auc'])
        
        print("\nSummary Statistics:")
        print("-" * 40)
        print(f"Number of completed folds: {len(all_fold_results)}")
        print(f"Mean Validation AUC: {np.mean(val_aucs):.4f} ± {np.std(val_aucs):.4f}")
        print(f"Mean Test AUC: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
        print(f"Min Test AUC: {np.min(test_aucs):.4f}")
        print(f"Max Test AUC: {np.max(test_aucs):.4f}")
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'dataset': data_path,
            'k_folds': k_folds,
            'frac_test': frac_test,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'no_pretrain': no_pretrain,
            'aggr': aggr,
            'fold_results': all_fold_results,
            'summary_stats': {
                'mean_val_auc': float(np.mean(val_aucs)),
                'std_val_auc': float(np.std(val_aucs)),
                'mean_test_auc': float(np.mean(test_aucs)),
                'std_test_auc': float(np.std(test_aucs)),
                'min_test_auc': float(np.min(test_aucs)),
                'max_test_auc': float(np.max(test_aucs)),
            }
        }
        
        summary_path = f"trained_models/{logdir}/kfold_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")
    else:
        print("No folds were completed successfully.")
    
    print(f"\nK-Fold cross-validation completed. Results saved in 'trained_models/{logdir}/'")