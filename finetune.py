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
from atomprop.dataloader.splitter import ScaffoldSplitter
from deepchem.data import NumpyDataset
import csv
import os

no_pretrain = False

data_path = "./data/moleculenet/tox21/tox21.csv"
x_col = "smiles"

pretrained_path = 'trained_models/pretrain_pubchem_scaffold/model_epoch2.pth'

criterion = MaskedBCELoss()

logdir = "finetune_gin7"
os.makedirs(f"trained_models/{logdir}", exist_ok=True)

batch_size = 32
test_batch_size = 32

num_epochs = 100
random_state = 42

aggr = 'attention'

if __name__ == "__main__":
    ### 1. Read from CSV
    df = pd.read_csv(data_path)
    headers = df.columns.tolist()
    
    exclude_list = ["mol_id", "name", "num"]
    exclude_list.append(x_col)
    y_cols = [col for col in headers if col not in exclude_list]
    
    smiles_list = df[x_col].tolist()
    # To note that, there is 0, 1 and missing value in y_cols
    # first we need to replace missing value with -1
    df[y_cols] = df[y_cols].fillna(-1).astype(float)
    labels = df[y_cols].values.tolist()

    ### 2. Load model from saved checkpoint
    embed_dim = 384

    backbone = Embedder(num_atom_types=120, embed_dim=embed_dim)
    neck = GNN(num_layers=7, embed_dim=embed_dim, gnn_type='gin', JK='last', dropout=0)
    aggrmodel = GNNAggr(embed_dim=embed_dim, aggr=aggr, layers=1)

    if not no_pretrain:
        backbone_ckpt = torch.load(pretrained_path)['backbone_state_dict']
        neck_ckpt = torch.load(pretrained_path)['neck_state_dict']

        backbone.load_state_dict(backbone_ckpt)
        neck.load_state_dict(neck_ckpt)
    
    ### 3. Define head for finetuning
    head = MLP(input_dim=embed_dim, hidden_dim=384, output_dim=len(y_cols), num_layers=2, dropout=0.5, batch_norm=True, output_activation=None)
    head.init_params(gain=2.0)
    
    print(backbone.__class__.__name__, f"Parameters: {sum(p.numel() for p in backbone.parameters() if p.requires_grad)}")
    print(neck.__class__.__name__, f"Parameters: {sum(p.numel() for p in neck.parameters() if p.requires_grad)}")
    print(head.__class__.__name__, f"Parameters: {sum(p.numel() for p in head.parameters() if p.requires_grad)}")

    ### 4. Train the model
    ### To note that, we set a bigger learning rate for head, and a small lr for backbone and neck
    ### Record training loss and metrics using tqdm and tensorboard
    ### Use a K-fold cross validation. Set aside 5% data for testing, and use rest 95% data for training with 5-fold cross validation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = backbone.to(device)
    neck = neck.to(device)
    head = head.to(device)
    if aggr == 'attention':
        aggrmodel = aggrmodel.to(device)

    optimizer_backbone_neck = torch.optim.Adam(
        [
            {'params': backbone.parameters(), 'lr': 1e-4},
            {'params': neck.parameters(), 'lr': 1e-4}
        ]
    )
    
    optimizer_head = torch.optim.Adam(
        [{'params': head.parameters(), 'lr': 5e-4}]
    )
    
    if aggr == 'attention':
        optimizer_aggr = torch.optim.Adam(
            [{'params': aggrmodel.parameters(), 'lr': 5e-4}]
        )
        optimizers = [optimizer_backbone_neck, optimizer_head, optimizer_aggr]
    else:
        optimizers = [optimizer_backbone_neck, optimizer_head]

    scheduler_backbone_neck = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_backbone_neck, T_max=num_epochs, eta_min=1e-6)
    scheduler_head = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_head, T_max=num_epochs, eta_min=4e-6)
    
    if aggr == 'attention':
        scheduler_aggr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_aggr, T_max=num_epochs, eta_min=4e-6)
        schedulers = [scheduler_backbone_neck, scheduler_head, scheduler_aggr]
    else:
        schedulers = [scheduler_backbone_neck, scheduler_head]

    writer = SummaryWriter(log_dir='runs/finetune_experiment')

    try:
        global_step = 0
        splitter = ScaffoldSplitter()

        dc_dataset = NumpyDataset(X=labels, ids=smiles_list)
        print("whole dataset size: ", len(dc_dataset))
        
        # default split on 8:1:1
        dc_train, dc_val, dc_test = splitter.train_valid_test_split(dc_dataset, seed=random_state)

        train_smiles = dc_train.ids
        train_labels = dc_train.X
        train_dataset = []
        val_smiles = dc_val.ids
        val_labels = dc_val.X
        val_dataset = []
        test_smiles = dc_test.ids
        test_labels = dc_test.X
        test_dataset = []

        for smi, label in zip(train_smiles, train_labels):
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
            train_dataset.append(data)

        for smi, label in zip(val_smiles, val_labels):
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
            val_dataset.append(data)

        for smi, label in zip(test_smiles, test_labels):
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
            test_dataset.append(data)
            
        # print length of 3 datasets
        print("train set size: ", len(train_dataset))
        print("val set size: ", len(val_dataset))
        print("test set size: ", len(test_dataset))

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=Batch.from_data_list)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=Batch.from_data_list)
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=Batch.from_data_list)

        best_val_auc = 0.0
        best_epoch = -1

        for epoch in range(num_epochs):
            backbone.train()
            neck.train()
            head.train()
            if aggr == 'attention':
                aggrmodel.train()
            epoch_loss = 0.0
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training"):
                batch = batch.to(device)
                for optimizer in optimizers:
                    optimizer.zero_grad()
                emb = backbone(batch.x.squeeze())
                emb = neck(Data(x=emb, edge_index=batch.edge_index, edge_attr=batch.edge_attr))
                graph_emb = aggrmodel(emb, batch.batch)
                preds = head(graph_emb)
                loss = criterion(preds.reshape(-1, len(y_cols)), batch.y.reshape(-1, len(y_cols)))
                loss.backward()
                for optimizer in optimizers:
                    optimizer.step()
                epoch_loss += loss.item()
                writer.add_scalar('Train/Loss', loss.item(), global_step)
                global_step += 1
            avg_epoch_loss = epoch_loss / len(train_dataloader)
            print(f"Epoch {epoch+1} Training Loss: {avg_epoch_loss:.4f}")
            for scheduler in schedulers:
                scheduler.step()

            # Validation
            backbone.eval()
            neck.eval()
            head.eval()
            if aggr == 'attention':
                aggrmodel.eval()
            val_loss = 0.0
            all_val_preds = []
            all_val_labels = []
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1} Validation"):
                    batch = batch.to(device)
                    emb = backbone(batch.x.squeeze())
                    emb = neck(Data(x=emb, edge_index=batch.edge_index, edge_attr=batch.edge_attr))
                    graph_emb = aggrmodel(emb, batch.batch)
                    preds = head(graph_emb)
                    loss = criterion(preds.reshape(-1, len(y_cols)), batch.y.reshape(-1, len(y_cols)))
                    val_loss += loss.item()
                    preds_np = F.sigmoid(preds).cpu().numpy()
                    labels_np = batch.y.reshape(-1, len(y_cols)).cpu().numpy()
                    all_val_preds.append(preds_np)
                    all_val_labels.append(labels_np)
            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
            writer.add_scalar('Val/Loss', avg_val_loss, epoch)

            # Compute validation AUC
            if len(all_val_preds) > 0:
                all_val_preds = np.vstack(all_val_preds)
                all_val_labels = np.vstack(all_val_labels)
                val_task_aucs = []
                for col_idx in range(len(y_cols)):
                    valid_mask = all_val_labels[:, col_idx] != -1
                    if valid_mask.sum() == 0:
                        continue
                    valid_labels = all_val_labels[valid_mask, col_idx]
                    valid_preds = all_val_preds[valid_mask, col_idx]
                    if len(np.unique(valid_labels)) < 2:
                        continue
                    try:
                        auc = roc_auc_score(valid_labels, valid_preds)
                        val_task_aucs.append(auc)
                    except Exception:
                        pass
                if len(val_task_aucs) > 0:
                    mean_val_auc = np.nanmean(val_task_aucs)
                    print(f"Epoch {epoch+1} Validation AUC: {mean_val_auc:.4f}")
                    writer.add_scalar('Val/AUC', mean_val_auc, epoch)
                    
                    # Save best model based on validation AUC
                    if mean_val_auc > best_val_auc:
                        best_val_auc = mean_val_auc
                        best_epoch = epoch + 1
                        save_model_name = f'trained_models/{logdir}/best_model_nopretrain.pth' if no_pretrain else f'trained_models/{logdir}/best_model.pth'
                        torch.save({
                            'backbone_state_dict': backbone.state_dict(),
                            'neck_state_dict': neck.state_dict(),
                            'head_state_dict': head.state_dict(),
                            'aggr': aggrmodel.state_dict()
                        }, save_model_name)
                        print(f"Best model saved at epoch {best_epoch} with validation AUC: {best_val_auc:.4f}")
                else:
                    print(f"Epoch {epoch+1} Validation: No valid AUC computed")
    except KeyboardInterrupt:
        print("Training interrupted. Saving current model...")
        save_model_name = f'trained_models/{logdir}/finetuned_model_interrupted_nopretrain.pth' if no_pretrain else f'trained_models/{logdir}/finetuned_model_interrupted.pth'
        torch.save({
            'backbone_state_dict': backbone.state_dict(),
            'neck_state_dict': neck.state_dict(),
            'head_state_dict': head.state_dict(),
        }, save_model_name)
            
    writer.close()
    
    # 5. Save final model
    save_model_name = f'trained_models/{logdir}/finetuned_model_nopretrain.pth' if no_pretrain else f'trained_models/{logdir}/finetuned_model.pth'
    torch.save({
        'backbone_state_dict': backbone.state_dict(),
        'neck_state_dict': neck.state_dict(),
        'head_state_dict': head.state_dict(),
        'aggr': aggrmodel.state_dict()
    }, save_model_name)

    print(f"Best model was at epoch {best_epoch} with validation AUC: {best_val_auc:.4f}")

    # 6. Test on the test set, report ROC-AUC
    # Load best model for testing
    best_model_name = f'trained_models/{logdir}/best_model_nopretrain.pth' if no_pretrain else f'trained_models/{logdir}/best_model.pth'
    best_checkpoint = torch.load(best_model_name)
    backbone.load_state_dict(best_checkpoint['backbone_state_dict'])
    neck.load_state_dict(best_checkpoint['neck_state_dict'])
    head.load_state_dict(best_checkpoint['head_state_dict'])
    if 'aggr' in best_checkpoint:
        aggrmodel.load_state_dict(best_checkpoint['aggr'])
    
    backbone.eval()
    neck.eval()
    head.eval()
    if aggr == 'attention':
        aggrmodel.eval()
    test_loss = 0.0
    all_preds = []  # list of (N, num_tasks)
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            batch = batch.to(device)
            emb = backbone(batch.x.squeeze())
            emb = neck(Data(x=emb, edge_index=batch.edge_index, edge_attr=batch.edge_attr))
            graph_emb = aggrmodel(emb, batch.batch)
            preds = head(graph_emb)
            # sigmoid to get probabilities
            preds_np = F.sigmoid(preds).cpu().numpy()
            labels_np = batch.y.reshape(-1, len(y_cols)).cpu().numpy()
            all_preds.append(preds_np)
            all_labels.append(labels_np)

    if len(all_preds) == 0:
        print("No predictions were produced on the test set.")
        exit(1)

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # compute per-task thresholds using ROC left-top point (min distance to (0,1))

    thresholds = np.full(len(y_cols), 0.5, dtype=float)
    task_aucs = []
    for col_idx in range(len(y_cols)):
        valid_mask = all_labels[:, col_idx] != -1
        if valid_mask.sum() == 0:
            # no labels for this task in test set
            thresholds[col_idx] = 0.5
            continue
        valid_labels = all_labels[valid_mask, col_idx]
        valid_preds = all_preds[valid_mask, col_idx]
        # if only one class present, skip ROC
        if len(np.unique(valid_labels)) < 2:
            thresholds[col_idx] = 0.5
            try:
                auc = roc_auc_score(valid_labels, valid_preds)
                task_aucs.append(auc)
            except Exception:
                pass
            continue
        fpr, tpr, thr = roc_curve(valid_labels, valid_preds)
        # distance to the point (0,1)
        distances = (fpr - 0.0) ** 2 + (1.0 - tpr) ** 2
        idx = np.nanargmin(distances)
        thresholds[col_idx] = thr[idx]
        try:
            auc = roc_auc_score(valid_labels, valid_preds)
            task_aucs.append(auc)
        except Exception:
            pass

    if len(task_aucs) > 0:
        mean_auc = np.nanmean(task_aucs)
    else:
        mean_auc = float('nan')
    print("Per-task thresholds:")
    for name, t in zip(y_cols, thresholds):
        print(f"  {name}: threshold={t:.4f}")
    print("Pre-task aucs:")
    for name, auc in zip(y_cols, task_aucs):
        print(f"  {name}: AUC={auc:.4f}")
    print(f"Test ROC-AUC (mean over tasks with valid AUC): {mean_auc:.4f}")

    # open a csv write stream and write per-molecule rows: probs, predicted labels (by thresholds), true labels, empty row
    output_csv_path = "test_preds_labels.csv"
    with open(output_csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # header
        csv_writer.writerow(y_cols)

        for i in range(all_preds.shape[0]):
            row_preds = all_preds[i].tolist()
            # predicted binary results according to thresholds; for missing label keep empty string
            row_pred_results = []
            for j in range(len(y_cols)):
                if all_labels[i, j] == -1:
                    row_pred_results.append("")
                else:
                    val = 1 if all_preds[i, j] >= thresholds[j] else 0
                    row_pred_results.append(int(val))
            row_labels = all_labels[i].astype(int).tolist()
            # replace all '-1's in row_labels with empty string
            row_labels = [lbl if lbl != -1 else "" for lbl in row_labels]
            csv_writer.writerow(row_preds)
            csv_writer.writerow(row_pred_results)
            csv_writer.writerow(row_labels)
            csv_writer.writerow([])