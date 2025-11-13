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
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from tqdm import tqdm
from torch_geometric.data import Data, Batch, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from deepchem.splits.splitters import ScaffoldSplitter
from deepchem.data import NumpyDataset
import csv

no_pretrain = True

data_path = "./data/moleculenet/tox21/tox21.csv"
pretrained_path = 'trained_models/ft_scaffold.pth'
batch_size = 64
test_batch_size = 256

num_epochs = 100
random_state = 42

x_col = "smiles"
y_cols = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]

split_methods = ['S-K-fold', 'scaffold']
split_method = 'scaffold'

if __name__ == "__main__":
    ### 1. Read from CSV
    df = pd.read_csv(data_path)
    smiles_list = df[x_col].tolist()
    # To note that, there is 0, 1 and missing value in y_cols
    # first we need to replace missing value with -1
    df[y_cols] = df[y_cols].fillna(-1).astype(float)
    labels = df[y_cols].values.tolist()

    ### 2. Load model from saved checkpoint
    embed_dim = 384

    backbone = Embedder(num_atom_types=120, embed_dim=embed_dim)
    neck = GNN(num_layers=3, embed_dim=embed_dim, gnn_type='gcn', JK='last', dropout=0.1)
    aggrmodel = GNNAggr(embed_dim=embed_dim, aggr='mean')

    if not no_pretrain:
        backbone_ckpt = torch.load(pretrained_path)['backbone_state_dict']
        neck_ckpt = torch.load(pretrained_path)['neck_state_dict']

        backbone.load_state_dict(backbone_ckpt)
        neck.load_state_dict(neck_ckpt)
    
    ### 3. Define head for finetuning
    head = MLP(input_dim=embed_dim, hidden_dim=512, output_dim=len(y_cols), num_layers=3, dropout=0.1, batch_norm=True, output_activation=None)
    head.init_params(gain=2.0)

    ### 4. Train the model
    ### To note that, we set a bigger learning rate for head, and a small lr for backbone and neck
    ### Record training loss and metrics using tqdm and tensorboard
    ### Use a K-fold cross validation. Set aside 5% data for testing, and use rest 95% data for training with 5-fold cross validation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = backbone.to(device)
    neck = neck.to(device)
    head = head.to(device)
    backbone.train()
    neck.train()
    head.train()
    optimizer = torch.optim.Adam([
        {'params': backbone.parameters(), 'lr': 1e-3},
        {'params': neck.parameters(), 'lr': 1e-3},
        {'params': head.parameters(), 'lr': 5e-3}
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)
    criterion = MaskedFocalLoss()

    writer = SummaryWriter(log_dir='runs/finetune_experiment')

    if split_method == 'S-K-fold':
        K = 5

        dataset = []
        created_labels = []
        for smi, label in zip(smiles_list, labels):
            atom_type_indices, edge_index, mol = SMILESToInputs.convert(smi, edge_output_type='edge_list', padding=False, sanitize=False)
            if atom_type_indices is None or edge_index is None:
                continue
            data = Data(x=atom_type_indices.unsqueeze(1), edge_index=edge_index.t().contiguous()[:2], y=torch.tensor(label))
            dataset.append(data)
            # prepare label for stratification: convert -1 -> 0 and treat positives as 1
            arr = np.array(label, dtype=float)
            binarized = (arr == 1.0).astype(int)
            created_labels.append(binarized)

        global_step = 0
        try:
            # Try to use iterative stratification for multilabel data
            try:
                splitter = MultilabelStratifiedKFold(n_splits=K, shuffle=True, random_state=random_state)
                splits = list(splitter.split(np.zeros(len(created_labels)), np.array(created_labels)))
            except Exception:
                # Fallback: collapse multilabel into a single aggregated label for stratification (imperfect)
                collapsed = []
                for lbl in created_labels:
                    # collapse binary vector to a string/tuple and map to int class
                    collapsed.append(tuple(int(x) for x in lbl))
                lbl_map = {}
                collapsed_ints = []
                for t in collapsed:
                    if t not in lbl_map:
                        lbl_map[t] = len(lbl_map)
                    collapsed_ints.append(lbl_map[t])
                splitter = StratifiedKFold(n_splits=K, shuffle=True, random_state=random_state)
                splits = list(splitter.split(np.zeros(len(created_labels)), np.array(collapsed_ints)))

            for fold_idx, (train_idx, val_idx) in enumerate(splits):
                k = fold_idx
                print(f"Starting fold {k+1}/{K}")
                train_dataset = [dataset[i] for i in train_idx]
                val_dataset = [dataset[i] for i in val_idx]
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=Batch.from_data_list)
                val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=Batch.from_data_list)

                for epoch in range(num_epochs):
                    backbone.train()
                    neck.train()
                    head.train()
                    epoch_loss = 0.0
                    for batch in tqdm(train_dataloader, desc=f"Fold {k+1} Epoch {epoch+1} Training"):
                        batch = batch.to(device)
                        optimizer.zero_grad()
                        emb = backbone(batch.x.squeeze())
                        emb = neck(Data(x=emb, edge_index=batch.edge_index))
                        graph_emb = aggrmodel(emb, batch.batch)
                        preds = head(graph_emb)
                        loss = criterion(preds.reshape(-1, len(y_cols)), batch.y.reshape(-1, len(y_cols)))
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                        writer.add_scalar('Train/Loss', loss.item(), global_step)
                        global_step += 1
                    avg_epoch_loss = epoch_loss / len(train_dataloader)
                    print(f"Fold {k+1} Epoch {epoch+1} Training Loss: {avg_epoch_loss:.4f}")
                    scheduler.step()

                    # Validation
                    backbone.eval()
                    neck.eval()
                    head.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for batch in tqdm(val_dataloader, desc=f"Fold {k+1} Epoch {epoch+1} Validation"):
                            batch = batch.to(device)
                            emb = backbone(batch.x.squeeze())
                            emb = neck(Data(x=emb, edge_index=batch.edge_index))
                            graph_emb = aggrmodel(emb, batch.batch)
                            preds = head(graph_emb)
                            loss = criterion(preds.reshape(-1, len(y_cols)), batch.y.reshape(-1, len(y_cols)))
                            val_loss += loss.item()
                    avg_val_loss = val_loss / len(val_dataloader)
                    print(f"Fold {k+1} Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
                    writer.add_scalar('Val/Loss', avg_val_loss, epoch + k * num_epochs)
        except KeyboardInterrupt:
            print("Training interrupted. Saving current model...")
            save_model_name = 'trained_models/finetuned_model_interrupted_nopretrain.pth' if no_pretrain else 'trained_models/finetuned_model_interrupted.pth'
            torch.save({
                'backbone_state_dict': backbone.state_dict(),
                'neck_state_dict': neck.state_dict(),
                'head_state_dict': head.state_dict(),
            }, save_model_name)

    elif split_method == 'scaffold':
        try:
            global_step = 0
            splitter = ScaffoldSplitter()

            dc_dataset = NumpyDataset(X=labels, ids=smiles_list)
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
                atom_type_indices, edge_index, mol = SMILESToInputs.convert(smi, edge_output_type='edge_list', padding=False, sanitize=False)
                if atom_type_indices is None or edge_index is None:
                    continue
                data = Data(x=atom_type_indices.unsqueeze(1), edge_index=edge_index.t().contiguous()[:2], y=torch.tensor(label))
                train_dataset.append(data)

            for smi, label in zip(val_smiles, val_labels):
                atom_type_indices, edge_index, mol = SMILESToInputs.convert(smi, edge_output_type='edge_list', padding=False, sanitize=False)
                if atom_type_indices is None or edge_index is None:
                    continue
                data = Data(x=atom_type_indices.unsqueeze(1), edge_index=edge_index.t().contiguous()[:2], y=torch.tensor(label))
                val_dataset.append(data)

            for smi, label in zip(test_smiles, test_labels):
                atom_type_indices, edge_index, mol = SMILESToInputs.convert(smi, edge_output_type='edge_list', padding=False, sanitize=False)
                if atom_type_indices is None or edge_index is None:
                    continue
                data = Data(x=atom_type_indices.unsqueeze(1), edge_index=edge_index.t().contiguous()[:2], y=torch.tensor(label))
                test_dataset.append(data)

            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=Batch.from_data_list)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=Batch.from_data_list)
            test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=Batch.from_data_list)

            for epoch in range(num_epochs):
                backbone.train()
                neck.train()
                head.train()
                epoch_loss = 0.0
                for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training"):
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    emb = backbone(batch.x.squeeze())
                    emb = neck(Data(x=emb, edge_index=batch.edge_index))
                    graph_emb = aggrmodel(emb, batch.batch)
                    preds = head(graph_emb)
                    loss = criterion(preds.reshape(-1, len(y_cols)), batch.y.reshape(-1, len(y_cols)))
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    writer.add_scalar('Train/Loss', loss.item(), global_step)
                    global_step += 1
                avg_epoch_loss = epoch_loss / len(train_dataloader)
                print(f"Epoch {epoch+1} Training Loss: {avg_epoch_loss:.4f}")
                scheduler.step()

                # Validation
                backbone.eval()
                neck.eval()
                head.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1} Validation"):
                        batch = batch.to(device)
                        emb = backbone(batch.x.squeeze())
                        emb = neck(Data(x=emb, edge_index=batch.edge_index))
                        graph_emb = aggrmodel(emb, batch.batch)
                        preds = head(graph_emb)
                        loss = criterion(preds.reshape(-1, len(y_cols)), batch.y.reshape(-1, len(y_cols)))
                        val_loss += loss.item()
                avg_val_loss = val_loss / len(val_dataloader)
                print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
                writer.add_scalar('Val/Loss', avg_val_loss, epoch)
        except KeyboardInterrupt:
            print("Training interrupted. Saving current model...")
            save_model_name = 'trained_models/finetuned_model_interrupted_nopretrain.pth' if no_pretrain else 'trained_models/finetuned_model_interrupted.pth'
            torch.save({
                'backbone_state_dict': backbone.state_dict(),
                'neck_state_dict': neck.state_dict(),
                'head_state_dict': head.state_dict(),
            }, save_model_name)
    else:
        raise ValueError(f"Unknown split method: {split_method}")
    writer.close()
    
    # 5. Save final model
    save_model_name = 'trained_models/finetuned_model_nopretrain.pth' if no_pretrain else 'trained_models/finetuned_model.pth'
    torch.save({
        'backbone_state_dict': backbone.state_dict(),
        'neck_state_dict': neck.state_dict(),
        'head_state_dict': head.state_dict(),
    }, save_model_name)

    # 6. Test on the test set, report ROC-AUC
    if split_method == 'S-K-fold':
        test_size = int(0.05 * len(dataset))
        test_dataset = dataset[-test_size:]
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=Batch.from_data_list)
    
    backbone.eval()
    neck.eval()
    head.eval()
    test_loss = 0.0
    all_preds = []  # list of (N, num_tasks)
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            batch = batch.to(device)
            emb = backbone(batch.x.squeeze())
            emb = neck(Data(x=emb, edge_index=batch.edge_index))
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
        mean_auc = np.mean(task_aucs)
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