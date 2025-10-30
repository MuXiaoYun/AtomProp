from atomprop.dataloader.dataloader import SMILESToInputs
from atomprop.models.GNNs import Embedder, GNN, GNNAggr, MaskedBCELoss, MaskedFocalLoss
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
from sklearn.metrics import roc_auc_score
from deepchem.splits.splitters import ScaffoldSplitter

data_path = "./data/moleculenet/tox21/tox21.csv"
batch_size = 64
test_batch_size = 256
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
split_method = 'S-K-fold'

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
    neck = GNN(num_layers=6, embed_dim=embed_dim, gnn_type='gcn', JK='last', dropout=0.1)
    aggrmodel = GNNAggr(embed_dim=embed_dim, aggr='mean')

    backbone_ckpt = torch.load('trained_models/ver1_1030.pth')['backbone_state_dict']
    neck_ckpt = torch.load('trained_models/ver1_1030.pth')['neck_state_dict']

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
    criterion = MaskedBCELoss()
    # params of convert() 
    # smiles: str, context_length: int = 420, edge_output_type = 'edge_list', padding = False
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

    writer = SummaryWriter(log_dir='runs/finetune_experiment')
    num_epochs = 1
    random_state=42

    if split_method == 'S-K-fold':
        K = 5

        global_step = 0
        try:
            # Try to use iterative stratification for multilabel data
            try:
                from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
                splitter = MultilabelStratifiedKFold(n_splits=K, shuffle=True, random_state=random_state)
                splits = list(splitter.split(np.zeros(len(created_labels)), np.array(created_labels)))
            except Exception:
                # Fallback: collapse multilabel into a single aggregated label for stratification (imperfect)
                from sklearn.model_selection import StratifiedKFold
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
            torch.save({
                'backbone_state_dict': backbone.state_dict(),
                'neck_state_dict': neck.state_dict(),
                'head_state_dict': head.state_dict(),
            }, 'trained_models/finetuned_model_interrupted.pth')
    elif split_method == 'scaffold':
        try:
            global_step = 0
            splitter = ScaffoldSplitter()
            #split on 8:1:1
            train_val_dataset, test_dataset = splitter.train_test_split(dataset, frac_train=0.9, random_state=random_state)
            train_dataset, val_dataset = splitter.train_test_split(train_val_dataset, frac_train=0.8889, random_state=random_state)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=Batch.from_data_list)
            val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=Batch.from_data_list)
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
            torch.save({
                'backbone_state_dict': backbone.state_dict(),
                'neck_state_dict': neck.state_dict(),
                'head_state_dict': head.state_dict(),
            }, 'trained_models/finetuned_model_interrupted.pth')
    else:
        raise ValueError(f"Unknown split method: {split_method}")
    writer.close()
    
    # 5. Save final model
    torch.save({
        'backbone_state_dict': backbone.state_dict(),
        'neck_state_dict': neck.state_dict(),
        'head_state_dict': head.state_dict(),
    }, 'trained_models/finetuned_model.pth')

    # 6. Test on the test set, report ROC-AUC
    test_size = int(0.05 * len(dataset))
    test_dataset = dataset[-test_size:]
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=Batch.from_data_list)
    backbone.eval()
    neck.eval()
    head.eval()
    test_loss = 0.0
    total_aucs = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            batch = batch.to(device)
            emb = backbone(batch.x.squeeze())
            emb = neck(Data(x=emb, edge_index=batch.edge_index))
            graph_emb = aggrmodel(emb, batch.batch)
            preds = head(graph_emb)
            loss = criterion(preds.reshape(-1, len(y_cols)), batch.y.reshape(-1, len(y_cols)))
            test_loss += loss.item()
            aucs = []
            for i in range(len(y_cols)):
                valid_mask = batch.y.reshape(-1, len(y_cols))[:, i] != -1
                if valid_mask.sum().item() == 0:
                    continue
                valid_labels = batch.y.reshape(-1, len(y_cols))[valid_mask, i].cpu().numpy()
                valid_preds = preds[valid_mask, i].cpu().numpy()
                try:
                    auc = roc_auc_score(valid_labels, valid_preds)
                    # if nan do not append
                    if np.isnan(auc):
                        continue
                    aucs.append(auc)
                except ValueError:
                    continue
            if len(aucs) > 0:
                total_aucs.append(np.mean(aucs))
            else:
                print("No valid AUCs for this batch! Skipping...")
    avg_test_loss = test_loss / len(test_dataloader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    mean_auc = np.mean(aucs)
    print(f"Test ROC-AUC: {mean_auc:.4f}")