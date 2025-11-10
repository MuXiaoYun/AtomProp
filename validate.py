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
from sklearn.metrics import roc_auc_score
from deepchem.splits.splitters import ScaffoldSplitter
import csv

model_path = "trained_models/ft_scaffold.pth"
data_path = "./data/moleculenet/tox21/tox21.csv"
batch_size = 64
test_batch_size = 256
embed_dim = 384
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

    backbone = Embedder(num_atom_types=120, embed_dim=embed_dim)
    neck = GNN(num_layers=6, embed_dim=embed_dim, gnn_type='gcn', JK='last', dropout=0.1)
    aggrmodel = GNNAggr(embed_dim=embed_dim, aggr='mean')

    backbone_ckpt = torch.load(model_path)['backbone_state_dict']
    neck_ckpt = torch.load(model_path)['neck_state_dict']
    head_ckpt = torch.load(model_path)['head_state_dict']

    backbone.load_state_dict(backbone_ckpt)
    neck.load_state_dict(neck_ckpt)
    
    ### 3. Define head for finetuning
    head = MLP(input_dim=embed_dim, hidden_dim=512, output_dim=len(y_cols), num_layers=3, dropout=0.1, batch_norm=True, output_activation=None)
    head.load_state_dict(head_ckpt)

    ### 4. Test the model on the test set
    ### Select the left-top point in ROC curve as threshold for classification
    ### Output the preds and labels into  an .csv for analysis
    ### Except for the header, the 1st row is the pred values (0.523...), the 2nd row is the pred results (0 or 1), the 3rd row is labels, and the 4th row is empty.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = backbone.to(device)
    neck = neck.to(device)
    head = head.to(device)
    backbone.eval()
    neck.eval()
    head.eval()

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

    random_state=42

    test_size = int(0.05 * len(dataset))
    test_dataset = dataset[-test_size:]
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=Batch.from_data_list)

    # We'll collect all predictions and labels across the test set first,
    # then compute per-task ROC and choose the left-top point as threshold.
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
    from sklearn.metrics import roc_curve

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