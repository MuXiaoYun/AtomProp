from atomprop.dataloader.dataloader import SMILESToInputs
from atomprop.models.GNNs import Embedder, GNN, GNNAggr
from atomprop.utils.mlp import MLP
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.data import Data, Batch, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from deepchem.splits.splitters import ScaffoldSplitter
from deepchem.data import NumpyDataset
import csv

model_path = "trained_models/finetuned_model_nopretrain.pth"
data_path = "./data/moleculenet/tox21/tox21.csv"
test_batch_size = 256
embed_dim = 384
x_col = "smiles"
y_cols = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", 
    "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", 
    "SR-HSE", "SR-MMP", "SR-p53",
]
random_state = 42

split_methods = ['S-K-fold', 'scaffold']
split_method = 'scaffold'

if __name__ == "__main__":
    df = pd.read_csv(data_path)
    smiles_list = df[x_col].tolist()
    df[y_cols] = df[y_cols].fillna(-1).astype(float)
    labels = df[y_cols].values.tolist()

    backbone = Embedder(num_atom_types=120, embed_dim=embed_dim)
    neck = GNN(num_layers=3, embed_dim=embed_dim, gnn_type='gcn', JK='last', dropout=0.1)
    aggrmodel = GNNAggr(embed_dim=embed_dim, aggr='mean')

    backbone_ckpt = torch.load(model_path)['backbone_state_dict']
    neck_ckpt = torch.load(model_path)['neck_state_dict']
    head_ckpt = torch.load(model_path)['head_state_dict']

    backbone.load_state_dict(backbone_ckpt)
    neck.load_state_dict(neck_ckpt)
    
    head = MLP(input_dim=embed_dim, hidden_dim=512, output_dim=len(y_cols), num_layers=3, dropout=0.1, batch_norm=True, output_activation=None)
    head.load_state_dict(head_ckpt)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = backbone.to(device)
    neck = neck.to(device)
    head = head.to(device)
    backbone.eval()
    neck.eval()
    head.eval()

    dataset = []
    for smi, label in zip(smiles_list, labels):
        atom_type_indices, edge_index, mol = SMILESToInputs.convert(smi, edge_output_type='edge_list', padding=False, sanitize=False)
        if atom_type_indices is None or edge_index is None:
            continue
        data = Data(x=atom_type_indices.unsqueeze(1), edge_index=edge_index.t().contiguous()[:2], y=torch.tensor(label))
        dataset.append(data)

    if split_method == 'S-K-fold':
        test_size = int(0.05 * len(dataset))
        test_dataset = dataset[-test_size:]
        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=Batch.from_data_list)
    
    elif split_method == 'scaffold':
        splitter = ScaffoldSplitter()
        dc_dataset = NumpyDataset(X=labels, ids=smiles_list)
        dc_train, dc_val, dc_test = splitter.train_valid_test_split(dc_dataset, seed=random_state)
        
        test_smiles = dc_test.ids
        test_labels = dc_test.X
        test_dataset = []

        for smi, label in zip(test_smiles, test_labels):
            atom_type_indices, edge_index, mol = SMILESToInputs.convert(smi, edge_output_type='edge_list', padding=False, sanitize=False)
            if atom_type_indices is None or edge_index is None:
                continue
            data = Data(x=atom_type_indices.unsqueeze(1), edge_index=edge_index.t().contiguous()[:2], y=torch.tensor(label))
            test_dataset.append(data)

        test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, collate_fn=Batch.from_data_list)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            batch = batch.to(device)
            emb = backbone(batch.x.squeeze())
            emb = neck(Data(x=emb, edge_index=batch.edge_index))
            graph_emb = aggrmodel(emb, batch.batch)
            preds = head(graph_emb)
            preds_np = F.sigmoid(preds).cpu().numpy()
            labels_np = batch.y.reshape(-1, len(y_cols)).cpu().numpy()
            all_preds.append(preds_np)
            all_labels.append(labels_np)

    if len(all_preds) == 0:
        print("No predictions were produced on the test set.")
        exit(1)

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    thresholds = np.full(len(y_cols), 0.5, dtype=float)
    task_aucs = []
    for col_idx in range(len(y_cols)):
        valid_mask = all_labels[:, col_idx] != -1
        if valid_mask.sum() == 0:
            thresholds[col_idx] = 0.5
            continue
        valid_labels = all_labels[valid_mask, col_idx]
        valid_preds = all_preds[valid_mask, col_idx]
        if len(np.unique(valid_labels)) < 2:
            thresholds[col_idx] = 0.5
            try:
                auc = roc_auc_score(valid_labels, valid_preds)
                task_aucs.append(auc)
            except Exception:
                pass
            continue
        fpr, tpr, thr = roc_curve(valid_labels, valid_preds)
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

    output_csv_path = "test_preds_labels.csv"
    with open(output_csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(y_cols)

        for i in range(all_preds.shape[0]):
            row_preds = all_preds[i].tolist()
            row_pred_results = []
            for j in range(len(y_cols)):
                if all_labels[i, j] == -1:
                    row_pred_results.append("")
                else:
                    val = 1 if all_preds[i, j] >= thresholds[j] else 0
                    row_pred_results.append(int(val))
            row_labels = all_labels[i].astype(int).tolist()
            row_labels = [lbl if lbl != -1 else "" for lbl in row_labels]
            csv_writer.writerow(row_preds)
            csv_writer.writerow(row_pred_results)
            csv_writer.writerow(row_labels)
            csv_writer.writerow([])