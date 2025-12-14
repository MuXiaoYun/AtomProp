from atomprop.tasks.tasks import NodeAttrPrediction, MaskedNodePrediction, GraphMaskContrast, BatchContrast, BondAnglePrediction, DihedralAnglePrediction, FunctionalGroupsPrediction, ScaffoldContrast
from atomprop.dataloader.dataloader import SMILESToInputs, PyGChunkDataListLoader, xyzBatchLoader, xyzBatchLoaderContext
from atomprop.models.GNNs import Embedder, GNN, GNNAggr
from atomprop.utils.mlp import MLP
from atomprop.utils.mask import MolGraphMask
from atomprop.utils.groups import TripletGroup, QuadrupletGroup
from atomprop.utils.features import FunctionalGroupUtils
from atomprop.utils.weights import WeightStratergy, EqualWeightStratergy, HardSwitch, SoftSwitch, GradNorm, ParetoOpt
from atomprop.utils.scaffold import ScaffoldSimilarityMatrix
from atomprop.embeddings.AtomEmbedding import BondTypes
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data, Batch
from torch.utils.tensorboard import SummaryWriter
from rdkit.Chem import FunctionalGroups
import os

record_freq = 100
dataset_size = -1
num_epochs = 8
batch_size = 512

data_path = "data/pubchem/pubchem-10m.txt"
pretrain_file_type = 'txt'
logdir = "pretrain_pubchem_scaffold_part_test"
os.makedirs(f"trained_models/{logdir}", exist_ok=True)
chunk_size = 65536
max_atom_num = 128
embed_dim = 384
device = torch.device("cuda:6") if torch.cuda.is_available() else torch.device("cpu")

backbone = Embedder(num_atom_types=120, embed_dim=embed_dim)
neck = GNN(num_layers=7, embed_dim=embed_dim, gnn_type='gin', JK='last', dropout=0.5)
aggrmodel = GNNAggr(embed_dim=embed_dim, aggr='mean')
task = ScaffoldContrast()

optimizer_configs = {
    "backbone": {
        "cls": torch.optim.Adam,
        "kwargs": {"lr": 5e-4, "weight_decay": 5e-5}
    },
    "neck": {
        "cls": torch.optim.AdamW,
        "kwargs": {"lr": 1e-3, "weight_decay": 1e-4}
    }
}

scheduler_configs = {
    "backbone": {
        "cls": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "kwargs": {"mode": "min", "factor": 0.7, "patience": 4, "min_lr": 1e-5}
    },
    "neck": {
        "cls": torch.optim.lr_scheduler.CosineAnnealingLR,
        "kwargs": {"T_max": 20, "eta_min": 1e-5}
    }
}

def get_dataset_info(data_path):
    total_rows = sum(1 for _ in open(data_path)) - 1
    sample_chunk = pd.read_csv(data_path, nrows=10)
    return total_rows, sample_chunk.columns.tolist()

def create_data_splits(total_size):
    indices = np.arange(total_size)
    train_size = int(0.85 * total_size)
    val_size = int(0.10 * total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    return train_indices, val_indices, test_indices

if __name__ == "__main__":
    writer = SummaryWriter(log_dir=f"runs/{logdir}")
    scaffold_calculator = ScaffoldSimilarityMatrix()
    total_rows, columns = get_dataset_info(data_path)
    print(f"Total rows in dataset: {total_rows}")

    if dataset_size > 0:
        total_rows = min(total_rows, dataset_size)

    train_indices, val_indices, test_indices = create_data_splits(total_rows)
    print(f"Train set size: {len(train_indices)}, Val set size: {len(val_indices)}, Test set size: {len(test_indices)}")

    BondTypes.set_bond_types(["SINGLE", "DOUBLE", "TRIPLE", 'AROMATIC'])

    print(f"Using computing device: {device}")
    backbone.to(device)
    neck.to(device)

    print(backbone.__class__.__name__, f"Parameters: {sum(p.numel() for p in backbone.parameters() if p.requires_grad)}")
    print(neck.__class__.__name__, f"Parameters: {sum(p.numel() for p in neck.parameters() if p.requires_grad)}")

    components = {
        "backbone": backbone,
        "neck": neck
    }

    optimizers = {}
    schedulers = {}

    for name, module in components.items():
        opt_conf = optimizer_configs.get(name)
        if opt_conf is None:
            raise ValueError(f"Missing optimizer configuration for component '{name}'")

        opt_cls = opt_conf.get("cls")
        opt_kwargs = opt_conf.get("kwargs", {})
        optimizers[name] = opt_cls(module.parameters(), **opt_kwargs)

        sched_conf = scheduler_configs.get(name)
        if sched_conf is not None:
            sched_cls = sched_conf.get("cls")
            sched_kwargs = sched_conf.get("kwargs", {})
            schedulers[name] = sched_cls(optimizers[name], **sched_kwargs)

    train_loader = PyGChunkDataListLoader(
        data_path=data_path,
        split_indices=train_indices,
        chunk_size=chunk_size,
        batch_size=batch_size,
        file_type=pretrain_file_type
    )
    val_loader = PyGChunkDataListLoader(
        data_path=data_path,
        split_indices=val_indices,
        chunk_size=chunk_size,
        batch_size=batch_size,
        file_type=pretrain_file_type
    )

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    eps = 1e-8

    for epoch in range(num_epochs):
        try:
            backbone.train()
            neck.train()
            total_train_loss = 0.0
            train_sample_count = 0

            train_pbar = tqdm(enumerate(train_loader),
                            total=train_loader.total_batches,
                            desc=f"Epoch {epoch+1}/{num_epochs} - Training")

            for batch_idx, (data_list, mols) in train_pbar:
                batch_data = Batch.from_data_list(data_list).to(device)

                for opt in optimizers.values():
                    opt.zero_grad()

                atom_emb = backbone(batch_data.x).squeeze()
                embedded_data = Data(x=atom_emb, edge_index=batch_data.edge_index, edge_attr=batch_data.edge_attr)
                graph_emb = neck(embedded_data)
                graph_emb = graph_emb.view(-1, graph_emb.size(-1))
                anchor_outputs = aggrmodel(graph_emb, batch_data.batch)

                scaffold_groups = scaffold_calculator.get_scaffold_groups(mol_list=mols)
                task.set_embeddings(anchor_outputs)
                task.set_group_label(scaffold_groups)
                loss = task.compute_loss()

                loss.backward()
                for opt in optimizers.values():
                    opt.step()

                if batch_idx == 0 or (batch_idx + 1) % record_freq == 0:
                    writer.add_scalar('Train/Loss_total', loss.item(), epoch * train_loader.total_batches + batch_idx)

                if batch_idx == train_loader.total_batches - 1:
                    metrics = task.get_metrics()
                    print(f"Batch {batch_idx+1}/{train_loader.total_batches} Metrics: {metrics}")

                batch_size_current = len(mols)
                total_train_loss += loss.item() * batch_size_current
                train_sample_count += batch_size_current

                train_pbar.set_postfix({"Batch Loss": f"{loss.item():.6f}"})

            avg_train_loss = total_train_loss / train_sample_count
            train_losses.append(avg_train_loss)

            backbone.eval()
            neck.eval()

            total_val_loss = 0.0
            val_sample_count = 0

            val_pbar = tqdm(enumerate(val_loader),
                                total=val_loader.total_batches,
                                desc=f"Epoch {epoch+1}/{num_epochs} - Validation")

            with torch.no_grad():
                for batch_idx, (data_list, mols) in val_pbar:
                    batch_data = Batch.from_data_list(data_list).to(device)

                    atom_emb = backbone(batch_data.x).squeeze()
                    embedded_data = Data(x=atom_emb, edge_index=batch_data.edge_index, edge_attr=batch_data.edge_attr)
                    graph_emb = neck(embedded_data)
                    graph_emb = graph_emb.view(-1, graph_emb.size(-1))
                    anchor_outputs = aggrmodel(graph_emb, batch_data.batch)

                    scaffold_groups = scaffold_calculator.get_scaffold_groups(mol_list=mols)
                    task.set_embeddings(anchor_outputs)
                    task.set_group_label(scaffold_groups)
                    loss = task.compute_loss()

                    if batch_idx == val_loader.total_batches - 1:
                        metrics = task.get_metrics()
                        print(f"Batch {batch_idx+1}/{val_loader.total_batches} Metrics: {metrics}")

                    batch_size_current = len(mols)
                    total_val_loss += loss.item() * batch_size_current
                    val_sample_count += batch_size_current

                    val_pbar.set_postfix({"Batch Loss": f"{loss.item():.6f}"})

            avg_val_loss = total_val_loss / val_sample_count
            val_losses.append(avg_val_loss)

            writer.add_scalar('Epoch/Train_loss', avg_train_loss, epoch)
            writer.add_scalar('Epoch/Val_loss', avg_val_loss, epoch)

            for name, scheduler in schedulers.items():
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()

            print(f"Epoch {epoch+1}/{num_epochs} Summary: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'backbone_state_dict': backbone.state_dict(),
                    'neck_state_dict': neck.state_dict(),
                    'optimizer_state_dict': {name: opt.state_dict() for name, opt in optimizers.items()},
                    'scheduler_state_dict': {name: sch.state_dict() for name, sch in schedulers.items()},
                    'epoch': epoch,
                    'best_val_loss': best_val_loss
                }, f'trained_models/{logdir}/best_model.pth')
                print(f"Best model saved at epoch {epoch+1} with Val Loss = {best_val_loss:.6f}")

            torch.save({
                'backbone_state_dict': backbone.state_dict(),
                'neck_state_dict': neck.state_dict(),
                'optimizer_state_dict': {name: opt.state_dict() for name, opt in optimizers.items()},
                'scheduler_state_dict': {name: sch.state_dict() for name, sch in schedulers.items()},
                'epoch': epoch,
                'val_loss': avg_val_loss
            }, f'trained_models/{logdir}/model_epoch{epoch}.pth')
            print(f"Model at epoch {epoch+1} saved with Val Loss = {avg_val_loss:.6f}")

        except KeyboardInterrupt:
            torch.save({
                'backbone_state_dict': backbone.state_dict(),
                'neck_state_dict': neck.state_dict(),
                'optimizer_state_dict': {name: opt.state_dict() for name, opt in optimizers.items()},
                'scheduler_state_dict': {name: sch.state_dict() for name, sch in schedulers.items()},
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }, f'trained_models/{logdir}/interrupted_model.pth')
            print("Training interrupted. Model state saved to 'interrupted_model.pth'.")

    writer.close()