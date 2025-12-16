from atomprop.tasks.tasks import NodeAttrPrediction, MaskedNodePrediction, GraphMaskContrast, BatchContrast, BondAnglePrediction, DihedralAnglePrediction, FunctionalGroupsPrediction, ScaffoldContrast
from atomprop.dataloader.dataloader import SMILESToInputs, PyGChunkDataListLoader, xyzBatchLoader, xyzBatchLoaderContext
from atomprop.models.GNNs import Embedder, GNN, GNNAggr
from atomprop.models.GeAT import GeATNet
from atomprop.utils.mlp import MLP
from atomprop.utils.mask import MolGraphMask
from atomprop.utils.groups import TripletGroup, QuadrupletGroup
from atomprop.utils.features import FunctionalGroupUtils
from atomprop.utils.weights import WeightStratergy, EqualWeightStratergy, HardSwitch, SoftSwitch, GradNorm, ParetoOpt, UncertaintyWeighting
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
from math import ceil
import os

record_freq = 100
dataset_size = 2000
num_epochs = 8
batch_size = 64

switch_timings = torch.tensor([0, 16000, 32000, 48000])
transition_width = 4000

# data_path = "data/zinc15/dataset/zinc_standard_agent/processed/smiles.csv"
data_path = "data/pubchem/pubchem-10m.txt"
pretrain_file_type = 'txt'

xyz_path = "data/pubchem/pubchem-xyzs.txt"
xyz_type = 'txt'

logdir = "pretrain_pubchem_geat"
os.makedirs(f"trained_models/{logdir}", exist_ok=True)

fg_list = None # if none, use default rdkit fgs

chunk_size = 65536
max_atom_num = 128

less_rate = 0.1
more_rate = 0.3
embed_dim = 384

device = torch.device("cuda:6") if torch.cuda.is_available() else torch.device("cpu")

if fg_list is None:
    fg_list = FunctionalGroups.BuildFuncGroupHierarchy()

backbone = Embedder(num_atom_types=120, embed_dim=embed_dim)
neck = GeATNet(embed_dim=embed_dim, dropout=0.5)
head0 = MLP(input_dim=embed_dim, hidden_dim=128, output_dim=157, num_layers=2, dropout=0.5) # used for atom attribute prediction
head1 = MLP(input_dim=embed_dim, hidden_dim=512, output_dim=embed_dim, num_layers=2, dropout=0.5) # used for masked node prediction
head2 = MLP(input_dim=embed_dim*3, hidden_dim=64, output_dim=1, num_layers=2, dropout=0.5) # used for bond angle prediction
head3 = MLP(input_dim=embed_dim*4, hidden_dim=64, output_dim=1, num_layers=2, dropout=0.5) # used for hydrogen bond prediction
head4 = MLP(input_dim=embed_dim, hidden_dim=128, output_dim=len(fg_list), num_layers=2, dropout=0.5) # used for functional group prediction
aggrmodel = GNNAggr(embed_dim=embed_dim, aggr='mean')

task0 = NodeAttrPrediction()
task1 = MaskedNodePrediction()
task2 = GraphMaskContrast(less_rate=less_rate, more_rate=more_rate)
task3 = BatchContrast()
task4 = BondAnglePrediction()
task5 = DihedralAnglePrediction()
task6 = FunctionalGroupsPrediction()
task7 = ScaffoldContrast()

tasks = [task0, task1, task2, task3, task4, task5, task6, task7]

weight_stratergy = UncertaintyWeighting(num_tasks=len(tasks))

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
    with xyzBatchLoaderContext(xyz_path) as xyz_loader:
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
        head0.to(device)
        head1.to(device)
        head2.to(device)
        head3.to(device)
        head4.to(device)
        weight_stratergy.to(device)

        print(backbone.__class__.__name__, f"Parameters: {sum(p.numel() for p in backbone.parameters() if p.requires_grad)}")
        print(neck.__class__.__name__, f"Parameters: {sum(p.numel() for p in neck.parameters() if p.requires_grad)}")
        print(head0.__class__.__name__, f"Parameters: {sum(p.numel() for p in head0.parameters() if p.requires_grad)}")
        print(head1.__class__.__name__, f"Parameters: {sum(p.numel() for p in head1.parameters() if p.requires_grad)}")
        print(head2.__class__.__name__, f"Parameters: {sum(p.numel() for p in head2.parameters() if p.requires_grad)}")
        print(head3.__class__.__name__, f"Parameters: {sum(p.numel() for p in head3.parameters() if p.requires_grad)}")
        print(head4.__class__.__name__, f"Parameters: {sum(p.numel() for p in head4.parameters() if p.requires_grad)}")
        
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
        
        components = {
            "backbone": backbone,
            "neck": neck,
            "head0": head0,
            "head1": head1,
            "head2": head2,
            "head3": head3,
            "head4": head4,
            "weight_stratergy": weight_stratergy
        }
        
        optimizer_configs = {
            "backbone": {
                "cls": torch.optim.Adam,
                "kwargs": {"lr": 5e-4, "weight_decay": 5e-5}
            },
            "neck": {
                "cls": torch.optim.AdamW,
                "kwargs": {"lr": 1e-3, "weight_decay": 1e-4}
            },
            "head0": {
                "cls": torch.optim.Adam,
                "kwargs": {"lr": 5e-4, "weight_decay": 1e-5}
            },
            "head1": {
                "cls": torch.optim.Adam,
                "kwargs": {"lr": 5e-4, "weight_decay": 1e-5}
            },
            "head2": {
                "cls": torch.optim.Adam,
                "kwargs": {"lr": 5e-4, "weight_decay": 1e-5}
            },
            "head3": {
                "cls": torch.optim.Adam,
                "kwargs": {"lr": 5e-4, "weight_decay": 1e-5}
            },
            "head4": {
                "cls": torch.optim.Adam,
                "kwargs": {"lr": 5e-4, "weight_decay": 1e-5}
            },
            "weight_stratergy": {
                "cls": torch.optim.Adam,
                "kwargs": {"lr": 1e-2, "weight_decay": 0}
            }
        }

        scheduler_configs = {
            "backbone": {
                "cls": torch.optim.lr_scheduler.OneCycleLR,
                "kwargs": {
                    "max_lr": 5e-4,
                    "total_steps": train_loader.total_batches * num_epochs, 
                    "pct_start": 0.1,
                    "anneal_strategy": "cos", 
                    "div_factor": 25.0, 
                    "final_div_factor": 1e4, 
                }
            },
            "neck": {
                "cls": torch.optim.lr_scheduler.OneCycleLR,
                "kwargs": {
                    "max_lr": 1e-3,
                    "total_steps": train_loader.total_batches * num_epochs,
                    "pct_start": 0.1,
                    "anneal_strategy": "cos",
                    "div_factor": 25.0,
                    "final_div_factor": 1e4,
                }
            },
            "head0": {
                "cls": torch.optim.lr_scheduler.OneCycleLR,
                "kwargs": {
                    "max_lr": 5e-4,
                    "total_steps": train_loader.total_batches * num_epochs,
                    "pct_start": 0.1,
                    "anneal_strategy": "cos",
                    "div_factor": 25.0,
                    "final_div_factor": 1e4,
                }
            },
            "head1": {
                "cls": torch.optim.lr_scheduler.OneCycleLR,
                "kwargs": {
                    "max_lr": 5e-4,
                    "total_steps": train_loader.total_batches * num_epochs,
                    "pct_start": 0.1,
                    "anneal_strategy": "cos",
                    "div_factor": 25.0,
                    "final_div_factor": 1e4,
                }
            },
            "head2": {
                "cls": torch.optim.lr_scheduler.OneCycleLR,
                "kwargs": {
                    "max_lr": 5e-4,
                    "total_steps": train_loader.total_batches * num_epochs,
                    "pct_start": 0.1,
                    "anneal_strategy": "cos",
                    "div_factor": 25.0,
                    "final_div_factor": 1e4,
                }
            },
            "head3": {
                "cls": torch.optim.lr_scheduler.OneCycleLR,
                "kwargs": {
                    "max_lr": 5e-4,
                    "total_steps": train_loader.total_batches * num_epochs,
                    "pct_start": 0.1,
                    "anneal_strategy": "cos",
                    "div_factor": 25.0,
                    "final_div_factor": 1e4,
                }
            },
            "head4": {
                "cls": torch.optim.lr_scheduler.OneCycleLR,
                "kwargs": {
                    "max_lr": 5e-4,
                    "total_steps": train_loader.total_batches * num_epochs,
                    "pct_start": 0.1,
                    "anneal_strategy": "cos",
                    "div_factor": 25.0,
                    "final_div_factor": 1e4,
                }
            },
            "weight_stratergy": {
                "cls": torch.optim.lr_scheduler.OneCycleLR,
                "kwargs": {
                    "max_lr": 1e-2,
                    "total_steps": train_loader.total_batches * num_epochs,
                    "pct_start": 0.05,
                    "anneal_strategy": "cos",
                    "div_factor": 10.0,
                    "final_div_factor": 1e3,
                }
            },
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
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        eps = 1e-8
        
        for epoch in range(num_epochs):
            try:
                xyz_loader.reset()

                backbone.train()
                neck.train()
                head0.train()
                head1.train()
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
                    graph_emb = neck(embedded_data, batch=batch_data.batch)
                    graph_emb = graph_emb.view(-1, graph_emb.size(-1))

                    outputs = head0(graph_emb)
                    outputs = outputs.view(-1, outputs.size(-1))
                    task0.set_pred(outputs)
                    task0.run_label(mols, device)
                    loss_atom_attr_pred = task0.compute_loss()
                    
                    mask_indices = MolGraphMask.select_mask_indices(batch_data.x)
                    masked_atom_emb = MolGraphMask.mask_atoms(atom_emb, mask_indices, torch.zeros(embed_dim))
                    masked_embedded_data = Data(x=masked_atom_emb, edge_index=batch_data.edge_index, edge_attr=batch_data.edge_attr)
                    graph_emb1 = neck(masked_embedded_data, batch=batch_data.batch)
                    graph_emb1_masked = graph_emb1.view(-1, graph_emb1.size(-1))[mask_indices]
                    outputs1 = head1(graph_emb1_masked)
                    outputs1 = outputs1.view(-1, outputs1.size(-1))
                    task1.set_pred(outputs1)
                    task1_labels = graph_emb[mask_indices]
                    task1.set_label(task1_labels)
                    loss_masked_atom_type_pred = task1.compute_loss()

                    less_mask_indices = MolGraphMask.select_mask_indices(batch_data.x, mask_ratio=less_rate)
                    less_masked_atom_emb = MolGraphMask.mask_atoms(atom_emb, less_mask_indices, torch.zeros(embed_dim))
                    less_masked_embedded_data = Data(x=less_masked_atom_emb, edge_index=batch_data.edge_index, edge_attr=batch_data.edge_attr)
                    less_graph_emb = neck(less_masked_embedded_data, batch=batch_data.batch)
                    less_graph_emb = less_graph_emb.view(-1, less_graph_emb.size(-1))
                    less_outputs = aggrmodel(less_graph_emb, batch_data.batch)

                    more_mask_indices = MolGraphMask.select_mask_indices(batch_data.x, mask_ratio=more_rate)
                    more_masked_atom_emb = MolGraphMask.mask_atoms(atom_emb, more_mask_indices, torch.zeros(embed_dim))
                    more_masked_embedded_data = Data(x=more_masked_atom_emb, edge_index=batch_data.edge_index, edge_attr=batch_data.edge_attr)
                    more_graph_emb = neck(more_masked_embedded_data, batch=batch_data.batch)
                    more_graph_emb = more_graph_emb.view(-1, more_graph_emb.size(-1))
                    more_outputs = aggrmodel(more_graph_emb, batch_data.batch)

                    anchor_outputs = aggrmodel(graph_emb, batch_data.batch)
                    task2.set_embeddings(anchor_outputs, less_outputs, more_outputs)
                    loss_triplet_contrast = task2.compute_loss()

                    outputs1_for_contrast = aggrmodel(graph_emb1, batch_data.batch)
                    task3.set_embeddings(anchor_outputs, outputs1_for_contrast)
                    loss_batch_contrast = task3.compute_loss()

                    xyzs = xyz_loader.get_batch(len(data_list)).to(device)
                    triplet_indices = TripletGroup.batch_generate(batch_data.edge_index).to(device)
                    triplet_emb = atom_emb[triplet_indices.view(-1)].view(-1, 3 * embed_dim).to(device) # (num_triplets, 3*embed_dim)
                    triplet_outputs = head2(triplet_emb) # (num_triplets, 1)
                    task4.set_label(xyzs, triplet_indices)
                    task4.set_pred(triplet_outputs)
                    loss_bond_angle_pred = task4.compute_loss()

                    quadruplet_indices = QuadrupletGroup.batch_generate(batch_data.edge_index).to(device)
                    task5.set_label(xyzs, quadruplet_indices)
                    quadruplet_emb = atom_emb[quadruplet_indices.view(-1)].view(-1, 4 * embed_dim).to(device) # (num_quadruplets, 4*embed_dim)
                    quadruplet_outputs = head3(quadruplet_emb) # (num_quadruplets, 1)
                    task5.set_pred(quadruplet_outputs)
                    loss_dihedral_angle_pred = task5.compute_loss()

                    outputs1_fg = head4(anchor_outputs)
                    fg_labels = FunctionalGroupUtils.batch_detect_with_rdkit_fg(mols, fg_list).to(device)
                    task6.set_pred(outputs1_fg)
                    task6.set_label(fg_labels)
                    loss_functional_group_pred = task6.compute_loss()
                    
                    scaffold_groups = scaffold_calculator.get_scaffold_groups(mol_list=mols)
                    task7.set_embeddings(anchor_outputs)
                    task7.set_group_label(scaffold_groups)
                    loss_scaffold_contrast = task7.compute_loss()

                    losses = [loss_atom_attr_pred, loss_masked_atom_type_pred, loss_triplet_contrast, loss_batch_contrast, loss_bond_angle_pred, loss_dihedral_angle_pred, loss_functional_group_pred, loss_scaffold_contrast]
                    loss = weight_stratergy(losses)

                    # backward and step
                    loss.backward()
                    for opt in optimizers.values():
                        opt.step()
                        
                    for name, scheduler in schedulers.items():
                        scheduler.step()
                    
                    if batch_idx == 0 or (batch_idx + 1) % record_freq == 0:
                        writer.add_scalar('Train/Loss_total', loss.item(), epoch * train_loader.total_batches + batch_idx)
                        writer.add_scalar('Train/Loss_atom_attr', loss_atom_attr_pred.item(), epoch * train_loader.total_batches + batch_idx)
                        writer.add_scalar('Train/Loss_masked_atom', loss_masked_atom_type_pred.item(), epoch * train_loader.total_batches + batch_idx)
                        writer.add_scalar('Train/Loss_triplet', loss_triplet_contrast.item(), epoch * train_loader.total_batches + batch_idx)
                        writer.add_scalar('Train/Loss_batch_contrast', loss_batch_contrast.item(), epoch * train_loader.total_batches + batch_idx)
                        writer.add_scalar('Train/Loss_bond_angle', loss_bond_angle_pred.item(), epoch * train_loader.total_batches + batch_idx)
                        writer.add_scalar('Train/Loss_dihedral_angle', loss_dihedral_angle_pred.item(), epoch * train_loader.total_batches + batch_idx)
                        writer.add_scalar('Train/Loss_functional_group', loss_functional_group_pred.item(), epoch * train_loader.total_batches + batch_idx)
                        writer.add_scalar('Train/Loss_scaffold_contrast', loss_scaffold_contrast.item(), epoch * train_loader.total_batches + batch_idx)
                        # log weights
                        try:
                            for i in range(len(tasks)):
                                writer.add_scalar(f'Weight/Uncertainty{i}', weight_stratergy.log_vars[i].item(), epoch * train_loader.total_batches + batch_idx)
                        except Exception:
                            # logging should not interrupt training
                            print("LOGGING ERROR: PLEASE CHECK")
                    
                    if batch_idx == train_loader.total_batches - 1:
                        metrics = {f"metrics_{i}": tasks[i].get_metrics() for i in range(8)}
                        print(f"Batch {batch_idx+1}/{train_loader.total_batches} Metrics: {metrics}")
                    
                    batch_size_current = len(mols)
                    total_train_loss += loss.item() * batch_size_current
                    train_sample_count += batch_size_current
                    
                    train_pbar.set_postfix({"Batch Loss": f"{loss.item():.6f}"})
                
                avg_train_loss = total_train_loss / train_sample_count
                train_losses.append(avg_train_loss)

                backbone.eval()
                neck.eval()
                head0.eval()
                head1.eval()
                head2.eval()
                head3.eval()
                head4.eval()
                weight_stratergy.eval()

                total_val_loss = 0.0
                total_val_loss_atom_attr = 0.0
                total_val_loss_masked_atom = 0.0
                total_val_loss_triplet = 0.0
                total_val_loss_batch_contrast = 0.0
                total_val_loss_bond_angle = 0.0
                total_val_loss_dihedral_angle = 0.0
                total_val_loss_functional_group = 0.0
                total_val_loss_scaffold_contrast = 0.0
                val_sample_count = 0
                
                val_pbar = tqdm(enumerate(val_loader),
                                    total=val_loader.total_batches, 
                                    desc=f"Epoch {epoch+1}/{num_epochs} - Validation")
                
                with torch.no_grad():
                    for batch_idx, (data_list, mols) in val_pbar:
                        batch_data = Batch.from_data_list(data_list).to(device)
                        
                        atom_emb = backbone(batch_data.x).squeeze()
                        embedded_data = Data(x=atom_emb, edge_index=batch_data.edge_index, edge_attr=batch_data.edge_attr)
                        graph_emb = neck(embedded_data, batch=batch_data.batch)
                        graph_emb = graph_emb.view(-1, graph_emb.size(-1))

                        outputs = head0(graph_emb)
                        outputs = outputs.view(-1, outputs.size(-1))
                        task0.set_pred(outputs)
                        task0.run_label(mols, device)
                        loss_atom_attr_pred = task0.compute_loss()
                        
                        mask_indices = MolGraphMask.select_mask_indices(batch_data.x)
                        masked_atom_emb = MolGraphMask.mask_atoms(atom_emb, mask_indices, torch.zeros(embed_dim))
                        masked_embedded_data = Data(x=masked_atom_emb, edge_index=batch_data.edge_index, edge_attr=batch_data.edge_attr)
                        graph_emb1 = neck(masked_embedded_data, batch=batch_data.batch)
                        graph_emb1_masked = graph_emb1.view(-1, graph_emb1.size(-1))[mask_indices]
                        outputs1 = head1(graph_emb1_masked)
                        outputs1 = outputs1.view(-1, outputs1.size(-1))
                        task1.set_pred(outputs1)
                        task1_labels = graph_emb[mask_indices]
                        task1.set_label(task1_labels)
                        loss_masked_atom_type_pred = task1.compute_loss()

                        less_mask_indices = MolGraphMask.select_mask_indices(batch_data.x, mask_ratio=less_rate)
                        less_masked_atom_emb = MolGraphMask.mask_atoms(atom_emb, less_mask_indices, torch.zeros(embed_dim))
                        less_masked_embedded_data = Data(x=less_masked_atom_emb, edge_index=batch_data.edge_index, edge_attr=batch_data.edge_attr)
                        less_graph_emb = neck(less_masked_embedded_data, batch=batch_data.batch)
                        less_graph_emb = less_graph_emb.view(-1, less_graph_emb.size(-1))
                        less_outputs = aggrmodel(less_graph_emb, batch_data.batch)

                        more_mask_indices = MolGraphMask.select_mask_indices(batch_data.x, mask_ratio=more_rate)
                        more_masked_atom_emb = MolGraphMask.mask_atoms(atom_emb, more_mask_indices, torch.zeros(embed_dim))
                        more_masked_embedded_data = Data(x=more_masked_atom_emb, edge_index=batch_data.edge_index, edge_attr=batch_data.edge_attr)
                        more_graph_emb = neck(more_masked_embedded_data, batch=batch_data.batch)
                        more_graph_emb = more_graph_emb.view(-1, more_graph_emb.size(-1))
                        more_outputs = aggrmodel(more_graph_emb, batch_data.batch)

                        anchor_outputs = aggrmodel(graph_emb, batch_data.batch)
                        task2.set_embeddings(anchor_outputs, less_outputs, more_outputs)
                        loss_triplet_contrast = task2.compute_loss()

                        outputs1_for_contrast = aggrmodel(graph_emb1, batch_data.batch)
                        task3.set_embeddings(anchor_outputs, outputs1_for_contrast)
                        loss_batch_contrast = task3.compute_loss()

                        xyzs = xyz_loader.get_batch(len(data_list)).to(device)
                        triplet_indices = TripletGroup.batch_generate(batch_data.edge_index).to(device)
                        triplet_emb = atom_emb[triplet_indices.view(-1)].view(-1, 3 * embed_dim).to(device) # (num_triplets, 3*embed_dim)
                        triplet_outputs = head2(triplet_emb) # (num_triplets, 1)
                        task4.set_label(xyzs, triplet_indices)
                        task4.set_pred(triplet_outputs)
                        loss_bond_angle_pred = task4.compute_loss()

                        quadruplet_indices = QuadrupletGroup.batch_generate(batch_data.edge_index).to(device)
                        task5.set_label(xyzs, quadruplet_indices)
                        quadruplet_emb = atom_emb[quadruplet_indices.view(-1)].view(-1, 4 * embed_dim).to(device) # (num_quadruplets, 4*embed_dim)
                        quadruplet_outputs = head3(quadruplet_emb) # (num_quadruplets, 1)
                        task5.set_pred(quadruplet_outputs)
                        loss_dihedral_angle_pred = task5.compute_loss()

                        outputs1_fg = head4(anchor_outputs)
                        fg_labels = FunctionalGroupUtils.batch_detect_with_rdkit_fg(mols, fg_list).to(device)
                        task6.set_pred(outputs1_fg)
                        task6.set_label(fg_labels)
                        loss_functional_group_pred = task6.compute_loss()
                        
                        scaffold_groups = scaffold_calculator.get_scaffold_groups(mol_list=mols)
                        task7.set_embeddings(anchor_outputs)
                        task7.set_group_label(scaffold_groups)
                        loss_scaffold_contrast = task7.compute_loss()
                        
                        losses = [loss_atom_attr_pred, loss_masked_atom_type_pred, loss_triplet_contrast, loss_batch_contrast, loss_bond_angle_pred, loss_dihedral_angle_pred, loss_functional_group_pred, loss_scaffold_contrast]

                        loss = weight_stratergy(losses)
                   
                        if batch_idx == val_loader.total_batches - 1:
                            metrics = {f"metrics_{i}": tasks[i].get_metrics() for i in range(8)}
                            print(f"Batch {batch_idx+1}/{val_loader.total_batches} Metrics: {metrics}")
                        
                        batch_size_current = len(mols)
                        total_val_loss += loss.item() * batch_size_current
                        total_val_loss_atom_attr += loss_atom_attr_pred.item() * batch_size_current
                        total_val_loss_masked_atom += loss_masked_atom_type_pred.item() * batch_size_current
                        total_val_loss_triplet += loss_triplet_contrast.item() * batch_size_current
                        total_val_loss_batch_contrast += loss_batch_contrast.item() * batch_size_current
                        total_val_loss_bond_angle += loss_bond_angle_pred.item() * batch_size_current
                        total_val_loss_dihedral_angle += loss_dihedral_angle_pred.item() * batch_size_current
                        total_val_loss_functional_group += loss_functional_group_pred.item() * batch_size_current
                        total_val_loss_scaffold_contrast += loss_scaffold_contrast.item() * batch_size_current
                        val_sample_count += batch_size_current
                        
                        val_pbar.set_postfix({"Batch Loss": f"{loss.item():.6f}"})
                
                avg_val_loss = total_val_loss / val_sample_count
                val_losses.append(avg_val_loss)
                
                writer.add_scalar('Epoch/Train_loss', avg_train_loss, epoch)
                writer.add_scalar('Epoch/Val_loss', avg_val_loss, epoch)
                writer.add_scalar('Epoch/Val_loss_atom_attr', total_val_loss_atom_attr / val_sample_count, epoch)
                writer.add_scalar('Epoch/Val_loss_masked_atom', total_val_loss_masked_atom / val_sample_count, epoch)
                writer.add_scalar('Epoch/Val_loss_triplet', total_val_loss_triplet / val_sample_count, epoch)
                writer.add_scalar('Epoch/Val_loss_batch_contrast', total_val_loss_batch_contrast / val_sample_count, epoch)
                writer.add_scalar('Epoch/Val_loss_bond_angle', total_val_loss_bond_angle / val_sample_count, epoch)
                writer.add_scalar('Epoch/Val_loss_dihedral_angle', total_val_loss_dihedral_angle / val_sample_count, epoch)
                writer.add_scalar('Epoch/Val_loss_functional_group', total_val_loss_functional_group / val_sample_count, epoch)
                writer.add_scalar('Epoch/Val_loss_scaffold_contrast', total_val_loss_scaffold_contrast / val_sample_count, epoch)
                
                print(f"Epoch {epoch+1}/{num_epochs} Summary: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save({
                        'backbone_state_dict': backbone.state_dict(),
                        'neck_state_dict': neck.state_dict(),
                        'head_state_dict': head0.state_dict(),
                        'head1_state_dict': head1.state_dict(),
                        'optimizer_state_dict': {name: opt.state_dict() for name, opt in optimizers.items()},
                        'scheduler_state_dict': {name: sch.state_dict() for name, sch in schedulers.items()},
                        'epoch': epoch,
                        'best_val_loss': best_val_loss
                    }, f'trained_models/{logdir}/best_model.pth')
                    print(f"Best model saved at epoch {epoch+1} with Val Loss = {best_val_loss:.6f}")
                # save model at each epoch
                torch.save({
                    'backbone_state_dict': backbone.state_dict(),
                    'neck_state_dict': neck.state_dict(),
                    'head_state_dict': head0.state_dict(),
                    'head1_state_dict': head1.state_dict(),
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
                    'head_state_dict': head0.state_dict(),
                    'head1_state_dict': head1.state_dict(),
                    'optimizer_state_dict': {name: opt.state_dict() for name, opt in optimizers.items()},
                    'scheduler_state_dict': {name: sch.state_dict() for name, sch in schedulers.items()},
                    'epoch': epoch,
                    'best_val_loss': best_val_loss
                }, f'trained_models/{logdir}/interrupted_model.pth')
                print("Training interrupted. Model state saved to 'interrupted_model.pth'.")
            
        writer.close()