from atomprop.tasks.tasks import NodeAttrPrediction, MaskedNodePrediction, GraphMaskContrast, BatchContrast
from atomprop.dataloader.dataloader import SMILESToInputs
from atomprop.models.GNNs import Embedder, GNN, GNNAggr
from atomprop.utils.mlp import MLP
from atomprop.utils.mask import MolGraphMask
from atomprop.embeddings.AtomEmbedding import BondTypes
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data, Batch
from torch.utils.tensorboard import SummaryWriter

embed_dim = 384

backbone = Embedder(num_atom_types=120, embed_dim=embed_dim)
neck = GNN(num_layers=6, embed_dim=embed_dim, gnn_type='gcn', JK='last', dropout=0.1)
head = MLP(input_dim=embed_dim, hidden_dim=256, output_dim=157, num_layers=1, dropout=0.1)
head1 = MLP(input_dim=embed_dim, hidden_dim=256, output_dim=embed_dim, num_layers=2, dropout=0.1)
aggrmodel = GNNAggr(embed_dim=embed_dim, aggr='mean')

less_rate = 0.1
more_rate = 0.3

record_freq = 100

task = NodeAttrPrediction()
task1 = MaskedNodePrediction()
task2 = GraphMaskContrast(less_rate=less_rate, more_rate=more_rate)
task3 = BatchContrast()

data_path = "data/nabladft/summary.csv"
dataset_size = -1
chunk_size = 65536
max_atom_num = 128
batch_size = 1024
num_epochs = 6

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

optimizer_configs = {
    "backbone": {
        "cls": torch.optim.Adam,
        "kwargs": {"lr": 5e-5, "weight_decay": 1e-5}
    },
    "neck": {
        "cls": torch.optim.AdamW,
        "kwargs": {"lr": 1e-4, "weight_decay": 5e-4}
    },
    "head": {
        "cls": torch.optim.Adam,
        "kwargs": {"lr": 2e-4, "weight_decay": 1e-6}
    },
    "head1": {
        "cls": torch.optim.Adam,
        "kwargs": {"lr": 2e-4, "weight_decay": 1e-6}
    }
}

scheduler_configs = {
    "backbone": {
        "cls": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "kwargs": {"mode": "min", "factor": 0.7, "patience": 4, "min_lr": 1e-6}
    },
    "neck": {
        "cls": torch.optim.lr_scheduler.CosineAnnealingLR,
        "kwargs": {"T_max": 20, "eta_min": 1e-6}
    },
    "head": {
        "cls": torch.optim.lr_scheduler.StepLR,
        "kwargs": {"step_size": 10, "gamma": 0.5}
    },
    "head1": {
        "cls": torch.optim.lr_scheduler.StepLR,
        "kwargs": {"step_size": 10, "gamma": 0.5}
    }
}

def get_dataset_info(data_path):
    total_rows = sum(1 for _ in open(data_path)) - 1
    sample_chunk = pd.read_csv(data_path, nrows=10)
    return total_rows, sample_chunk.columns.tolist()

def create_data_splits(total_size):
    indices = np.random.permutation(total_size)
    train_size = int(0.85 * total_size)
    val_size = int(0.10 * total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    return train_indices, val_indices, test_indices

def smiles_to_pyg_data(smiles, max_atom_num=None):
    atom_indices, edges, mol = SMILESToInputs.convert(
        smiles=smiles,
        context_length=max_atom_num
    )

    if mol is None:
        return None
    
    num_atoms = len(mol.GetAtoms())
    x = atom_indices[:num_atoms]
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    
    if edges.dim() == 2 and edges.size(1) == 3:
        edge_index = edges[:, :2].t().contiguous()
        edge_attr = edges[:, 2].unsqueeze(-1)
    else:
        edge_index = edges
        edge_attr = torch.ones(edges.size(1), 1) if edges.dim() == 2 else torch.ones(1, 1)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles, mol=mol)

class PyGChunkDataListLoader:
    def __init__(self, data_path, split_indices, chunk_size=65536, max_atom_num=128, batch_size=32, device=None):
        self.data_path = data_path
        self.split_indices = split_indices
        self.chunk_size = chunk_size
        self.max_atom_num = max_atom_num
        self.batch_size = batch_size
        self.current_chunk_idx = 0
        self.current_chunk_data = None
        self.current_chunk_start = 0
        self.headers = pd.read_csv(data_path, nrows=0).columns.tolist()
        self.device = device
        self.sorted_indices = np.sort(split_indices)
        self.total_batches = len(self.sorted_indices) // self.batch_size
        if len(self.sorted_indices) % self.batch_size != 0:
            self.total_batches += 1

    def __iter__(self):
        self.current_chunk_idx = 0
        self.current_chunk_data = None
        return self

    def __next__(self):
        data_list = []
        mols_list = []

        while len(data_list) < self.batch_size:
            if self.current_chunk_idx >= len(self.sorted_indices):
                if len(data_list) > 0:
                    return data_list, mols_list
                else:
                    raise StopIteration

            target_idx = self.sorted_indices[self.current_chunk_idx]
            chunk_num = target_idx // self.chunk_size
            chunk_start = chunk_num * self.chunk_size

            if self.current_chunk_data is None or chunk_start != self.current_chunk_start:
                self.current_chunk_data = pd.read_csv(
                    self.data_path,
                    skiprows=chunk_start + 1,
                    nrows=self.chunk_size,
                    header=None,
                    names=self.headers,
                    usecols=['SMILES']
                )
                self.current_chunk_start = chunk_start

            local_idx = target_idx % self.chunk_size
            smiles = self.current_chunk_data.iloc[local_idx]['SMILES']

            data = smiles_to_pyg_data(smiles, self.max_atom_num)

            if data is None:
                print(f"Invalid SMILES at index {target_idx}: {smiles}")
                self.current_chunk_idx += 1
                continue

            if self.device is not None:
                data = data.to(self.device)

            data_list.append(data)
            mols_list.append(data.mol)
            self.current_chunk_idx += 1

        return data_list, mols_list


if __name__ == "__main__":
    writer = SummaryWriter()
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
    head.to(device)
    head1.to(device)

    print(backbone.__class__.__name__, f"Parameters: {sum(p.numel() for p in backbone.parameters() if p.requires_grad)}")
    print(neck.__class__.__name__, f"Parameters: {sum(p.numel() for p in neck.parameters() if p.requires_grad)}")
    print(head.__class__.__name__, f"Parameters: {sum(p.numel() for p in head.parameters() if p.requires_grad)}")

    components = {
        "backbone": backbone,
        "neck": neck,
        "head": head,
        "head1": head1
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
        max_atom_num=max_atom_num,
        batch_size=batch_size,
    )
    val_loader = PyGChunkDataListLoader(
        data_path=data_path,
        split_indices=val_indices,
        chunk_size=chunk_size,
        max_atom_num=max_atom_num,
        batch_size=batch_size,
    )
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    eps = 1e-8
    
    for epoch in range(num_epochs):
        try:
            backbone.train()
            neck.train()
            head.train()
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
                embedded_data = Data(x=atom_emb, edge_index=batch_data.edge_index)
                graph_emb = neck(embedded_data)
                graph_emb = graph_emb.view(-1, graph_emb.size(-1))

                outputs = head(graph_emb)
                outputs = outputs.view(-1, outputs.size(-1))
                task.set_pred(outputs)
                task.run_label(mols, device)
                loss_atom_attr_pred = task.compute_loss()
                
                mask_indices = MolGraphMask.select_mask_indices(batch_data.x)
                masked_atom_emb = MolGraphMask.mask_atoms(atom_emb, mask_indices, torch.zeros(embed_dim))
                masked_embedded_data = Data(x=masked_atom_emb, edge_index=batch_data.edge_index)
                graph_emb1 = neck(masked_embedded_data)
                graph_emb1_masked = graph_emb1.view(-1, graph_emb1.size(-1))[mask_indices]
                outputs1 = head1(graph_emb1_masked)
                outputs1 = outputs1.view(-1, outputs1.size(-1))
                task1.set_pred(outputs1)
                task1_labels = graph_emb[mask_indices]
                task1.set_label(task1_labels)
                loss_masked_atom_type_pred = task1.compute_loss()

                less_mask_indices = MolGraphMask.select_mask_indices(batch_data.x, mask_ratio=less_rate)
                less_masked_atom_emb = MolGraphMask.mask_atoms(atom_emb, less_mask_indices, torch.zeros(embed_dim))
                less_masked_embedded_data = Data(x=less_masked_atom_emb, edge_index=batch_data.edge_index)
                less_graph_emb = neck(less_masked_embedded_data)
                less_graph_emb = less_graph_emb.view(-1, less_graph_emb.size(-1))
                less_outputs = aggrmodel(less_graph_emb, batch_data.batch)

                more_mask_indices = MolGraphMask.select_mask_indices(batch_data.x, mask_ratio=more_rate)
                more_masked_atom_emb = MolGraphMask.mask_atoms(atom_emb, more_mask_indices, torch.zeros(embed_dim))
                more_masked_embedded_data = Data(x=more_masked_atom_emb, edge_index=batch_data.edge_index)
                more_graph_emb = neck(more_masked_embedded_data)
                more_graph_emb = more_graph_emb.view(-1, more_graph_emb.size(-1))
                more_outputs = aggrmodel(more_graph_emb, batch_data.batch)

                anchor_outputs = aggrmodel(graph_emb, batch_data.batch)
                task2.set_embeddings(anchor_outputs, less_outputs, more_outputs)
                loss_triplet_contrast = task2.compute_loss()

                outputs1_for_contrast = aggrmodel(graph_emb1, batch_data.batch)
                task3.set_embeddings(anchor_outputs, outputs1_for_contrast)
                loss_batch_contrast = task3.compute_loss()

                # --- compute per-task gradient norms (proxy: gradients w.r.t atom embeddings) ---
                # Use atom_emb (output of backbone) as a shared representation to measure influence of each task.
                # Allow unused in case a particular loss does not depend on atom_emb for some rare batch.
                grads = []
                g = torch.autograd.grad(loss_atom_attr_pred, atom_emb, retain_graph=True, create_graph=False, allow_unused=True)[0]
                grads.append(g)
                g = torch.autograd.grad(loss_masked_atom_type_pred, atom_emb, retain_graph=True, create_graph=False, allow_unused=True)[0]
                grads.append(g)
                g = torch.autograd.grad(loss_triplet_contrast, atom_emb, retain_graph=True, create_graph=False, allow_unused=True)[0]
                grads.append(g)
                g = torch.autograd.grad(loss_batch_contrast, atom_emb, retain_graph=True, create_graph=False, allow_unused=True)[0]
                grads.append(g)

                # compute L2 norms (handle None grads)
                norms = []
                for gg in grads:
                    if gg is None:
                        norms.append(torch.tensor(0.0, device=device))
                    else:
                        norms.append(gg.norm())
                norms = torch.stack(norms)
                inv = 1.0 / (norms.detach() + eps)
                weights = inv / inv.sum()

                # final weighted loss
                loss = (weights[0] * loss_atom_attr_pred
                        + weights[1] * loss_masked_atom_type_pred
                        + weights[2] * loss_triplet_contrast
                        + weights[3] * loss_batch_contrast)

                # backward and step
                loss.backward()
                for opt in optimizers.values():
                    opt.step()
                
                if batch_idx == 0 or (batch_idx + 1) % record_freq == 0:
                    writer.add_scalar('Train/Loss_total', loss.item(), epoch * train_loader.total_batches + batch_idx)
                    writer.add_scalar('Train/Loss_atom_attr', loss_atom_attr_pred.item(), epoch * train_loader.total_batches + batch_idx)
                    writer.add_scalar('Train/Loss_masked_atom', loss_masked_atom_type_pred.item(), epoch * train_loader.total_batches + batch_idx)
                    writer.add_scalar('Train/Loss_triplet', loss_triplet_contrast.item(), epoch * train_loader.total_batches + batch_idx)
                    writer.add_scalar('Train/Loss_batch_contrast', loss_batch_contrast.item(), epoch * train_loader.total_batches + batch_idx)
                    # log grad norms and weights
                    try:
                        writer.add_scalar('Train/GradNorm_atom_attr', norms[0].item(), epoch * train_loader.total_batches + batch_idx)
                        writer.add_scalar('Train/GradNorm_masked_atom', norms[1].item(), epoch * train_loader.total_batches + batch_idx)
                        writer.add_scalar('Train/GradNorm_triplet', norms[2].item(), epoch * train_loader.total_batches + batch_idx)
                        writer.add_scalar('Train/GradNorm_batch_contrast', norms[3].item(), epoch * train_loader.total_batches + batch_idx)
                        writer.add_scalar('Train/Weight_atom_attr', weights[0].item(), epoch * train_loader.total_batches + batch_idx)
                        writer.add_scalar('Train/Weight_masked_atom', weights[1].item(), epoch * train_loader.total_batches + batch_idx)
                        writer.add_scalar('Train/Weight_triplet', weights[2].item(), epoch * train_loader.total_batches + batch_idx)
                        writer.add_scalar('Train/Weight_batch_contrast', weights[3].item(), epoch * train_loader.total_batches + batch_idx)
                    except Exception:
                        # logging should not interrupt training
                        pass
                
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
            head.eval()
            head1.eval()

            total_val_loss = 0.0
            total_val_loss_atom_attr = 0.0
            total_val_loss_masked_atom = 0.0
            total_val_loss_triplet = 0.0
            total_val_loss_batch_contrast = 0.0
            val_sample_count = 0
            
            val_pbar = tqdm(enumerate(val_loader),
                                total=val_loader.total_batches, 
                                desc=f"Epoch {epoch+1}/{num_epochs} - Validation")
            
            with torch.no_grad():
                for batch_idx, (data_list, mols) in val_pbar:
                    batch_data = Batch.from_data_list(data_list).to(device)
                    
                    atom_emb = backbone(batch_data.x).squeeze()
                    embedded_data = Data(x=atom_emb, edge_index=batch_data.edge_index)
                    graph_emb = neck(embedded_data)
                    graph_emb = graph_emb.view(-1, graph_emb.size(-1))

                    outputs = head(graph_emb)
                    outputs = outputs.view(-1, outputs.size(-1))
                    task.set_pred(outputs)
                    task.run_label(mols, device)
                    loss_atom_attr_pred = task.compute_loss()
                    
                    mask_indices = MolGraphMask.select_mask_indices(batch_data.x)
                    masked_atom_emb = MolGraphMask.mask_atoms(atom_emb, mask_indices, torch.zeros(embed_dim))
                    masked_embedded_data = Data(x=masked_atom_emb, edge_index=batch_data.edge_index)
                    graph_emb1 = neck(masked_embedded_data)
                    graph_emb1_masked = graph_emb1.view(-1, graph_emb1.size(-1))[mask_indices]
                    outputs1 = head1(graph_emb1_masked)
                    outputs1 = outputs1.view(-1, outputs1.size(-1))
                    task1.set_pred(outputs1)
                    task1_labels = graph_emb[mask_indices]
                    task1.set_label(task1_labels)
                    loss_masked_atom_type_pred = task1.compute_loss()

                    less_mask_indices = MolGraphMask.select_mask_indices(batch_data.x, mask_ratio=less_rate)
                    less_masked_atom_emb = MolGraphMask.mask_atoms(atom_emb, less_mask_indices, torch.zeros(embed_dim))
                    less_masked_embedded_data = Data(x=less_masked_atom_emb, edge_index=batch_data.edge_index)
                    less_graph_emb = neck(less_masked_embedded_data)
                    less_graph_emb = less_graph_emb.view(-1, less_graph_emb.size(-1))
                    less_outputs = aggrmodel(less_graph_emb, batch_data.batch)

                    more_mask_indices = MolGraphMask.select_mask_indices(batch_data.x, mask_ratio=more_rate)
                    more_masked_atom_emb = MolGraphMask.mask_atoms(atom_emb, more_mask_indices, torch.zeros(embed_dim))
                    more_masked_embedded_data = Data(x=more_masked_atom_emb, edge_index=batch_data.edge_index)
                    more_graph_emb = neck(more_masked_embedded_data)
                    more_graph_emb = more_graph_emb.view(-1, more_graph_emb.size(-1))
                    more_outputs = aggrmodel(more_graph_emb, batch_data.batch)

                    anchor_outputs = aggrmodel(graph_emb, batch_data.batch)
                    task2.set_embeddings(anchor_outputs, less_outputs, more_outputs)
                    loss_triplet_contrast = task2.compute_loss()

                    outputs1_for_contrast = aggrmodel(graph_emb1, batch_data.batch)
                    task3.set_embeddings(anchor_outputs, outputs1_for_contrast)
                    loss_batch_contrast = task3.compute_loss()

                    loss = (loss_atom_attr_pred
                            + loss_masked_atom_type_pred
                            + loss_triplet_contrast
                            + loss_batch_contrast)

                    if batch_idx == 0 or (batch_idx + 1) % record_freq == 0:
                        writer.add_scalar('Val/Loss_total', loss.item(), epoch * val_loader.total_batches + batch_idx)
                        writer.add_scalar('Val/Loss_atom_attr', loss_atom_attr_pred.item(), epoch * val_loader.total_batches + batch_idx)
                        writer.add_scalar('Val/Loss_masked_atom', loss_masked_atom_type_pred.item(), epoch * val_loader.total_batches + batch_idx)
                        writer.add_scalar('Val/Loss_triplet', loss_triplet_contrast.item(), epoch * val_loader.total_batches + batch_idx)
                        writer.add_scalar('Val/Loss_batch_contrast', loss_batch_contrast.item(), epoch * val_loader.total_batches + batch_idx)
                    
                    if batch_idx == val_loader.total_batches - 1:
                        metrics = task.get_metrics()
                        print(f"Batch {batch_idx+1}/{val_loader.total_batches} Metrics: {metrics}")
                    
                    batch_size_current = len(mols)
                    total_val_loss += loss.item() * batch_size_current
                    total_val_loss_atom_attr += loss_atom_attr_pred.item() * batch_size_current
                    total_val_loss_masked_atom += loss_masked_atom_type_pred.item() * batch_size_current
                    total_val_loss_triplet += loss_triplet_contrast.item() * batch_size_current
                    total_val_loss_batch_contrast += loss_batch_contrast.item() * batch_size_current
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
                    'head_state_dict': head.state_dict(),
                    'head1_state_dict': head1.state_dict(),
                    'optimizer_state_dict': {name: opt.state_dict() for name, opt in optimizers.items()},
                    'scheduler_state_dict': {name: sch.state_dict() for name, sch in schedulers.items()},
                    'epoch': epoch,
                    'best_val_loss': best_val_loss
                }, 'trained_models/best_model.pth')
                print(f"Best model saved at epoch {epoch+1} with Val Loss = {best_val_loss:.6f}")
            # save model at each epoch
            torch.save({
                'backbone_state_dict': backbone.state_dict(),
                'neck_state_dict': neck.state_dict(),
                'head_state_dict': head.state_dict(),
                'head1_state_dict': head1.state_dict(),
                'optimizer_state_dict': {name: opt.state_dict() for name, opt in optimizers.items()},
                'scheduler_state_dict': {name: sch.state_dict() for name, sch in schedulers.items()},
                'epoch': epoch,
                'val_loss': avg_val_loss
            }, f'trained_models/model_epoch{epoch}.pth')
            print(f"Model at epoch {epoch+1} saved with Val Loss = {avg_val_loss:.6f}")

        except KeyboardInterrupt:
            torch.save({
                'backbone_state_dict': backbone.state_dict(),
                'neck_state_dict': neck.state_dict(),
                'head_state_dict': head.state_dict(),
                'head1_state_dict': head1.state_dict(),
                'optimizer_state_dict': {name: opt.state_dict() for name, opt in optimizers.items()},
                'scheduler_state_dict': {name: sch.state_dict() for name, sch in schedulers.items()},
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }, 'trained_models/interrupted_model.pth')
            print("Training interrupted. Model state saved to 'interrupted_model.pth'.")
        
    writer.close()