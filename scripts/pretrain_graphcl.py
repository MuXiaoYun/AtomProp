from atomprop.tasks.tasks import NodeAttrPrediction, MaskedNodePrediction, GraphMaskContrast, BatchContrast, FunctionalGroupsPrediction, ScaffoldContrast
from atomprop.dataloader.dataloader import SMILESToInputs, PyGChunkDataListLoader
from atomprop.models.gnns import Embedder, GNN, GNNAggr
from atomprop.models.geat import GeATNet
from atomprop.utils.mlp import MLP
from atomprop.utils.mask import MolGraphMask
from atomprop.utils.features import FunctionalGroupUtils
from atomprop.utils.weights import GradNorm, AdaptiveUncertaintyWeighting, FixedUncertaintyWeighting
from atomprop.utils.scaffold import ScaffoldSimilarityMatrix
from atomprop.embeddings.atom_embedding import BondTypes
from atomprop.utils.timer import TrainingTimer
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data, Batch
from torch.utils.tensorboard import SummaryWriter
from rdkit.Chem import FunctionalGroups
from contextlib import nullcontext
import os
import configs.config as cfg

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from atomprop.utils.utils import remove_module_prefix

record_freq = cfg.record_freq
dataset_size = cfg.dataset_size
num_epochs = cfg.num_epochs
batch_size = cfg.batch_size

data_path = cfg.data_path
pretrain_file_type = cfg.pretrain_file_type

logdir = cfg.logdir
os.makedirs(f"trained_models/{logdir}", exist_ok=True)

fg_list = cfg.fg_list

chunk_size = cfg.chunk_size
max_atom_num = cfg.max_atom_num
weight_type = cfg.weight_type

less_rate = cfg.less_rate
more_rate = cfg.more_rate
embed_dim = cfg.embed_dim

embedding_layer = Embedder(num_atom_types=120, embed_dim=embed_dim)
backbone = GeATNet(embed_dim=embed_dim,
               num_heads=cfg.num_heads,
               global_num_heads=cfg.global_num_heads,
               output_negative_slope=cfg.output_negative_slope,
               dropout=cfg.geat_dropout,
               geat_num_layers=cfg.geat_num_layers,
               aggr_num_layers=cfg.aggr_num_layers,
               FFN_hidden_dim=cfg.FFN_hidden_dim,
               FFN_num_experts=cfg.FFN_num_experts,
               FFN_num_layers=cfg.FFN_num_layers,
               FFN_top_k=cfg.FFN_top_k,
               FFN_type=cfg.FFN_type,
               use_edge_embedding=cfg.use_edge_embedding)
head1 = MLP(input_dim=embed_dim, hidden_dim=128, output_dim=120, num_layers=2, dropout=cfg.head_dropout)
aggrmodel = GNNAggr(embed_dim=embed_dim, aggr='mean')

if cfg.from_scratch == False:
    ckpt = torch.load(cfg.from_model_path, weights_only=False)
    embedding_layer.load_state_dict(remove_module_prefix(ckpt['embedding_layer_state_dict']))
    backbone.load_state_dict(remove_module_prefix(ckpt['backbone_state_dict']))
    head1.load_state_dict(remove_module_prefix(ckpt['head1_state_dict']))

task3 = BatchContrast()

tasks = [task3]
task_types = ["classification"]

weight_strategy = None

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        return None, 1, 0
    
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank
 
def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def get_dataset_info(data_path):
    total_rows = sum(1 for _ in open(data_path)) - 1
    sample_chunk = pd.read_csv(data_path, nrows=10)
    return total_rows, sample_chunk.columns.tolist()

def create_data_splits(total_size):
    indices = np.arange(total_size)
    if cfg.shuffle:
        indices = np.random.permutation(indices)
    train_size = int(0.85 * total_size)
    val_size = int(0.10 * total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    return train_indices, val_indices, test_indices

def get_geat_layer_parameters(model, layer_decay=0.9):
    if hasattr(model, 'module'):
        geat_model = model.module
    else:
        geat_model = model
    
    param_groups = []
    
    geat_conv = geat_model.backbone
    num_layers = len(geat_conv.geat_layers)
    
    for i, layer in enumerate(geat_conv.geat_layers):
        layer_lr_scale = layer_decay ** (num_layers - 1 - i)
        params = []
        for name, param in layer.named_parameters():
            if param.requires_grad:
                params.append(param)
        if params:
            param_groups.append({
                'params': params,
                'lr_scale': layer_lr_scale,
                'layer_idx': i,
                'name': f'geat_layer_{i}'
            })
    
    for i, norm_layer in enumerate(geat_conv.norm_layers):
        layer_lr_scale = layer_decay ** (num_layers - 1 - i)
        params = []
        for name, param in norm_layer.named_parameters():
            if param.requires_grad:
                params.append(param)
        if params:
            param_groups.append({
                'params': params,
                'lr_scale': layer_lr_scale,
                'layer_idx': i,
                'name': f'geat_norm_{i}'
            })
    
    neck_layers = geat_model.neck.global_attns
    neck_norm_layers = geat_model.neck.norm_layers
    
    for i in range(len(neck_layers)):
        layer_lr_scale = layer_decay ** (num_layers + i)
        params = []
        for name, param in neck_layers[i].named_parameters():
            if param.requires_grad:
                params.append(param)
        if params:
            param_groups.append({
                'params': params,
                'lr_scale': layer_lr_scale,
                'layer_idx': num_layers + i,
                'name': f'global_attn_{i}'
            })
        
        params = []
        for name, param in neck_norm_layers[i].named_parameters():
            if param.requires_grad:
                params.append(param)
        if params:
            param_groups.append({
                'params': params,
                'lr_scale': layer_lr_scale,
                'layer_idx': num_layers + i,
                'name': f'global_norm_{i}'
            })
    
    ffn_layer_idx = num_layers + len(neck_layers)
    params = []
    for name, param in geat_model.ffn.named_parameters():
        if param.requires_grad:
            params.append(param)
    if params:
        param_groups.append({
            'params': params,
            'lr_scale': layer_decay ** ffn_layer_idx,
            'layer_idx': ffn_layer_idx,
            'name': 'ffn'
        })
    
    params = []
    for name, param in geat_model.edge_type_embedding.named_parameters():
        if param.requires_grad:
            params.append(param)
    if params:
        param_groups.append({
            'params': params,
            'lr_scale': 1.0,
            'layer_idx': 0,
            'name': 'edge_type_embedding'
        })
    
    params = []
    for name, param in geat_model.edge_direction_embedding.named_parameters():
        if param.requires_grad:
            params.append(param)
    if params:
        param_groups.append({
            'params': params,
            'lr_scale': 1.0,
            'layer_idx': 0,
            'name': 'edge_direction_embedding'
        })
    
    return param_groups

def print_batch_progress(rank, epoch, current_batch, total_batches, loss, timer: TrainingTimer, stage="Training"):
    if rank is None or rank == 0:
        progress = (current_batch + 1) / total_batches * 100
        timer.print(f"[{stage}] Epoch {epoch+1}: Batch {current_batch+1}/{total_batches} ({progress:.1f}%) - Loss: {loss:.4f}")

def print_epoch_start(rank, epoch, total_epochs, timer: TrainingTimer, stage="Training"):
    if rank is None or rank == 0:
        print(f"\n{'='*70}")
        timer.print(f"{stage} - Epoch {epoch+1}/{total_epochs}")
        print(f"{'='*70}")

def print_epoch_end(rank, epoch, avg_loss, timer: TrainingTimer, stage="Training"):
    if rank is None or rank == 0:
        print(f"{'='*70}")
        timer.print(f"{stage} Epoch {epoch+1} Completed - Average Loss: {avg_loss:.6f}")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    rank, world_size, local_rank = setup_distributed()
    
    if rank is None or rank == 0:
        writer = SummaryWriter(log_dir=f"runs/{logdir}")
    else:
        writer = None
        
    if rank is None or rank == 0:
        total_rows, columns = get_dataset_info(data_path)
        print(f"Total rows in dataset: {total_rows}")
    else:
        total_rows = 0
    
    if rank is not None and world_size > 1:
        total_rows_tensor = torch.tensor([total_rows], device='cuda')
        dist.broadcast(total_rows_tensor, src=0)
        total_rows = int(total_rows_tensor.item())
    
    if dataset_size > 0:
        total_rows = min(total_rows, dataset_size)
    
    train_indices, val_indices, test_indices = create_data_splits(total_rows)
    
    if rank is None or rank == 0:
        print(f"Train set size: {len(train_indices)}, Val set size: {len(val_indices)}, Test set size: {len(test_indices)}")
        print(f"Using computing device: cuda:{local_rank if rank is not None else 0}")
    
    device = torch.device(f"cuda:{local_rank}" if rank is not None else "cuda" if torch.cuda.is_available() else "cpu")    

    with nullcontext():
        embedding_layer.to(device)
        backbone.to(device)
        head1.to(device)
        backbone.print_params()
        
        if rank is not None and world_size > 1:
            embedding_layer = DDP(embedding_layer, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
            backbone = DDP(backbone, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
            head1 = DDP(head1, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

        if rank is None or rank == 0:
            cfg.print_all_params()
            print(f"{embedding_layer.__class__.__name__} Parameters: {sum(p.numel() for p in embedding_layer.parameters() if p.requires_grad)}")
            print(f"{backbone.__class__.__name__} Parameters: {sum(p.numel() for p in backbone.parameters() if p.requires_grad)}")
            print(f"{head1.__class__.__name__} Parameters: {sum(p.numel() for p in head1.parameters() if p.requires_grad)}")
        
        train_sampler = DistributedSampler(train_indices, num_replicas=world_size, rank=rank, shuffle=False) if rank is not None else None
        val_sampler = DistributedSampler(val_indices, num_replicas=world_size, rank=rank, shuffle=False) if rank is not None else None
        
        train_loader = PyGChunkDataListLoader(
            data_path=data_path,
            split_indices=train_indices,
            chunk_size=chunk_size,
            batch_size=batch_size,
            file_type=pretrain_file_type,
            sampler=train_sampler
        )
        val_loader = PyGChunkDataListLoader(
            data_path=data_path,
            split_indices=val_indices,
            chunk_size=chunk_size,
            batch_size=batch_size,
            file_type=pretrain_file_type,
            sampler=val_sampler
        )
            
        components = {
            "embedding_layer": embedding_layer,
            "backbone": backbone,
            "head1": head1,
        }
        
        optimizer_configs = {
            "embedding_layer": {
                "cls": torch.optim.Adam,
                "kwargs": {"lr": cfg.embedding_layer_lr, "weight_decay": cfg.embedding_layer_wd}
            },
            "backbone": {
                "cls": torch.optim.AdamW,
                "kwargs": {"lr": cfg.backbone_lr, "weight_decay": cfg.backbone_wd}
            },
            "head1": {
                "cls": torch.optim.Adam,
                "kwargs": {"lr": cfg.head_lr, "weight_decay": cfg.head_wd}
            },
        }

        if cfg.use_layer_decay and cfg.layer_decay_rate > 0:
            if rank is None or rank == 0:
                print(f"Using layer-wise learning rate decay with rate: {cfg.layer_decay_rate}")
            backbone_param_groups = get_geat_layer_parameters(backbone, layer_decay=cfg.layer_decay_rate)
            backbone_params = []
            for group in backbone_param_groups:
                base_lr = cfg.backbone_lr
                scaled_lr = base_lr * group['lr_scale']
                backbone_params.append({
                    'params': group['params'],
                    'lr': scaled_lr,
                    'weight_decay': cfg.backbone_wd,
                    'name': group['name']
                })
                if rank is None or rank == 0:
                    print(f"  Layer {group['layer_idx']} ({group['name']}): LR scale = {group['lr_scale']:.4f}, Effective LR = {scaled_lr:.6f}")
            optimizer_configs["backbone"]["kwargs"] = {
                "params": backbone_params,
                "lr": cfg.backbone_lr,
                "weight_decay": cfg.backbone_wd
            }
        else:
            if rank is None or rank == 0:
                print("Using uniform learning rate for all GeAT layers")

        scheduler_configs = {
            "embedding_layer": {
                "cls": torch.optim.lr_scheduler.CosineAnnealingLR,
                "kwargs": {
                    "T_max": train_loader.total_batches * num_epochs,
                    "eta_min": cfg.embedding_layer_eta_min
                }
            },
            "backbone": {
                "cls": torch.optim.lr_scheduler.CosineAnnealingLR,
                "kwargs": {
                    "T_max": train_loader.total_batches * num_epochs,
                    "eta_min": cfg.backbone_eta_min
                }
            },
            "head1": {
                "cls": torch.optim.lr_scheduler.CosineAnnealingLR,
                "kwargs": {
                    "T_max": train_loader.total_batches * num_epochs,
                    "eta_min": cfg.head_eta_min
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
            
            if name == "backbone" and cfg.use_layer_decay and cfg.layer_decay_rate > 0:
                optimizers[name] = opt_cls(**opt_kwargs)
            else:
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
        
        if rank is None or rank == 0:
            timer = TrainingTimer()
        else:
            timer = None
        
        for epoch in range(num_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            
            try:
                print_epoch_start(rank, epoch, num_epochs, timer, "Training")
                
                embedding_layer.train()
                backbone.train()
                head1.train()
                total_train_loss = 0.0
                train_sample_count = 0

                for batch_idx, (data_list, mols) in enumerate(train_loader):
                    batch_data = Batch.from_data_list(data_list).to(device)
                    
                    for opt in optimizers.values():
                        opt.zero_grad()

                    atom_emb = embedding_layer(batch_data.x).squeeze()
                    embedded_data = Data(x=atom_emb, edge_index=batch_data.edge_index, edge_attr=batch_data.edge_attr)
                    graph_emb = backbone(embedded_data, batch=batch_data.batch)
                    graph_emb = graph_emb.view(-1, graph_emb.size(-1))

                    mask_indices = MolGraphMask.select_mask_indices(batch_data.x)
                    masked_atom_emb = MolGraphMask.mask_atoms(atom_emb, mask_indices, torch.zeros(embed_dim))
                    masked_embedded_data = Data(x=masked_atom_emb, edge_index=batch_data.edge_index, edge_attr=batch_data.edge_attr)
                    graph_emb1 = backbone(masked_embedded_data, batch=batch_data.batch)
                    graph_emb1_masked = graph_emb1.view(-1, graph_emb1.size(-1))[mask_indices]
                    outputs1 = head1(graph_emb1_masked)
                    outputs1 = outputs1.view(-1, outputs1.size(-1))
                    task1_labels = batch_data.x[:,0][mask_indices]
                    
                    anchor_outputs = aggrmodel(graph_emb, batch_data.batch)
                    outputs1_for_contrast = aggrmodel(graph_emb1, batch_data.batch)
                    task3.set_embeddings(anchor_outputs, outputs1_for_contrast)
                    loss_batch_contrast = task3.compute_loss()

                    loss = loss_batch_contrast
                    
                    loss.backward()
                              
                    for opt in optimizers.values():
                        opt.step()
                        
                    for name, scheduler in schedulers.items():
                        scheduler.step()
                    
                    if cfg.use_layer_decay and cfg.layer_decay_rate > 0 and (batch_idx == 0 or (batch_idx + 1) % record_freq == 0):
                        for param_group in optimizers["backbone"].param_groups:
                            if 'name' in param_group:
                                if rank is None or rank == 0:
                                    writer.add_scalar(f'LR/backbone_{param_group["name"]}', param_group['lr'], epoch * train_loader.total_batches + batch_idx)
                    
                    if batch_idx == 0 or (batch_idx + 1) % record_freq == 0:
                        if rank is None or rank == 0:
                            writer.add_scalar('Train/Loss_total', loss.item(), epoch * train_loader.total_batches + batch_idx)
                            writer.add_scalar('Train/Loss_batch_contrast', loss_batch_contrast.item(), epoch * train_loader.total_batches + batch_idx)
                    
                    batch_size_current = len(mols)
                    total_train_loss += loss.item() * batch_size_current
                    train_sample_count += batch_size_current
                    
                    print_batch_progress(rank, epoch, batch_idx, train_loader.total_batches, loss.item(), timer, "Training")
                
                avg_train_loss = total_train_loss / train_sample_count
                train_losses.append(avg_train_loss)
                
                print_epoch_end(rank, epoch, avg_train_loss, timer, "Training")

                print_epoch_start(rank, epoch, num_epochs, timer, "Validation")
                
                embedding_layer.eval()
                backbone.eval()
                head1.eval()

                total_val_loss = 0.0
                total_val_loss_batch_contrast = 0.0
                val_sample_count = 0

                with torch.no_grad():
                    for batch_idx, (data_list, mols) in enumerate(val_loader):
                        batch_data = Batch.from_data_list(data_list).to(device)
                        
                        atom_emb = embedding_layer(batch_data.x).squeeze()
                        embedded_data = Data(x=atom_emb, edge_index=batch_data.edge_index, edge_attr=batch_data.edge_attr)
                        graph_emb = backbone(embedded_data, batch=batch_data.batch)
                        graph_emb = graph_emb.view(-1, graph_emb.size(-1))

                        mask_indices = MolGraphMask.select_mask_indices(batch_data.x)
                        masked_atom_emb = MolGraphMask.mask_atoms(atom_emb, mask_indices, torch.zeros(embed_dim))
                        masked_embedded_data = Data(x=masked_atom_emb, edge_index=batch_data.edge_index, edge_attr=batch_data.edge_attr)
                        graph_emb1 = backbone(masked_embedded_data, batch=batch_data.batch)
                        graph_emb1_masked = graph_emb1.view(-1, graph_emb1.size(-1))[mask_indices]
                        outputs1 = head1(graph_emb1_masked)
                        outputs1 = outputs1.view(-1, outputs1.size(-1))
                        task1_labels = batch_data.x[:,0][mask_indices]
                        
                        anchor_outputs = aggrmodel(graph_emb, batch_data.batch)
                        outputs1_for_contrast = aggrmodel(graph_emb1, batch_data.batch)
                        task3.set_embeddings(anchor_outputs, outputs1_for_contrast)
                        loss_batch_contrast = task3.compute_loss()
                        
                        loss = loss_batch_contrast
                        
                        batch_size_current = len(mols)
                        total_val_loss += loss.item() * batch_size_current
                        total_val_loss_batch_contrast += loss_batch_contrast.item() * batch_size_current
                        val_sample_count += batch_size_current
                        
                        print_batch_progress(rank, epoch, batch_idx, val_loader.total_batches, loss.item(), timer, "Validation")
                
                avg_val_loss = total_val_loss / val_sample_count
                val_losses.append(avg_val_loss)
                
                print_epoch_end(rank, epoch, avg_val_loss, timer, "Validation")
                
                if rank is None or rank == 0:
                    writer.add_scalar('Epoch/Train_loss', avg_train_loss, epoch)
                    writer.add_scalar('Epoch/Val_loss', avg_val_loss, epoch)
                    writer.add_scalar('Epoch/Val_loss_batch_contrast', total_val_loss_batch_contrast / val_sample_count, epoch)
                
                if rank is None or rank == 0:
                    print(f"Epoch {epoch+1}/{num_epochs} Summary: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
                
                if rank is None or rank == 0 and avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save({
                        'embedding_layer_state_dict': embedding_layer.state_dict(),
                        'backbone_state_dict': backbone.state_dict(),
                        'optimizer_state_dict': {name: opt.state_dict() for name, opt in optimizers.items()},
                        'scheduler_state_dict': {name: sch.state_dict() for name, sch in schedulers.items()},
                        "head1_state_dict": head1.state_dict(),
                        'epoch': epoch,
                        'best_val_loss': best_val_loss
                    }, f'trained_models/{logdir}/best_model.pth')
                    print(f"Best model saved at epoch {epoch+1} with Val Loss = {best_val_loss:.6f}")
                
                if rank is None or rank == 0:
                    torch.save({
                        'embedding_layer_state_dict': embedding_layer.state_dict(),
                        'backbone_state_dict': backbone.state_dict(),
                        'optimizer_state_dict': {name: opt.state_dict() for name, opt in optimizers.items()},
                        'scheduler_state_dict': {name: sch.state_dict() for name, sch in schedulers.items()},
                        "head1_state_dict": head1.state_dict(),
                        'epoch': epoch,
                        'val_loss': avg_val_loss
                    }, f'trained_models/{logdir}/model_epoch{epoch}.pth')
                    print(f"Model at epoch {epoch+1} saved with Val Loss = {avg_val_loss:.6f}")

            except KeyboardInterrupt:
                if rank is None or rank == 0:
                    torch.save({
                        'embedding_layer_state_dict': embedding_layer.state_dict(),
                        'backbone_state_dict': backbone.state_dict(),
                        'optimizer_state_dict': {name: opt.state_dict() for name, opt in optimizers.items()},
                        'scheduler_state_dict': {name: sch.state_dict() for name, sch in schedulers.items()},
                        "head1_state_dict": head1.state_dict(),
                        'epoch': epoch,
                        'best_val_loss': best_val_loss
                    }, f'trained_models/{logdir}/interrupted_model.pth')
                    print("Training interrupted. Model state saved to 'interrupted_model.pth'.")
                break
            
        if writer is not None:
            writer.close()
        
        cleanup_distributed()