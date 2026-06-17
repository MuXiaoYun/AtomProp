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

# torch.autograd.set_detect_anomaly(True)

record_freq = cfg.record_freq
dataset_size = cfg.dataset_size
num_epochs = cfg.num_epochs
batch_size = cfg.batch_size

data_path = cfg.data_path
pretrain_file_type = cfg.pretrain_file_type

logdir = cfg.logdir
os.makedirs(f"trained_models/{logdir}", exist_ok=True)

fg_list = cfg.fg_list  # if none, use default rdkit fgs

chunk_size = cfg.chunk_size
max_atom_num = cfg.max_atom_num
weight_type = cfg.weight_type

less_rate = cfg.less_rate
more_rate = cfg.more_rate
embed_dim = cfg.embed_dim

if fg_list is None:
    fg_list = FunctionalGroups.BuildFuncGroupHierarchy()

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
               use_edge_embedding=cfg.use_edge_embedding,
               per_layer_FFN_type=cfg.per_layer_FFN_type,
               per_layer_FFN_hidden_dim=cfg.per_layer_FFN_hidden_dim,
               per_layer_FFN_num_layers=cfg.per_layer_FFN_num_layers,
               per_layer_FFN_dropout=cfg.per_layer_FFN_dropout,
               per_layer_FFN_num_experts=cfg.per_layer_FFN_num_experts,
               per_layer_FFN_top_k=cfg.per_layer_FFN_top_k,
               attention_rank=cfg.attention_rank)
head0 = MLP(input_dim=embed_dim, hidden_dim=128, output_dim=157, num_layers=2, dropout=cfg.head_dropout) # used for atom attribute prediction
head1 = MLP(input_dim=embed_dim, hidden_dim=128, output_dim=120, num_layers=2, dropout=cfg.head_dropout) # used for masked node prediction
head4 = MLP(input_dim=embed_dim, hidden_dim=128, output_dim=len(fg_list), num_layers=2, dropout=cfg.head_dropout) # used for functional group prediction
aggrmodel = GNNAggr(embed_dim=embed_dim, aggr='mean')

if cfg.from_scratch == False:
    # load weights
    ckpt = torch.load(cfg.from_model_path, weights_only=False)
    embedding_layer.load_state_dict(remove_module_prefix(ckpt['embedding_layer_state_dict']))
    backbone.load_state_dict(remove_module_prefix(ckpt['backbone_state_dict']))
    head0.load_state_dict(remove_module_prefix(ckpt['head0_state_dict']))
    head1.load_state_dict(remove_module_prefix(ckpt['head1_state_dict']))
    head4.load_state_dict(remove_module_prefix(ckpt['head4_state_dict']))

task0 = NodeAttrPrediction()
task1 = MaskedNodePrediction()
task2 = GraphMaskContrast(less_rate=less_rate, more_rate=more_rate)
task3 = BatchContrast()
task6 = FunctionalGroupsPrediction()
task7 = ScaffoldContrast()

tasks = [task0, task1, task2, task3, task6, task7]
task_types = ["regression", "classification", "regression", "classification", "classification", "classification"]

weight_strategy = None
if weight_type == "UW":
    if cfg.fix_uncertainty == True:
        weight_strategy = FixedUncertaintyWeighting(num_tasks=len(tasks))
    else:
        weight_strategy = AdaptiveUncertaintyWeighting(num_tasks=len(tasks), task_types=task_types)
elif weight_type == "GRADNORM":
    weight_strategy = GradNorm(num_tasks=len(tasks), init_weights=torch.exp(-cfg.fixed_log_vars))
    
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
    shuffle_mode = cfg.shuffle

    if shuffle_mode == "full":
        # Full random permutation — breaks chunk locality, random I/O
        indices = np.random.permutation(indices)
    elif shuffle_mode == "chunk":
        # Chunk-level shuffle — shuffle chunk order + shuffle within each chunk.
        # Maintains chunk-locality for efficient sequential disk reads.
        num_chunks = (total_size + chunk_size - 1) // chunk_size
        chunk_order = np.random.permutation(num_chunks)
        shuffled = []
        for c in chunk_order:
            start = c * chunk_size
            end = min(start + chunk_size, total_size)
            chunk = indices[start:end].copy()
            np.random.shuffle(chunk)
            shuffled.append(chunk)
        indices = np.concatenate(shuffled)
    # else "none" / False: keep sequential order

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

    # Each GeATLayer now contains: Q/K/V, edge_attention, project, norm1, norm2, ffn
    # All params in a layer get the same layer-wise LR decay
    for i, layer in enumerate(geat_conv.geat_layers):
        layer_lr_scale = layer_decay ** i
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

    # Neck layers (global attention + norm)
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

    # Final global FFN
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

    # Final FFN LayerNorm
    if hasattr(geat_model, 'FFN_norm'):
        params = []
        for name, param in geat_model.FFN_norm.named_parameters():
            if param.requires_grad:
                params.append(param)
        if params:
            param_groups.append({
                'params': params,
                'lr_scale': layer_decay ** ffn_layer_idx,
                'layer_idx': ffn_layer_idx,
                'name': 'ffn_norm'
            })

    # Edge type embedding
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

    # Edge direction embedding
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
    """Print batch progress information (only on rank 0)"""
    if rank is None or rank == 0:
        progress = (current_batch + 1) / total_batches * 100
        timer.print(f"[{stage}] Epoch {epoch+1}: Batch {current_batch+1}/{total_batches} ({progress:.1f}%) - Loss: {loss:.4f}")

def print_epoch_start(rank, epoch, total_epochs, timer: TrainingTimer, stage="Training"):
    """Print epoch start information (only on rank 0)"""
    if rank is None or rank == 0:
        print(f"\n{'='*70}")
        timer.print(f"{stage} - Epoch {epoch+1}/{total_epochs}")
        print(f"{'='*70}")

def print_epoch_end(rank, epoch, avg_loss, timer: TrainingTimer, stage="Training"):
    """Print epoch end information (only on rank 0)"""
    if rank is None or rank == 0:
        print(f"{'='*70}")
        timer.print(f"{stage} Epoch {epoch+1} Completed - Average Loss: {avg_loss:.6f}")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    # Initialize distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Only create tensorboard writer for rank 0
    if rank is None or rank == 0:
        writer = SummaryWriter(log_dir=f"runs/{logdir}")
    else:
        writer = None
        
    scaffold_calculator = ScaffoldSimilarityMatrix()
    
    # Get dataset info (only on rank 0)
    if rank is None or rank == 0:
        total_rows, columns = get_dataset_info(data_path)
        print(f"Total rows in dataset: {total_rows}")
    else:
        total_rows = 0
    
    # Broadcast dataset size to all processes
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
        scaffold_calculator = ScaffoldSimilarityMatrix()

        embedding_layer.to(device)
        backbone.to(device)
        head0.to(device)
        head1.to(device)
        head4.to(device)
        weight_strategy.to(device)
        backbone.print_params()
        
        if rank is not None and world_size > 1:
            embedding_layer = DDP(embedding_layer, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
            backbone = DDP(backbone, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
            head0 = DDP(head0, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
            head1 = DDP(head1, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
            head4 = DDP(head4, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
            weight_strategy = DDP(weight_strategy, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

        if rank is None or rank == 0:
            cfg.print_all_params()
            print(f"{embedding_layer.__class__.__name__} Parameters: {sum(p.numel() for p in embedding_layer.parameters() if p.requires_grad)}")
            print(f"{backbone.__class__.__name__} Parameters: {sum(p.numel() for p in backbone.parameters() if p.requires_grad)}")
            print(f"{head0.__class__.__name__} Parameters: {sum(p.numel() for p in head0.parameters() if p.requires_grad)}")
            print(f"{head1.__class__.__name__} Parameters: {sum(p.numel() for p in head1.parameters() if p.requires_grad)}")
            print(f"{head4.__class__.__name__} Parameters: {sum(p.numel() for p in head4.parameters() if p.requires_grad)}")
        
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
            "head0": head0,
            "head1": head1,
            "head4": head4,
            "weight_strategy": weight_strategy
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
            "head0": {
                "cls": torch.optim.Adam,
                "kwargs": {"lr": cfg.head_lr, "weight_decay": cfg.head_wd}
            },
            "head1": {
                "cls": torch.optim.Adam,
                "kwargs": {"lr": cfg.head_lr, "weight_decay": cfg.head_wd}
            },
            "head2": {
                "cls": torch.optim.Adam,
                "kwargs": {"lr": cfg.head_lr, "weight_decay": cfg.head_wd}
            },
            "head3": {
                "cls": torch.optim.Adam,
                "kwargs": {"lr": cfg.head_lr, "weight_decay": cfg.head_wd}
            },
            "head4": {
                "cls": torch.optim.Adam,
                "kwargs": {"lr": cfg.head_lr, "weight_decay": cfg.head_wd}
            },
            "weight_strategy": {
                "cls": torch.optim.Adam,
                "kwargs": {"lr": cfg.weight_strategy_lr, "weight_decay": cfg.weight_strategy_wd}
            }
        }

        # Get layer-wise parameter groups for GeATNet if layer decay is enabled
        if cfg.use_layer_decay and cfg.layer_decay_rate > 0:
            if rank is None or rank == 0:
                print(f"Using layer-wise learning rate decay with rate: {cfg.layer_decay_rate}")
            backbone_param_groups = get_geat_layer_parameters(backbone, layer_decay=cfg.layer_decay_rate)
            
            # Create parameter groups with scaled learning rates
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
            
            # Update backbone optimizer configuration
            optimizer_configs["backbone"]["kwargs"] = {
                "params": backbone_params,
                "lr": cfg.backbone_lr,  # Base LR, but individual params have scaled LRs
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
            "head0": {
                "cls": torch.optim.lr_scheduler.CosineAnnealingLR,
                "kwargs": {
                    "T_max": train_loader.total_batches * num_epochs,
                    "eta_min": cfg.head_eta_min
                }
            },
            "head1": {
                "cls": torch.optim.lr_scheduler.CosineAnnealingLR,
                "kwargs": {
                    "T_max": train_loader.total_batches * num_epochs,
                    "eta_min": cfg.head_eta_min
                }
            },
            "head2": {
                "cls": torch.optim.lr_scheduler.CosineAnnealingLR,
                "kwargs": {
                    "T_max": train_loader.total_batches * num_epochs,
                    "eta_min": cfg.head_eta_min
                }
            },
            "head3": {
                "cls": torch.optim.lr_scheduler.CosineAnnealingLR,
                "kwargs": {
                    "T_max": train_loader.total_batches * num_epochs,
                    "eta_min": cfg.head_eta_min
                }
            },
            "head4": {
                "cls": torch.optim.lr_scheduler.CosineAnnealingLR,
                "kwargs": {
                    "T_max": train_loader.total_batches * num_epochs,
                    "eta_min": cfg.head_eta_min
                }
            },
            "weight_strategy": {
                "cls": torch.optim.lr_scheduler.CosineAnnealingLR,
                "kwargs": {
                    "T_max": train_loader.total_batches * num_epochs,
                    "eta_min": cfg.weight_strategy_eta_min
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
            
            # Special handling for backbone when using layer decay
            if name == "backbone" and cfg.use_layer_decay and cfg.layer_decay_rate > 0:
                # backbone_params already contains parameter groups with individual LRs
                optimizers[name] = opt_cls(**opt_kwargs)
            else:
                # Standard initialization
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
        
        # Only create timer for rank 0
        if rank is None or rank == 0:
            timer = TrainingTimer()
        else:
            timer = None
        
        for epoch in range(num_epochs):
            # Training loop
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            
            try:
                # Print epoch start information
                print_epoch_start(rank, epoch, num_epochs, timer, "Training")
                
                embedding_layer.train()
                backbone.train()
                head0.train()
                head1.train()
                total_train_loss = 0.0
                train_sample_count = 0

                for batch_idx, (data_list, mols) in enumerate(train_loader):
                    # Skip first N batches if configured (e.g. resume after OOM crash)
                    if batch_idx < cfg.skip_batch:
                        if batch_idx == 0 and (rank is None or rank == 0):
                            print(f"[Skip] cfg.skip_batch={cfg.skip_batch}, skipping first {cfg.skip_batch} batches")
                        continue

                    # Stack-based OOM recovery: each item = (data_list, mols).
                    # On OOM (forward or backward), split & retry; DDP ranks sync via all_reduce.
                    # If a single molecule still OOMs, skip it (all ranks agree).
                    pending = [(data_list, mols)]
                    is_first_piece = True
                    oom_split_count = 0

                    while pending:
                        current_data, current_mols = pending.pop()
                        piece_skipped = False

                        # ---- Forward + Loss + Backward + Step, all inside OOM retry ----
                        while True:
                            oom_flag_local = False
                            try:
                                batch_data = Batch.from_data_list(current_data).to(device)

                                for opt in optimizers.values():
                                    opt.zero_grad()

                                atom_emb = embedding_layer(batch_data.x).squeeze()
                                embedded_data = Data(x=atom_emb, edge_index=batch_data.edge_index, edge_attr=batch_data.edge_attr)
                                graph_emb = backbone(embedded_data, batch=batch_data.batch)
                                graph_emb = graph_emb.view(-1, graph_emb.size(-1))

                                outputs = head0(graph_emb)
                                outputs = outputs.view(-1, outputs.size(-1))
                                task0.set_pred(outputs)
                                task0.run_label(current_mols, device)
                                loss_atom_attr_pred = task0.compute_loss()

                                mask_indices = MolGraphMask.select_mask_indices(batch_data.x)
                                masked_atom_emb = MolGraphMask.mask_atoms(atom_emb, mask_indices, torch.zeros(embed_dim))
                                masked_embedded_data = Data(x=masked_atom_emb, edge_index=batch_data.edge_index, edge_attr=batch_data.edge_attr)
                                graph_emb1 = backbone(masked_embedded_data, batch=batch_data.batch)
                                graph_emb1_masked = graph_emb1.view(-1, graph_emb1.size(-1))[mask_indices]
                                outputs1 = head1(graph_emb1_masked)
                                outputs1 = outputs1.view(-1, outputs1.size(-1))
                                task1.set_pred(outputs1)
                                task1_labels = batch_data.x[:,0][mask_indices]
                                task1.set_label(task1_labels)
                                loss_masked_atom_type_pred = task1.compute_loss()

                                less_mask_indices = MolGraphMask.select_mask_indices(batch_data.x, mask_ratio=less_rate)
                                less_masked_atom_emb = MolGraphMask.mask_atoms(atom_emb, less_mask_indices, torch.zeros(embed_dim))
                                less_masked_embedded_data = Data(x=less_masked_atom_emb, edge_index=batch_data.edge_index, edge_attr=batch_data.edge_attr)
                                less_graph_emb = backbone(less_masked_embedded_data, batch=batch_data.batch)
                                less_graph_emb = less_graph_emb.view(-1, less_graph_emb.size(-1))
                                less_outputs = aggrmodel(less_graph_emb, batch_data.batch)

                                more_mask_indices = MolGraphMask.select_mask_indices(batch_data.x, mask_ratio=more_rate)
                                more_masked_atom_emb = MolGraphMask.mask_atoms(atom_emb, more_mask_indices, torch.zeros(embed_dim))
                                more_masked_embedded_data = Data(x=more_masked_atom_emb, edge_index=batch_data.edge_index, edge_attr=batch_data.edge_attr)
                                more_graph_emb = backbone(more_masked_embedded_data, batch=batch_data.batch)
                                more_graph_emb = more_graph_emb.view(-1, more_graph_emb.size(-1))
                                more_outputs = aggrmodel(more_graph_emb, batch_data.batch)

                                anchor_outputs = aggrmodel(graph_emb, batch_data.batch)
                                task2.set_embeddings(anchor_outputs, less_outputs, more_outputs)
                                loss_triplet_contrast = task2.compute_loss()

                                outputs1_for_contrast = aggrmodel(graph_emb1, batch_data.batch)
                                task3.set_embeddings(anchor_outputs, outputs1_for_contrast)
                                loss_batch_contrast = task3.compute_loss()

                                outputs1_fg = head4(anchor_outputs)
                                fg_labels = FunctionalGroupUtils.batch_detect_with_rdkit_fg(current_mols, fg_list).to(device)
                                task6.set_pred(outputs1_fg)
                                task6.set_label(fg_labels)
                                loss_functional_group_pred = task6.compute_loss()

                                scaffold_groups = scaffold_calculator.get_scaffold_groups(mol_list=current_mols)
                                task7.set_embeddings(anchor_outputs)
                                task7.set_group_label(scaffold_groups)
                                loss_scaffold_contrast = task7.compute_loss()

                                # ---- Compute combined loss ----
                                losses = [loss_atom_attr_pred, loss_masked_atom_type_pred, loss_triplet_contrast, loss_batch_contrast, loss_functional_group_pred, loss_scaffold_contrast]
                                loss = weight_strategy(losses) if weight_type != "GRADNORM" else weight_strategy(losses, list(embedding_layer.module.parameters())+list(backbone.module.parameters()))

                                # ---- Pre-backward memory gate (all ranks sync here) ----
                                # If forward used >85% memory on any rank, all ranks split
                                # BEFORE backward starts (DDP gradient sync not yet active).
                                _free_mem, _total_mem = torch.cuda.mem_get_info(device)
                                _mem_pressure = (_free_mem < _total_mem * 0.15)
                                if rank is not None and world_size > 1:
                                    _p = torch.tensor([1 if _mem_pressure else 0], device=device)
                                    dist.all_reduce(_p, op=dist.ReduceOp.MAX)
                                    _mem_pressure = (_p.item() > 0)
                                if _mem_pressure:
                                    raise RuntimeError(
                                        f"out of memory: proactive split "
                                        f"(free={_free_mem/1024**3:.1f} GiB / total={_total_mem/1024**3:.1f} GiB)"
                                    )

                                # ---- Backward ----
                                if weight_type == "GRADNORM":
                                    loss, gn_loss, _ = loss
                                    loss.backward(retain_graph=True)
                                    gn_loss.backward()
                                else:
                                    loss.backward()

                                # ---- Optimizer & scheduler step ----
                                for opt in optimizers.values():
                                    opt.step()
                                for name, scheduler in schedulers.items():
                                    scheduler.step()

                                # All succeeded
                                break

                            except RuntimeError as e:
                                if "out of memory" in str(e):
                                    oom_flag_local = True
                                else:
                                    raise

                            # ---- DDP sync: any rank OOM → all ranks act ----
                            if rank is not None and world_size > 1:
                                oom_tensor = torch.tensor([1 if oom_flag_local else 0], device=device)
                                dist.all_reduce(oom_tensor, op=dist.ReduceOp.MAX)
                                oom_flag_global = (oom_tensor.item() > 0)
                            else:
                                oom_flag_global = oom_flag_local

                            if oom_flag_global:
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()

                                # Check if we can still split
                                if len(current_data) <= 1:
                                    # ---- Single molecule too large: skip it ----
                                    # All ranks must agree to skip (no backward → DDP stays in sync)
                                    for opt in optimizers.values():
                                        opt.zero_grad()
                                    if rank is None or rank == 0:
                                        print(f"[OOM Recovery] Batch {batch_idx+1}: single molecule too large, SKIPPING")
                                    piece_skipped = True
                                    loss = torch.tensor(0.0, device=device)
                                    loss_atom_attr_pred = torch.tensor(0.0, device=device)
                                    loss_masked_atom_type_pred = torch.tensor(0.0, device=device)
                                    loss_triplet_contrast = torch.tensor(0.0, device=device)
                                    loss_batch_contrast = torch.tensor(0.0, device=device)
                                    loss_functional_group_pred = torch.tensor(0.0, device=device)
                                    loss_scaffold_contrast = torch.tensor(0.0, device=device)
                                    break  # exit retry loop, piece_skipped=True

                                # ---- Split current batch in half ----
                                mid = len(current_data) // 2
                                oom_split_count += 1
                                if rank is None or rank == 0:
                                    print(f"[OOM Recovery] Batch {batch_idx+1}: {len(current_data)} → {mid}  "
                                          f"(split #{oom_split_count}, pending={len(pending)})")
                                pending.append((current_data[mid:], current_mols[mid:]))
                                current_data = current_data[:mid]
                                current_mols = current_mols[:mid]
                                # continue retry loop
                            else:
                                raise e

                        # ---- Per-piece done (success or skip) ----
                        if piece_skipped:
                            batch_size_current = 0  # no contribution to loss average
                        else:
                            batch_size_current = len(current_mols)
                            total_train_loss += loss.item() * batch_size_current
                            train_sample_count += batch_size_current

                        # Logging (only first piece of each original batch, and only if not skipped)
                        if is_first_piece and not piece_skipped:
                            if cfg.use_layer_decay and cfg.layer_decay_rate > 0 and (batch_idx == 0 or (batch_idx + 1) % record_freq == 0):
                                for param_group in optimizers["backbone"].param_groups:
                                    if 'name' in param_group:
                                        if rank is None or rank == 0:
                                            writer.add_scalar(f'LR/backbone_{param_group["name"]}', param_group['lr'], epoch * train_loader.total_batches + batch_idx)

                            if batch_idx == 0 or (batch_idx + 1) % record_freq == 0:
                                if rank is None or rank == 0:
                                    writer.add_scalar('Train/Loss_total', loss.item(), epoch * train_loader.total_batches + batch_idx)
                                    writer.add_scalar('Train/Loss_atom_attr', loss_atom_attr_pred.item(), epoch * train_loader.total_batches + batch_idx)
                                    writer.add_scalar('Train/Loss_masked_atom', loss_masked_atom_type_pred.item(), epoch * train_loader.total_batches + batch_idx)
                                    writer.add_scalar('Train/Loss_triplet', loss_triplet_contrast.item(), epoch * train_loader.total_batches + batch_idx)
                                    writer.add_scalar('Train/Loss_batch_contrast', loss_batch_contrast.item(), epoch * train_loader.total_batches + batch_idx)
                                    writer.add_scalar('Train/Loss_functional_group', loss_functional_group_pred.item(), epoch * train_loader.total_batches + batch_idx)
                                    writer.add_scalar('Train/Loss_scaffold_contrast', loss_scaffold_contrast.item(), epoch * train_loader.total_batches + batch_idx)
                                    try:
                                        for i in range(len(tasks)):
                                            if weight_type == "GRADNORM":
                                                item_name = "Uncertainty"
                                                item = weight_strategy.module.task_weights[i].item()
                                            elif weight_type == "UW":
                                                item_name = "TaskWeight"
                                                item = weight_strategy.module.log_vars[i].item()
                                            writer.add_scalar(f'Weight/{item_name}{i}', item, epoch * train_loader.total_batches + batch_idx)
                                    except Exception as e:
                                        print("TRAINING ERROR!")
                                        raise ValueError(e)

                        is_first_piece = False

                        if batch_idx == train_loader.total_batches - 1 and len(pending) == 0:
                            if rank is None or rank == 0:
                                metrics = {f"metrics_{i}": tasks[i].get_metrics() for i in range(len(tasks))}
                                print(f"Batch {batch_idx+1}/{train_loader.total_batches} Metrics: {metrics}")

                    # Print progress after all pieces of this batch are done
                    print_batch_progress(rank, epoch, batch_idx, train_loader.total_batches, loss.item(), timer, "Training")
                    if oom_split_count > 0 and (rank is None or rank == 0):
                        print(f"  [OOM Recovery] Batch {batch_idx+1} split into {oom_split_count + 1} sub-pieces")
                
                avg_train_loss = total_train_loss / train_sample_count
                train_losses.append(avg_train_loss)
                
                # Print epoch end information
                print_epoch_end(rank, epoch, avg_train_loss, timer, "Training")

                # Validation loop
                print_epoch_start(rank, epoch, num_epochs, timer, "Validation")
                
                embedding_layer.eval()
                backbone.eval()
                head0.eval()
                head1.eval()
                head4.eval()
                weight_strategy.eval()

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

                with torch.no_grad():
                    for batch_idx, (data_list, mols) in enumerate(val_loader):
                        batch_data = Batch.from_data_list(data_list).to(device)
                        
                        atom_emb = embedding_layer(batch_data.x).squeeze()
                        embedded_data = Data(x=atom_emb, edge_index=batch_data.edge_index, edge_attr=batch_data.edge_attr)
                        graph_emb = backbone(embedded_data, batch=batch_data.batch)
                        graph_emb = graph_emb.view(-1, graph_emb.size(-1))

                        outputs = head0(graph_emb)
                        outputs = outputs.view(-1, outputs.size(-1))
                        task0.set_pred(outputs)
                        task0.run_label(mols, device)
                        loss_atom_attr_pred = task0.compute_loss()
                        
                        mask_indices = MolGraphMask.select_mask_indices(batch_data.x)
                        masked_atom_emb = MolGraphMask.mask_atoms(atom_emb, mask_indices, torch.zeros(embed_dim))
                        masked_embedded_data = Data(x=masked_atom_emb, edge_index=batch_data.edge_index, edge_attr=batch_data.edge_attr)
                        graph_emb1 = backbone(masked_embedded_data, batch=batch_data.batch)
                        graph_emb1_masked = graph_emb1.view(-1, graph_emb1.size(-1))[mask_indices]
                        outputs1 = head1(graph_emb1_masked)
                        outputs1 = outputs1.view(-1, outputs1.size(-1))
                        task1.set_pred(outputs1)
                        task1_labels = batch_data.x[:,0][mask_indices]
                        task1.set_label(task1_labels)
                        loss_masked_atom_type_pred = task1.compute_loss()

                        less_mask_indices = MolGraphMask.select_mask_indices(batch_data.x, mask_ratio=less_rate)
                        less_masked_atom_emb = MolGraphMask.mask_atoms(atom_emb, less_mask_indices, torch.zeros(embed_dim))
                        less_masked_embedded_data = Data(x=less_masked_atom_emb, edge_index=batch_data.edge_index, edge_attr=batch_data.edge_attr)
                        less_graph_emb = backbone(less_masked_embedded_data, batch=batch_data.batch)
                        less_graph_emb = less_graph_emb.view(-1, less_graph_emb.size(-1))
                        less_outputs = aggrmodel(less_graph_emb, batch_data.batch)

                        more_mask_indices = MolGraphMask.select_mask_indices(batch_data.x, mask_ratio=more_rate)
                        more_masked_atom_emb = MolGraphMask.mask_atoms(atom_emb, more_mask_indices, torch.zeros(embed_dim))
                        more_masked_embedded_data = Data(x=more_masked_atom_emb, edge_index=batch_data.edge_index, edge_attr=batch_data.edge_attr)
                        more_graph_emb = backbone(more_masked_embedded_data, batch=batch_data.batch)
                        more_graph_emb = more_graph_emb.view(-1, more_graph_emb.size(-1))
                        more_outputs = aggrmodel(more_graph_emb, batch_data.batch)

                        anchor_outputs = aggrmodel(graph_emb, batch_data.batch)
                        task2.set_embeddings(anchor_outputs, less_outputs, more_outputs)
                        loss_triplet_contrast = task2.compute_loss()

                        outputs1_for_contrast = aggrmodel(graph_emb1, batch_data.batch)
                        task3.set_embeddings(anchor_outputs, outputs1_for_contrast)
                        loss_batch_contrast = task3.compute_loss()

                        outputs1_fg = head4(anchor_outputs)
                        fg_labels = FunctionalGroupUtils.batch_detect_with_rdkit_fg(mols, fg_list).to(device)
                        task6.set_pred(outputs1_fg)
                        task6.set_label(fg_labels)
                        loss_functional_group_pred = task6.compute_loss()
                        
                        scaffold_groups = scaffold_calculator.get_scaffold_groups(mol_list=mols)
                        task7.set_embeddings(anchor_outputs)
                        task7.set_group_label(scaffold_groups)
                        loss_scaffold_contrast = task7.compute_loss()
                        
                        losses = [loss_atom_attr_pred, loss_masked_atom_type_pred, loss_triplet_contrast, loss_batch_contrast, loss_functional_group_pred, loss_scaffold_contrast]

                        loss = weight_strategy(losses) if weight_type != "GRADNORM" else weight_strategy(losses, list(embedding_layer.module.parameters())+list(backbone.module.parameters()))[0]
                        
                        if batch_idx == val_loader.total_batches - 1:
                            metrics = {f"metrics_{i}": tasks[i].get_metrics() for i in range(len(tasks))}
                            if rank is None or rank == 0:
                                print(f"Batch {batch_idx+1}/{val_loader.total_batches} Metrics: {metrics}")
                        
                        batch_size_current = len(mols)
                        total_val_loss += loss.item() * batch_size_current
                        total_val_loss_atom_attr += loss_atom_attr_pred.item() * batch_size_current
                        total_val_loss_masked_atom += loss_masked_atom_type_pred.item() * batch_size_current
                        total_val_loss_triplet += loss_triplet_contrast.item() * batch_size_current
                        total_val_loss_batch_contrast += loss_batch_contrast.item() * batch_size_current
                        total_val_loss_functional_group += loss_functional_group_pred.item() * batch_size_current
                        total_val_loss_scaffold_contrast += loss_scaffold_contrast.item() * batch_size_current
                        val_sample_count += batch_size_current
                        
                        # Print batch progress (only on rank 0)
                        print_batch_progress(rank, epoch, batch_idx, val_loader.total_batches, loss.item(), timer, "Validation")
                
                avg_val_loss = total_val_loss / val_sample_count
                val_losses.append(avg_val_loss)
                
                # Print epoch end information for validation
                print_epoch_end(rank, epoch, avg_val_loss, timer, "Validation")
                
                if rank is None or rank == 0:
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
                
                if rank is None or rank == 0:
                    print(f"Epoch {epoch+1}/{num_epochs} Summary: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
                
                if rank is None or rank == 0 and avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save({
                        'embedding_layer_state_dict': embedding_layer.state_dict(),
                        'backbone_state_dict': backbone.state_dict(),
                        'optimizer_state_dict': {name: opt.state_dict() for name, opt in optimizers.items()},
                        'scheduler_state_dict': {name: sch.state_dict() for name, sch in schedulers.items()},
                        "head0_state_dict": head0.state_dict(),
                        "head1_state_dict": head1.state_dict(),
                        "head4_state_dict": head4.state_dict(),
                        'epoch': epoch,
                        'best_val_loss': best_val_loss
                    }, f'trained_models/{logdir}/best_model.pth')
                    print(f"Best model saved at epoch {epoch+1} with Val Loss = {best_val_loss:.6f}")
                
                # Save model at each epoch
                if rank is None or rank == 0:
                    torch.save({
                        'embedding_layer_state_dict': embedding_layer.state_dict(),
                        'backbone_state_dict': backbone.state_dict(),
                        'optimizer_state_dict': {name: opt.state_dict() for name, opt in optimizers.items()},
                        'scheduler_state_dict': {name: sch.state_dict() for name, sch in schedulers.items()},
                        "head0_state_dict": head0.state_dict(),
                        "head1_state_dict": head1.state_dict(),
                        "head4_state_dict": head4.state_dict(),
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
                        "head0_state_dict": head0.state_dict(),
                        "head1_state_dict": head1.state_dict(),
                        "head4_state_dict": head4.state_dict(),
                        'epoch': epoch,
                        'best_val_loss': best_val_loss
                    }, f'trained_models/{logdir}/interrupted_model.pth')
                    print("Training interrupted. Model state saved to 'interrupted_model.pth'.")
                break
            
        # Close tensorboard writer if exists
        if writer is not None:
            writer.close()
        
        cleanup_distributed()