from atomprop.tasks.tasks import NodeAttrPrediction, MaskedNodePrediction, GraphMaskContrast, BatchContrast, BondAnglePrediction, DihedralAnglePrediction, FunctionalGroupsPrediction, ScaffoldContrast
from atomprop.dataloader.dataloader import SMILESToInputs, PyGChunkDataListLoader, xyzBatchLoader, xyzBatchLoaderContext
from atomprop.models.GNNs import Embedder, GNN, GNNAggr
from atomprop.models.GeAT import GeATNet
from atomprop.utils.mlp import MLP
from atomprop.utils.mask import MolGraphMask
from atomprop.utils.groups import TripletGroup, QuadrupletGroup
from atomprop.utils.features import FunctionalGroupUtils
from atomprop.utils.weights import WeightStratergy, EqualWeightStratergy, HardSwitch, SoftSwitch, GradNorm, ParetoOpt, UncertaintyWeighting, FixedUncertaintyWeighting
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
from contextlib import nullcontext
import os
import configs.config as cfg

cfg.print_all_params()

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

less_rate = cfg.less_rate
more_rate = cfg.more_rate
embed_dim = cfg.embed_dim

device = torch.device(cfg.device_str) if torch.cuda.is_available() else torch.device("cpu")

if fg_list is None:
    fg_list = FunctionalGroups.BuildFuncGroupHierarchy()

backbone = Embedder(num_atom_types=120, embed_dim=embed_dim)
neck = GeATNet(embed_dim=embed_dim,
               num_heads=cfg.num_heads,
               global_num_heads=cfg.global_num_heads,
               output_negative_slope=cfg.output_negative_slope,
               dropout=cfg.geat_dropout,
               geat_num_layers=cfg.geat_num_layers,
               aggr_num_layers=cfg.aggr_num_layers,
               FFN_hidden_dim=cfg.FFN_hidden_dim,
               FFN_num_experts=cfg.FFN_num_experts,
               FFN_num_layers=cfg.FFN_num_layers,
               FFN_top_k=cfg.FFN_top_k)
head0 = MLP(input_dim=embed_dim, hidden_dim=128, output_dim=157, num_layers=2, dropout=0.5) # used for atom attribute prediction
head1 = MLP(input_dim=embed_dim, hidden_dim=128, output_dim=120, num_layers=2, dropout=0.5) # used for masked node prediction
head4 = MLP(input_dim=embed_dim, hidden_dim=128, output_dim=len(fg_list), num_layers=2, dropout=0.5) # used for functional group prediction
aggrmodel = GNNAggr(embed_dim=embed_dim, aggr='mean')

if cfg.from_scratch == False:
    # load weights
    ckpt = torch.load(cfg.from_model_path, weights_only=False)
    backbone.load_state_dict(ckpt['backbone_state_dict'])
    neck.load_state_dict(ckpt['neck_state_dict'])

task0 = NodeAttrPrediction()
task1 = MaskedNodePrediction()
task2 = GraphMaskContrast(less_rate=less_rate, more_rate=more_rate)
task3 = BatchContrast()
task6 = FunctionalGroupsPrediction()
task7 = ScaffoldContrast()

tasks = [task0, task1, task2, task3, task6, task7]
task_types = ["classification", "regression", "other", "other", "classification", "other"]

weight_stratergy = None
if cfg.fix_uncertainty == True:
    weight_stratergy = FixedUncertaintyWeighting(num_tasks=len(tasks))
else:
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

def get_geat_layer_parameters(model, layer_decay=0.9):
    """
    Extract parameters from GeATNet with layer-wise decay.
    
    Args:
        model: GeATNet instance
        layer_decay: decay factor for each layer (e.g., 0.9 means each layer has 90% LR of previous layer)
    
    Returns:
        List of parameter groups with different learning rates
    """
    param_groups = []
    
    # Extract GeAT layers from the backbone of GeATNet
    geat_conv = model.backbone
    num_layers = len(geat_conv.geat_layers)
    
    # Layer-wise parameters for GeAT layers
    for i, layer in enumerate(geat_conv.geat_layers):
        layer_lr_scale = layer_decay ** (num_layers - 1 - i)  # Deeper layers get smaller LR
        params = []
        
        # Get all parameters from this layer
        for name, param in layer.named_parameters():
            if param.requires_grad:
                params.append(param)
        
        if params:  # Only add if there are trainable parameters
            param_groups.append({
                'params': params,
                'lr_scale': layer_lr_scale,
                'layer_idx': i,
                'name': f'geat_layer_{i}'
            })
    
    # Parameters from norm layers in GeATConv
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
    
    # Parameters from neck (GlobalAttnConv) - treat as additional layers
    neck_layers = model.neck.global_attns
    neck_norm_layers = model.neck.norm_layers
    
    for i in range(len(neck_layers)):
        # Global attention layers
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
        
        # Norm layers for global attention
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
    
    # Parameters from FFN (MoE) - treat as the final layers
    ffn_layer_idx = num_layers + len(neck_layers)
    params = []
    for name, param in model.ffn.named_parameters():
        if param.requires_grad:
            params.append(param)
    
    if params:
        param_groups.append({
            'params': params,
            'lr_scale': layer_decay ** ffn_layer_idx,
            'layer_idx': ffn_layer_idx,
            'name': 'ffn'
        })
    
    # Edge embeddings - treat as input layer (no decay)
    params = []
    for name, param in model.edge_type_embedding.named_parameters():
        if param.requires_grad:
            params.append(param)
    
    if params:
        param_groups.append({
            'params': params,
            'lr_scale': 1.0,  # No decay for embeddings
            'layer_idx': 0,
            'name': 'edge_type_embedding'
        })
    
    params = []
    for name, param in model.edge_direction_embedding.named_parameters():
        if param.requires_grad:
            params.append(param)
    
    if params:
        param_groups.append({
            'params': params,
            'lr_scale': 1.0,  # No decay for embeddings
            'layer_idx': 0,
            'name': 'edge_direction_embedding'
        })
    
    return param_groups

if __name__ == "__main__":
    with nullcontext():
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
        head4.to(device)
        weight_stratergy.to(device)
        neck.print_params()

        print(backbone.__class__.__name__, f"Parameters: {sum(p.numel() for p in backbone.parameters() if p.requires_grad)}")
        print(neck.__class__.__name__, f"Parameters: {sum(p.numel() for p in neck.parameters() if p.requires_grad)}")
        print(head0.__class__.__name__, f"Parameters: {sum(p.numel() for p in head0.parameters() if p.requires_grad)}")
        print(head1.__class__.__name__, f"Parameters: {sum(p.numel() for p in head1.parameters() if p.requires_grad)}")
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
            "head4": head4,
            "weight_stratergy": weight_stratergy
        }
        
        optimizer_configs = {
            "backbone": {
                "cls": torch.optim.Adam,
                "kwargs": {"lr": cfg.backbone_lr, "weight_decay": cfg.backbone_wd}
            },
            "neck": {
                "cls": torch.optim.AdamW,
                "kwargs": {"lr": cfg.neck_lr, "weight_decay": cfg.neck_wd}
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
            "weight_stratergy": {
                "cls": torch.optim.Adam,
                "kwargs": {"lr": cfg.weight_strategy_lr, "weight_decay": cfg.weight_strategy_wd}
            }
        }

        # Get layer-wise parameter groups for GeATNet if layer decay is enabled
        if cfg.use_layer_decay and cfg.layer_decay_rate > 0:
            print(f"Using layer-wise learning rate decay with rate: {cfg.layer_decay_rate}")
            neck_param_groups = get_geat_layer_parameters(neck, layer_decay=cfg.layer_decay_rate)
            
            # Create parameter groups with scaled learning rates
            neck_params = []
            for group in neck_param_groups:
                base_lr = cfg.neck_lr
                scaled_lr = base_lr * group['lr_scale']
                neck_params.append({
                    'params': group['params'],
                    'lr': scaled_lr,
                    'weight_decay': cfg.neck_wd,
                    'name': group['name']
                })
                print(f"  Layer {group['layer_idx']} ({group['name']}): LR scale = {group['lr_scale']:.4f}, Effective LR = {scaled_lr:.6f}")
            
            # Update neck optimizer configuration
            optimizer_configs["neck"]["kwargs"] = {
                "params": neck_params,
                "lr": cfg.neck_lr,  # Base LR, but individual params have scaled LRs
                "weight_decay": cfg.neck_wd
            }
        else:
            print("Using uniform learning rate for all GeAT layers")

        scheduler_configs = {
            "backbone": {
                "cls": torch.optim.lr_scheduler.OneCycleLR,
                "kwargs": {
                    "max_lr": cfg.backbone_lr,
                    "total_steps": train_loader.total_batches * num_epochs, 
                    "pct_start": cfg.pct_start,
                    "anneal_strategy": cfg.anneal_strategy, 
                    "div_factor": cfg.div_factor, 
                    "final_div_factor": cfg.final_div_factor, 
                }
            },
            "neck": {
                "cls": torch.optim.lr_scheduler.OneCycleLR,
                "kwargs": {
                    "max_lr": cfg.neck_scheduler_max_lr,
                    "total_steps": train_loader.total_batches * num_epochs,
                    "pct_start": cfg.pct_start,
                    "anneal_strategy": cfg.anneal_strategy,
                    "div_factor": cfg.div_factor,
                    "final_div_factor": cfg.final_div_factor,
                }
            },
            "head0": {
                "cls": torch.optim.lr_scheduler.OneCycleLR,
                "kwargs": {
                    "max_lr": cfg.head_lr,
                    "total_steps": train_loader.total_batches * num_epochs,
                    "pct_start": cfg.pct_start,
                    "anneal_strategy": cfg.anneal_strategy,
                    "div_factor": cfg.div_factor,
                    "final_div_factor": cfg.final_div_factor,
                }
            },
            "head1": {
                "cls": torch.optim.lr_scheduler.OneCycleLR,
                "kwargs": {
                    "max_lr": cfg.head_lr,
                    "total_steps": train_loader.total_batches * num_epochs,
                    "pct_start": cfg.pct_start,
                    "anneal_strategy": cfg.anneal_strategy,
                    "div_factor": cfg.div_factor,
                    "final_div_factor": cfg.final_div_factor,
                }
            },
            "head2": {
                "cls": torch.optim.lr_scheduler.OneCycleLR,
                "kwargs": {
                    "max_lr": cfg.head_lr,
                    "total_steps": train_loader.total_batches * num_epochs,
                    "pct_start": cfg.pct_start,
                    "anneal_strategy": cfg.anneal_strategy,
                    "div_factor": cfg.div_factor,
                    "final_div_factor": cfg.final_div_factor,
                }
            },
            "head3": {
                "cls": torch.optim.lr_scheduler.OneCycleLR,
                "kwargs": {
                    "max_lr": cfg.head_lr,
                    "total_steps": train_loader.total_batches * num_epochs,
                    "pct_start": cfg.pct_start,
                    "anneal_strategy": cfg.anneal_strategy,
                    "div_factor": cfg.div_factor,
                    "final_div_factor": cfg.final_div_factor,
                }
            },
            "head4": {
                "cls": torch.optim.lr_scheduler.OneCycleLR,
                "kwargs": {
                    "max_lr": cfg.head_lr,
                    "total_steps": train_loader.total_batches * num_epochs,
                    "pct_start": cfg.pct_start,
                    "anneal_strategy": cfg.anneal_strategy,
                    "div_factor": cfg.div_factor,
                    "final_div_factor": cfg.final_div_factor,
                }
            },
            "weight_stratergy": {
                "cls": torch.optim.lr_scheduler.OneCycleLR,
                "kwargs": {
                    "max_lr": cfg.weight_strategy_lr,
                    "total_steps": train_loader.total_batches * num_epochs,
                    "pct_start": cfg.weight_strategy_pct_start,
                    "anneal_strategy": cfg.anneal_strategy,
                    "div_factor": cfg.weight_strategy_div_factor,
                    "final_div_factor": cfg.final_div_factor,
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
            
            # Special handling for neck when using layer decay
            if name == "neck" and cfg.use_layer_decay and cfg.layer_decay_rate > 0:
                # neck_params already contains parameter groups with individual LRs
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
        
        for epoch in range(num_epochs):
            try:
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
                    task1_labels = batch_data.x[:,0][mask_indices]
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
                    loss = weight_stratergy(losses)

                    # backward and step
                    loss.backward()
                    for opt in optimizers.values():
                        opt.step()
                        
                    for name, scheduler in schedulers.items():
                        scheduler.step()
                    
                    # Log layer-wise learning rates for neck if using layer decay
                    if cfg.use_layer_decay and cfg.layer_decay_rate > 0 and (batch_idx == 0 or (batch_idx + 1) % record_freq == 0):
                        for param_group in optimizers["neck"].param_groups:
                            if 'name' in param_group:
                                writer.add_scalar(f'LR/neck_{param_group["name"]}', param_group['lr'], epoch * train_loader.total_batches + batch_idx)
                    
                    if batch_idx == 0 or (batch_idx + 1) % record_freq == 0:
                        writer.add_scalar('Train/Loss_total', loss.item(), epoch * train_loader.total_batches + batch_idx)
                        writer.add_scalar('Train/Loss_atom_attr', loss_atom_attr_pred.item(), epoch * train_loader.total_batches + batch_idx)
                        writer.add_scalar('Train/Loss_masked_atom', loss_masked_atom_type_pred.item(), epoch * train_loader.total_batches + batch_idx)
                        writer.add_scalar('Train/Loss_triplet', loss_triplet_contrast.item(), epoch * train_loader.total_batches + batch_idx)
                        writer.add_scalar('Train/Loss_batch_contrast', loss_batch_contrast.item(), epoch * train_loader.total_batches + batch_idx)
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
                        metrics = {f"metrics_{i}": tasks[i].get_metrics() for i in range(len(tasks))}
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
                        task1_labels = batch_data.x[:,0][mask_indices]
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

                        loss = weight_stratergy(losses)
                   
                        if batch_idx == val_loader.total_batches - 1:
                            metrics = {f"metrics_{i}": tasks[i].get_metrics() for i in range(len(tasks))}
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