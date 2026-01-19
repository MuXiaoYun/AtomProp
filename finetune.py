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
from tqdm import tqdm
from torch_geometric.data import Data, Batch, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, roc_curve
from atomprop.dataloader.splitter import ScaffoldSplitter
from deepchem.data import NumpyDataset
import csv
import os
import json
from datetime import datetime
from atomprop.models.GeAT import GeATNet
import configs.config_finetune as cfg
from atomprop.utils.head import DownstreamHead

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from atomprop.utils.utils import remove_module_prefix

def setup_distributed():
    """Initialize distributed training environment."""
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))


def cleanup_distributed():
    """Clean up distributed training environment."""
    dist.destroy_process_group()


criterion = MaskedBCELoss()


def create_dataset_from_smiles_labels(smiles_list, labels_list):
    """Create PyG dataset from SMILES and labels"""
    dataset = []
    for smi, label in zip(smiles_list, labels_list):
        atom_info, edge_info, mol = SMILESToInputs.convert(smi, sanitize=False)
        if atom_info is None or edge_info is None:
            continue
        if edge_info.dim() == 2 and edge_info.size(1) == 4:
            edge_index = edge_info[:, :2].t().contiguous()
            edge_attr = edge_info[:, 2:]
        else:
            edge_index = torch.tensor([[], []], dtype=torch.long)
            edge_attr = torch.tensor([], dtype=torch.long).view(0, 2)
            
        data = Data(x=atom_info, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor(label))
        dataset.append(data)
    
    return dataset


def evaluate_model(model_components, dataloader, criterion, y_cols, device, aggr='attention'):
    """Evaluate model performance"""
    embedding_layer, backbone, head, aggrmodel = model_components
    
    embedding_layer.eval()
    backbone.eval()
    head.eval()
    if aggr == 'attention':
        aggrmodel.eval()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            emb = embedding_layer(batch.x.squeeze())
            emb = backbone(Data(x=emb, edge_index=batch.edge_index, edge_attr=batch.edge_attr), batch=batch.batch)
            graph_emb = aggrmodel(emb, batch.batch)
            preds = head(graph_emb)
            
            loss = criterion(preds.reshape(-1, len(y_cols)), batch.y.reshape(-1, len(y_cols)))
            total_loss += loss.item()
            
            preds_np = F.sigmoid(preds).cpu().numpy()
            labels_np = batch.y.reshape(-1, len(y_cols)).cpu().numpy()
            all_preds.append(preds_np)
            all_labels.append(labels_np)
    
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    
    if len(all_preds) > 0:
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        task_aucs = []
        
        for col_idx in range(len(y_cols)):
            valid_mask = all_labels[:, col_idx] != -1
            if valid_mask.sum() == 0:
                continue
            
            valid_labels = all_labels[valid_mask, col_idx]
            valid_preds = all_preds[valid_mask, col_idx]
            
            if len(np.unique(valid_labels)) < 2:
                continue
            
            try:
                auc = roc_auc_score(valid_labels, valid_preds)
                task_aucs.append(auc)
            except Exception:
                pass
        
        mean_auc = np.nanmean(task_aucs) if len(task_aucs) > 0 else float('nan')
    else:
        mean_auc = float('nan')
        task_aucs = []
    
    return avg_loss, mean_auc, task_aucs, all_preds, all_labels


def train(train_dataloader, val_dataloader, test_dataloader, model_components, optimizers, schedulers, device, num_epochs, y_cols, logdir, no_pretrain, aggr='attention'):
    """Train using DDP across all ranks"""
    embedding_layer, backbone, head, aggrmodel = model_components
    local_rank = int(os.environ['LOCAL_RANK'])
    if local_rank == 0:
        writer = SummaryWriter(log_dir=f'runs/finetune_{logdir}')
    else:
        writer = None
    
    best_val_auc = 0.0
    best_epoch = -1
    global_step = 0
    
    for epoch in range(num_epochs):
        embedding_layer.train()
        backbone.train()
        head.train()
        if cfg.aggr == 'attention':
            aggrmodel.train()
        
        # Set sampler epoch for shuffling
        train_dataloader.sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        num_batches = 0
        
        # calculate negative ratio as alphas
        pos_count = torch.zeros(len(y_cols), device='cpu')
        neg_count = torch.zeros(len(y_cols), device='cpu')
        total_valid = torch.zeros(len(y_cols), device='cpu')
        for batch in train_dataloader:
            y = batch.y.reshape(-1,len(y_cols))
            valid_mask = (y != -1)
            pos_mask = (y == 1)
            neg_mask = (y == 0)
            pos_count += pos_mask.sum(dim=0).cpu()
            neg_count += neg_mask.sum(dim=0).cpu()
            total_valid += valid_mask.sum(dim=0).cpu()
        neg_ratio = neg_count / total_valid
        train_criterion = MaskedFocalLoss(alpha=neg_ratio, gamma=cfg.gamma, reduction='mean')
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training", disable=(local_rank != 0)):
            batch = batch.to(device)
            
            for optimizer in optimizers:
                optimizer.zero_grad()
            
            emb = embedding_layer(batch.x.squeeze())
            emb = backbone(Data(x=emb, edge_index=batch.edge_index, edge_attr=batch.edge_attr), batch=batch.batch)
            graph_emb = aggrmodel(emb, batch.batch)
            preds = head(graph_emb)
            
            loss = train_criterion(preds.reshape(-1, len(y_cols)), batch.y.reshape(-1, len(y_cols))) - cfg.norm_lambda * torch.mean(torch.abs(preds.reshape(-1, len(y_cols))-0.5))
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            if writer is not None:
                writer.add_scalar(f'Train/Loss', loss.item(), global_step)
            global_step += 1
        
        # Reduce loss across ranks
        epoch_loss_tensor = torch.tensor([epoch_loss / num_batches if num_batches > 0 else 0.0], device=device)
        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.AVG)
        avg_epoch_loss = epoch_loss_tensor.item()
        
        if local_rank == 0:
            print(f"Epoch {epoch+1} Training Loss: {avg_epoch_loss:.4f}")
        
        for scheduler in schedulers:
            scheduler.step()
        
        val_loss, val_auc, _, _, _ = evaluate_model(
            model_components, val_dataloader, criterion, y_cols, device, cfg.aggr
        )
        
        # Gather validation AUC from all ranks (use rank 0's value since evaluation is identical)
        val_auc_tensor = torch.tensor([val_auc if not np.isnan(val_auc) else 0.0], device=device)
        dist.broadcast(val_auc_tensor, src=0)
        val_auc = val_auc_tensor.item()
        
        if local_rank == 0:
            print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}, AUC: {val_auc:.4f}")
            if writer is not None:
                writer.add_scalar(f'Val/Loss', val_loss, epoch)
                writer.add_scalar(f'Val/AUC', val_auc, epoch)
        
        if local_rank == 0 and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            
            model_suffix = "nopretrain" if no_pretrain else "pretrained"
            save_path = f"trained_models/{logdir}/best_model_{model_suffix}.pth"
            
            torch.save({
                'epoch': epoch + 1,
                'embedding_layer_state_dict': embedding_layer.module.state_dict(),
                'backbone_state_dict': backbone.module.state_dict(),
                'head_state_dict': head.module.state_dict(),
                'aggr_state_dict': aggrmodel.module.state_dict() if cfg.aggr == 'attention' else None,
                'val_auc': val_auc,
                'val_loss': val_loss,
                'optimizer_embedding_layer_backbone_state_dict': optimizers[0].state_dict(),
                'optimizer_head_state_dict': optimizers[1].state_dict(),
                'optimizer_aggr_state_dict': optimizers[2].state_dict() if len(optimizers) > 2 else None,
            }, save_path)
            
            print(f"Best model saved at epoch {best_epoch} with validation AUC: {best_val_auc:.4f}")
    
    if writer is not None:
        writer.close()
    
    # Test phase (only rank 0 saves results)
    if local_rank == 0:
        print(f"\n--- Testing ---")
        model_suffix = "nopretrain" if cfg.no_pretrain else "pretrained"
        load_path = f"trained_models/{logdir}/best_model_{model_suffix}.pth"
        checkpoint = torch.load(load_path, weights_only=False, map_location=device)
        
        embedding_layer.module.load_state_dict(checkpoint['embedding_layer_state_dict'])
        backbone.module.load_state_dict(checkpoint['backbone_state_dict'])
        head.module.load_state_dict(checkpoint['head_state_dict'])
        if cfg.aggr == 'attention' and checkpoint['aggr_state_dict']:
            aggrmodel.module.load_state_dict(checkpoint['aggr_state_dict'])
    
    # Sync before evaluation
    dist.barrier()
    
    test_loss, test_auc, test_task_aucs, all_test_preds, all_test_labels = evaluate_model(
        model_components, test_dataloader, criterion, y_cols, device, cfg.aggr
    )
    
    if local_rank == 0:
        print(f"Test Results:")
        print(f"  Test Loss: {test_loss:.4f}")
        print(f"  Mean Test AUC: {test_auc:.4f}")
        
        output_csv_path = f"trained_models/{logdir}/test_predictions_{model_suffix}.csv"
        if len(all_test_preds) > 0:
            all_test_preds = np.vstack(all_test_preds)
            all_test_labels = np.vstack(all_test_labels)
            
            with open(output_csv_path, mode='w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(y_cols)
                
                for i in range(all_test_preds.shape[0]):
                    row_preds = all_test_preds[i].tolist()
                    row_labels = all_test_labels[i].astype(int).tolist()
                    row_labels = [lbl if lbl != -1 else "" for lbl in row_labels]
                    csv_writer.writerow(row_preds)
                    csv_writer.writerow(row_labels)
                    csv_writer.writerow([])
    else:
        output_csv_path = ""
        test_task_aucs = []
    
    return {
        'best_val_auc': best_val_auc if local_rank == 0 else 0.0,
        'best_epoch': best_epoch if local_rank == 0 else 0,
        'test_auc': test_auc if local_rank == 0 else 0.0,
        'test_loss': test_loss if local_rank == 0 else 0.0,
        'test_task_aucs': test_task_aucs,
        'test_predictions_path': output_csv_path
    }


def main(ft_dataset=None):
    setup_distributed()
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = dist.get_world_size()
    device = torch.device('cuda', local_rank)

    # 1. Load data
    parser = argparse.ArgumentParser(description='Finetuning script')
    parser.add_argument('--dataset', type=str, default='', help='The downstream task for finetuning.')
    args = parser.parse_args()
    
    if ft_dataset is None:
        ft_dataset = args.dataset

    if len(ft_dataset) > 0:
        cfg.set_data_path(ft_dataset)
        if local_rank == 0:
            print(f"Finetuning on {ft_dataset}...")
    
    if local_rank == 0:
        os.makedirs(f"trained_models/{cfg.logdir}", exist_ok=True)
    dist.barrier()

    df = pd.read_csv(cfg.data_path)
    headers = df.columns.tolist()
    y_cols = [col for col in headers if col not in cfg.exclude_list + [cfg.x_col]]
    
    smiles_list = df[cfg.x_col].tolist()
    df[y_cols] = df[y_cols].fillna(-1).astype(float)
    labels = df[y_cols].values.tolist()
    
    if local_rank == 0:
        cfg.print_all_params()
    
    # 2. Initialize base model components (for weight loading if needed)
    embedding_layer = Embedder(num_atom_types=120, embed_dim=cfg.embed_dim)
    backbone = GeATNet(embed_dim=cfg.embed_dim,
                   num_heads=cfg.num_heads,
                   global_num_heads=cfg.global_num_heads,
                   output_negative_slope=cfg.output_negative_slope,
                   dropout=cfg.geat_dropout,
                   geat_num_layers=cfg.geat_num_layers,
                   aggr_num_layers=cfg.aggr_num_layers,
                   FFN_type=cfg.FFN_type,
                   FFN_hidden_dim=cfg.FFN_hidden_dim,
                   FFN_num_experts=cfg.FFN_num_experts,
                   FFN_num_layers=cfg.FFN_num_layers,
                   FFN_top_k=cfg.FFN_top_k,
                   use_edge_embedding=cfg.use_edge_embedding)
    aggrmodel = GNNAggr(embed_dim=cfg.embed_dim, aggr=cfg.aggr, layers=1)
    
    if local_rank == 0:
        backbone.print_params()
    
    if cfg.no_pretrain == False:
        ckpt = torch.load(cfg.pretrained_path, weights_only=False, map_location=device)
        embedding_layer.load_state_dict(remove_module_prefix(ckpt['embedding_layer_state_dict']))
        backbone.load_state_dict(remove_module_prefix(ckpt['backbone_state_dict']))
    
    head = MLP(input_dim=cfg.embed_dim, hidden_dim=cfg.head_hidden_dim, output_dim=len(y_cols),
                    num_layers=2, dropout=cfg.head_dropout, batch_norm=True, output_activation=None)
    # head = DownstreamHead(
    #     input_dim=cfg.embed_dim,
    #     hidden_dim=cfg.head_hidden_dim,
    #     output_dim=len(y_cols),
    #     mlp_num_layers=2,
    #     attn_num_layers=cfg.downstream_head_attn_num_layers,
    #     dropout=cfg.head_dropout,
    #     batch_norm=True,
    #     output_activation=None
    # )
    
    if local_rank == 0:
        print(f"embedding_layer Parameters: {sum(p.numel() for p in embedding_layer.parameters() if p.requires_grad)}")
        print(f"backbone Parameters: {sum(p.numel() for p in backbone.parameters() if p.requires_grad)}")
        print(f"Head Parameters: {sum(p.numel() for p in head.parameters() if p.requires_grad)}")
    
    # 3. Prepare dataset and splitter
    dc_dataset = NumpyDataset(X=labels, ids=smiles_list)
    splitter = ScaffoldSplitter()
    
    # 4. Train
    all_results = []
    if local_rank == 0:
        print(f"\nStarting...")
    
    train_inds, valid_inds, test_inds = splitter.split(dc_dataset)
    
    train_smiles = [dc_dataset.ids[i] for i in train_inds]
    train_labels = dc_dataset.X[train_inds]
    val_smiles = [dc_dataset.ids[i] for i in valid_inds]
    val_labels = dc_dataset.X[valid_inds]
    test_smiles = [dc_dataset.ids[i] for i in test_inds]
    test_labels = dc_dataset.X[test_inds]
    
    if local_rank == 0:
        print(f"Training set size: {len(train_smiles)}")
        print(f"Validation set size: {len(val_smiles)}")
        print(f"Test set size: {len(test_smiles)}")
    
    train_dataset = create_dataset_from_smiles_labels(train_smiles, train_labels)
    val_dataset = create_dataset_from_smiles_labels(val_smiles, val_labels)
    test_dataset = create_dataset_from_smiles_labels(test_smiles, test_labels)
    
    if local_rank == 0:
        print(f"Valid training graphs: {len(train_dataset)}")
        print(f"Valid validation graphs: {len(val_dataset)}")
        print(f"Valid test graphs: {len(test_dataset)}")
    
    # Use DistributedSampler for all datasets
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=train_sampler, collate_fn=Batch.from_data_list, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, sampler=val_sampler, collate_fn=Batch.from_data_list, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, sampler=test_sampler, collate_fn=Batch.from_data_list, num_workers=0)
    
    embedding_layer = embedding_layer.to(device)
    backbone = backbone.to(device)
    head = head.to(device)
    if cfg.aggr == 'attention':
        aggrmodel = aggrmodel.to(device)
    
    # Wrap with DDP, enable find_unused_parameters
    embedding_layer = DDP(embedding_layer, device_ids=[local_rank], find_unused_parameters=True)
    backbone = DDP(backbone, device_ids=[local_rank], find_unused_parameters=True)
    head = DDP(head, device_ids=[local_rank], find_unused_parameters=True)
    if cfg.aggr == 'attention':
        aggrmodel = DDP(aggrmodel, device_ids=[local_rank], find_unused_parameters=True)
    
    # Optimizers
    optimizer_embedding_layer_backbone = torch.optim.Adam([
        {'params': embedding_layer.parameters(), 'lr': cfg.lr_embedding_layer_backbone},
        {'params': backbone.parameters(), 'lr': cfg.lr_embedding_layer_backbone}
    ])
    optimizer_head = torch.optim.Adam(head.parameters(), lr=cfg.lr_head)
    optimizers = [optimizer_embedding_layer_backbone, optimizer_head]
    
    # Schedulers
    scheduler_embedding_layer_backbone = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_embedding_layer_backbone, T_max=cfg.T_max, eta_min=cfg.eta_min_embedding_layer_backbone
    )
    scheduler_head = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_head, T_max=cfg.T_max, eta_min=cfg.eta_min_head
    )
    schedulers = [scheduler_embedding_layer_backbone, scheduler_head]
    
    model_components = (embedding_layer, backbone, head, aggrmodel)
    
    try:
        result = train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            model_components=model_components,
            optimizers=optimizers,
            schedulers=schedulers,
            device=device,
            num_epochs=cfg.num_epochs,
            y_cols=y_cols,
            logdir=cfg.logdir,
            no_pretrain=cfg.no_pretrain,
            aggr=cfg.aggr
        )
        if local_rank == 0:
            all_results.append(result)
    except KeyboardInterrupt:
        if local_rank == 0:
            print(f"\nTraining interrupted. Saving current progress...")
    except Exception as e:
        if local_rank == 0:
            print(f"\nError: {e}")

    # Only rank 0 collects and saves results
    if local_rank == 0:
        # 5. Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        if len(all_results) > 0:
            val_aucs = []
            test_aucs = []
            print("\nIndividual Results:")
            print("-" * 40)
            for result in all_results:
                print(f"  Best Validation AUC: {result['best_val_auc']:.4f} (epoch {result['best_epoch']})")
                print(f"  Test AUC: {result['test_auc']:.4f}")
                print(f"  Test Loss: {result['test_loss']:.4f}\n")
                val_aucs.append(result['best_val_auc'])
                test_aucs.append(result['test_auc'])
            
            print("\nSummary Statistics:")
            print("-" * 40)
            print(f"Mean Validation AUC: {np.mean(val_aucs):.4f} ± {np.std(val_aucs):.4f}")
            print(f"Mean Test AUC: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
            print(f"Min Test AUC: {np.min(test_aucs):.4f}")
            print(f"Max Test AUC: {np.max(test_aucs):.4f}")
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'dataset': cfg.data_path,
                'frac_test': cfg.frac_test,
                'num_epochs': cfg.num_epochs,
                'batch_size': cfg.batch_size,
                'cfg.no_pretrain': cfg.no_pretrain,
                'aggr': cfg.aggr,
                'results': all_results,
                'summary_stats': {
                    'mean_val_auc': float(np.mean(val_aucs)),
                    'std_val_auc': float(np.std(val_aucs)),
                    'mean_test_auc': float(np.mean(test_aucs)),
                    'std_test_auc': float(np.std(test_aucs)),
                    'min_test_auc': float(np.min(test_aucs)),
                    'max_test_auc': float(np.max(test_aucs)),
                }
            }
            
            summary_path = f"trained_models/{cfg.logdir}/summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\nSummary saved to: {summary_path}")
        else:
            print("TRAINING FAILED!")
        
        print(f"\nCompleted. Results saved in 'trained_models/{cfg.logdir}/'")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()