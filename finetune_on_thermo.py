import os
import math
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error
import json

# ==========================================
# 1. 配置与超参数
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GEAT_DIM = 768  # 请根据你的 GeAT 模型实际输出维度修改
HIDDEN_DIM = 256
DEFAULT_TASK_NAME = "thermo_vp_finetune"

class ThermoDataset(Dataset):
    """
    自定义数据集类，支持 GeAT 表征预加载
    """
    def __init__(self, dataframe, t_min=None, t_max=None):
        self.data = dataframe
        self.smiles = dataframe['smiles'].tolist()
        
        # 温度归一化参数
        self.T_min = t_min if t_min is not None else self.data['T'].min()
        self.T_max = t_max if t_max is not None else self.data['T'].max()
        
        if self.T_max == self.T_min:
            self.T_max += 1e-6
            
        # 目标值：预测 ln(P)
        self.targets = np.log(self.data['P'].values)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        smiles = self.data.iloc[index]['smiles']
        T_raw = self.data.iloc[index]['T']
        
        # 归一化温度
        T_norm = (T_raw - self.T_min) / (self.T_max - self.T_min)
        
        target = self.targets[index]

        return {
            'smiles': smiles,
            'T_norm': torch.tensor(T_norm, dtype=torch.float),
            'target': torch.tensor(target, dtype=torch.float)
        }

class GeATHead(nn.Module):
    """
    预测头：接收 [GeAT_Embedding, T_norm] 拼接向量
    """
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super(GeATHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)

def get_geat_embedding(smiles_list, batch_size=32):
    """
    模拟 GeAT 模型推理 (请替换为真实模型代码)
    """
    print(f"Generating embeddings for {len(smiles_list)} molecules...")
    embeddings = []
    
    # TODO: 替换为真实的 GeAT 模型加载和推理逻辑
    # 示例: 
    # model.eval()
    # with torch.no_grad():
    #   for batch in DataLoader(...):
    #       emb = model(batch)
    #       embeddings.append(emb.cpu())
    
    # 模拟代码 (生成随机向量):
    for i in range(0, len(smiles_list), batch_size):
        batch_size_actual = min(batch_size, len(smiles_list) - i)
        fake_embed = torch.rand(batch_size_actual, GEAT_DIM)
        embeddings.append(fake_embed)
        
    return torch.cat(embeddings, dim=0)

def create_data_loaders(args, train_df, val_df, test_df):
    """
    创建 Dataloader，并处理 GeAT 表征
    """
    # 1. 获取所有唯一 SMILES 的表征 (去重加速)
    all_smiles = list(set(train_df['smiles'].tolist() + 
                          val_df['smiles'].tolist() + 
                          test_df['smiles'].tolist()))
    smiles_to_idx = {smi: idx for idx, smi in enumerate(all_smiles)}
    geat_embs_tensor = get_geat_embedding(all_smiles) # [N_unique, D]
    
    # 2. 定义 Dataset (直接传入预计算的 Embedding 索引)
    class ThermoDatasetWithEmbed(ThermoDataset):
        def __init__(self, dataframe, geat_embs, smiles_to_idx, t_min=None, t_max=None):
            super().__init__(dataframe, t_min, t_max)
            self.geat_embs = geat_embs
            self.smiles_to_idx = smiles_to_idx
            
        def __getitem__(self, index):
            item = super().__getitem__(index)
            smi = self.data.iloc[index]['smiles']
            emb_idx = self.smiles_to_idx[smi]
            item['geat_emb'] = self.geat_embs[emb_idx]
            return item

    # 3. 构建 Dataset 和 DataLoader
    # 计算全局 T_min/T_max 用于归一化
    global_t_min = min(train_df['T'].min(), val_df['T'].min(), test_df['T'].min())
    global_t_max = max(train_df['T'].max(), val_df['T'].max(), test_df['T'].max())

    train_dataset = ThermoDatasetWithEmbed(train_df, geat_embs_tensor, smiles_to_idx, global_t_min, global_t_max)
    val_dataset = ThermoDatasetWithEmbed(val_df, geat_embs_tensor, smiles_to_idx, global_t_min, global_t_max)
    test_dataset = ThermoDatasetWithEmbed(test_df, geat_embs_tensor, smiles_to_idx, global_t_min, global_t_max)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def train_and_evaluate(args):
    # 1. 读取数据
    print(f"Loading data from {args.data_file}...")
    df = pd.read_csv(args.data_file)
    
    # 2. Debug 模式：采样子集
    if args.max_samples > 0 and args.max_samples < len(df):
        print(f"Debug mode: Sampling {args.max_samples} rows...")
        df = df.sample(n=args.max_samples, random_state=42).reset_index(drop=True)
    
    # 3. 简单划分训练/验证/测试集 (8:1:1)
    train_df = df.sample(frac=0.8, random_state=42)
    temp_df = df.drop(train_df.index)
    val_df = temp_df.sample(frac=0.5, random_state=42)
    test_df = temp_df.drop(val_df.index)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 4. 创建 Dataloader (包含 GeAT 表征预计算)
    train_loader, val_loader, test_loader = create_data_loaders(args, train_df, val_df, test_df)

    # 5. 初始化模型
    model = GeATHead(input_dim=GEAT_DIM + 1, hidden_dim=HIDDEN_DIM).to(DEVICE) # +1 for T
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # 6. 模型保存路径 (效仿微调脚本)
    task_name = args.task_name if args.task_name else DEFAULT_TASK_NAME
    save_dir = f"trained_models/{task_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    best_metrics = {}

    print(f"Starting training on {DEVICE}...")

    # 7. 训练循环
    for epoch in range(args.epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            geat_emb = batch['geat_emb'].to(DEVICE) # [B, D]
            T_norm = batch['T_norm'].to(DEVICE)    # [B]
            targets = batch['target'].to(DEVICE)   # [B]
            
            # 特征拼接
            combined_features = torch.cat([geat_emb, T_norm.unsqueeze(1)], dim=1)
            
            optimizer.zero_grad()
            outputs = model(combined_features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                geat_emb = batch['geat_emb'].to(DEVICE)
                T_norm = batch['T_norm'].to(DEVICE)
                targets = batch['target'].to(DEVICE)
                
                combined_features = torch.cat([geat_emb, T_norm.unsqueeze(1)], dim=1)
                preds = model(combined_features)
                
                val_loss += criterion(preds, targets).item()
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # 计算 R^2 和 RMSE
        r2 = r2_score(all_targets, all_preds)
        rmse = math.sqrt(mean_squared_error(all_targets, all_preds))
        
        print(f"Epoch {epoch+1:2d} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val R²: {r2:.4f} | "
              f"Val RMSE: {rmse:.4f}")

        # 8. 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_metrics = {
                'epoch': epoch + 1,
                'val_loss': avg_val_loss,
                'val_r2': r2,
                'val_rmse': rmse
            }
            
            # 保存模型状态
            save_path = f"{save_dir}/best_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_r2': r2,
                'scaler_stats': {'T_min': train_loader.dataset.T_min, 'T_max': train_loader.dataset.T_max},
                'args': vars(args)
            }, save_path)
            
            print(f"   -> Best model saved (Val Loss improved)")

    # 9. 最终测试与总结
    print("\n" + "="*50)
    print("Training Completed. Best Validation Metrics:")
    print(json.dumps(best_metrics, indent=2))
    
    # 如果需要，可以在这里加载最佳模型并对 Test Set 进行最终评估
    # model.load_state_dict(torch.load(f"{save_dir}/best_model.pth")['model_state_dict'])
    # final_test_metrics = evaluate(model, test_loader)
    # print("Final Test Metrics:", final_test_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune GeAT with Temperature for Vapor Pressure")
    
    # 数据参数
    parser.add_argument("--data_file", type=str, default="thermo_vapor_pressure_estimated.csv", help="Path to dataset")
    parser.add_argument("--max_samples", type=int, default=-1, help="Number of samples for debug (-1 for all)")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--task_name", type=str, default=None, help="Subdirectory name under trained_models")
    
    args = parser.parse_args()
    train_and_evaluate(args)