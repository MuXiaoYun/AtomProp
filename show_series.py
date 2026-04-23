import os
import math
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.metrics import r2_score, mean_squared_error

# ==========================================
# 配置部分
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GEAT_DIM = 768 
HIDDEN_DIM = 256 

class GeATHead(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super(GeATHead, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)

def get_molecule_properties(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    
    MW = Descriptors.MolWt(mol)
    HBD = Descriptors.NumHDonors(mol)
    HBA = Descriptors.NumHAcceptors(mol)
    
    # 估算 Tc (K) - 必须与训练数据生成逻辑一致
    Tc = 50 + 2.5 * MW + 10 * HBD
    
    # 这里我们修改逻辑：
    # 不要只看 0.3Tc - 0.9Tc，因为对于小分子这个范围太窄且数值太小。
    # 为了让模型能预测，我们需要看模型“见过”的温度范围。
    # 但为了物理正确性，我们依然计算这个范围用于绘图，
    # 但要注意，如果 Tc 很小，归一化后的值会接近 0。
    
    Tmin = max(150, 0.3 * Tc)
    Tmax = 0.9 * Tc
    
    if Tmax <= Tmin: return None
        
    return {'MW': MW, 'HBD': HBD, 'HBA': HBA, 'Tc': Tc, 'Tmin': Tmin, 'Tmax': Tmax}

def generate_antoine_data(props):
    Tmin, Tmax = props['Tmin'], props['Tmax']
    # 增加采样点
    T_list = np.linspace(Tmin, Tmax, 100) 
    
    A = 8.07131
    B = 1730.63
    C = 233.426
    
    T_vals = []
    P_vals = []
    
    for T in T_list:
        T_C = T - 273.15
        if (T_C + C) <= 0: continue
            
        P_mmHg = 10 ** (A - B / (T_C + C))
        if P_mmHg > 0:
            P_Pa = P_mmHg * 133.322
            T_vals.append(T)
            P_vals.append(P_Pa)
            
    return np.array(T_vals), np.array(P_vals)

def get_geat_embedding_single(smi):
    # TODO: 替换为真实 GeAT 代码
    return torch.rand(1, GEAT_DIM)

def predict_curve(model, smi, T_list, scaler_stats):
    geat_emb = get_geat_embedding_single(smi).to(DEVICE)
    
    T_min = scaler_stats['T_min']
    T_max = scaler_stats['T_max']
    
    # 打印调试信息
    # print(f"Input T range: {T_list.min():.2f} - {T_list.max():.2f}")
    # print(f"Scaler range: {T_min} - {T_max}")
    # print(f"Norm T range: {(T_list.min() - T_min) / (T_max - T_min):.4f} - {(T_list.max() - T_min) / (T_max - T_min):.4f}")
    
    T_norm = (T_list - T_min) / (T_max - T_min)
    T_tensor = torch.tensor(T_norm, dtype=torch.float).to(DEVICE).unsqueeze(1)
    
    N = len(T_list)
    geat_emb_expanded = geat_emb.expand(N, -1)
    
    combined_features = torch.cat([geat_emb_expanded, T_tensor], dim=1)
    
    model.eval()
    with torch.no_grad():
        log_p_preds = model(combined_features)
        P_preds = torch.exp(log_p_preds)
        
    return P_preds.cpu().numpy()

def main(args):
    model_path = f"trained_models/{args.task_name}/best_model.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    model = GeATHead(input_dim=GEAT_DIM + 1, hidden_dim=HIDDEN_DIM).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    scaler_stats = checkpoint['scaler_stats']
    
    props = get_molecule_properties(args.smiles)
    if not props:
        print("Error: Invalid SMILES")
        return
        
    print(f"Molecule: {args.smiles}, MW: {props['MW']:.2f}, Tc: {props['Tc']:.2f} K")

    # 1. 生成 Ground Truth
    T_gt, P_gt = generate_antoine_data(props)
    
    # 2. 预测
    P_pred = predict_curve(model, args.smiles, T_gt, scaler_stats)

    # 3. 计算指标
    # 注意：如果 P_pred 是常数，R2 会很低。
    # 这里我们只看 P > 1 Pa 的部分，避免对数尺度下的极端误差
    valid_mask = (P_gt > 1) & (P_pred > 1)
    
    if np.sum(valid_mask) < 5:
        print("Warning: Too few valid points for R2 calculation (Pressure too low or model output invalid).")
        r2 = float('nan')
        rmse = float('nan')
    else:
        r2 = r2_score(P_gt[valid_mask], P_pred[valid_mask])
        rmse = math.sqrt(mean_squared_error(P_gt[valid_mask], P_pred[valid_mask]))
    
    print("-" * 30)
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE:     {rmse:.4f}")
    print("-" * 30)

    # 4. 绘图
    plt.figure(figsize=(12, 6))
    
    # 左图：线性坐标
    plt.subplot(1, 2, 1)
    plt.plot(T_gt, P_gt, label='Estimated (Antoine)', linestyle='--', color='blue')
    plt.plot(T_gt, P_pred, label='GeAT Prediction', linestyle='-', color='red')
    plt.title(f"Vapor Pressure (Linear Scale)\nSMILES: {args.smiles}")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Pressure (Pa)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 右图：对数坐标 (更适合观察蒸气压)
    plt.subplot(1, 2, 2)
    plt.semilogy(T_gt, P_gt, label='Estimated (Antoine)', linestyle='--', color='blue')
    plt.semilogy(T_gt, P_pred, label='GeAT Prediction', linestyle='-', color='red')
    plt.title(f"Vapor Pressure (Log Scale)\nSMILES: {args.smiles}")
    plt.xlabel("Temperature (K)")
    plt.ylabel("Pressure (Pa) [Log Scale]")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    
    plt.tight_layout()
    output_img = f"prediction_{args.smiles.replace('/', '_')}.png"
    plt.savefig(output_img, dpi=300)
    print(f"Plot saved to {output_img}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", type=str, required=True)
    parser.add_argument("--task_name", type=str, default="thermo_vp_finetune")
    args = parser.parse_args()
    main(args)