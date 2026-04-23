import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# 忽略警告
warnings.filterwarnings("ignore")

# =========================================================
# 1. 基团贡献参数库 (参考维基百科 Benson Group Increment Theory)
# 单位: kcal/mol
# 格式: "中心原子类型": 贡献值
# =========================================================

# 这里的数值主要参考 NIST 和维基百科关于 Benson Group Increments 的数据
# 注意：这只是一个基础子集，涵盖了常见的 C, H, O, N 化合物
GROUP_CONTRIBUTIONS = {
    # --- 碳氢化合物 (C, H) ---
    "C-(H)3(C)": -10.20,  # 甲基 (Methyl)
    "C-(H)2(C)2": -4.93,  # 亚甲基 (Methylene)
    "C-(H)(C)3": -1.90,   # 次甲基 (Methine)
    "C-(C)4": 0.50,       # 季碳 (Quaternary C)
    
    "C-(H)3(Cd)": -10.00, # 烯丙基甲基 (Allyl methyl)
    "C-(H)2(C)1(Cd)": -4.76, # 烯丙基亚甲基
    "C-(H)3(Cb)": -10.03, # 苯基甲基 (Benzyl methyl)
    
    # 双键碳 (Cd)
    "Cd-(H)2": 6.26,      # 末端双键 =CH2
    "Cd-(H)(C)": 8.59,     # 内部双键 =CH-
    "Cd-(C)2": 10.34,     # 四取代双键 =C<
    "Cd-(C)(Cb)": 9.80,   # 连接苯环的双键碳
    
    # 三键碳 (Ct)
    "Ct-(C)": 27.3,       # 炔烃端基
    "Ct-(Ct)": 30.0,      # 炔烃内部 (估算值)

    # --- 芳香环 (Cb) ---
    "Cb-H": 3.30,         # 苯环上的 CH
    "Cb-(C)": 3.00,       # 苯环上的 C-烷基
    "Cb-(Cb)": 3.00,      # 联苯 (估算)

    # --- 醇和醚 (O) ---
    "O-(H)(C)": -37.90,   # 醇羟基 -OH
    "O-(C)2": -18.00,     # 醚键 -O-
    
    # --- 醛和酮 (CO) ---
    "CO-(H)(C)": -25.00,  # 醛基
    "CO-(C)2": -31.00,    # 酮基

    # --- 羧酸和酯 (COO) ---
    "COO-(H)(C)": -92.0,  # 羧酸 -COOH
    "COO-(C)2": -85.0,    # 酯 -COO-

    # --- 胺 (N) ---
    "N-(H)2(C)": -10.0,   # 伯胺 -NH2
    "N-(H)(C)2": -4.5,    # 仲胺 >NH
    "N-(C)3": 0.0,        # 叔胺 >N-
}

# 转换因子: kcal/mol -> kJ/mol
KCAL_TO_KJ = 4.184

# =========================================================
# 2. 基团识别器
# =========================================================

class BensonGroupAnalyzer:
    """
    使用 RDKit 识别分子中的基团并匹配 Benson 参数
    """
    
    def __init__(self):
        # 预编译 SMARTS 模式以加速匹配
        self.patterns = {
            "C-(H)3(C)": Chem.MolFromSmarts("[CH3][!#1]"), # 简单的甲基，连接非氢
            "C-(H)2(C)2": Chem.MolFromSmarts("[CH2]([!#1])[!#1]"),
            "C-(H)(C)3": Chem.MolFromSmarts("[CH]([!#1])([!#1])[!#1]"),
            "C-(C)4": Chem.MolFromSmarts("[C]([!#1])([!#1])([!#1])[!#1]"),
            
            # 双键
            "Cd-(H)2": Chem.MolFromSmarts("[CH2]=[#6]"),
            "Cd-(H)(C)": Chem.MolFromSmarts("[CH]=[#6]"),
            "Cd-(C)2": Chem.MolFromSmarts("[#6]=[#6]"), # 需后续逻辑过滤
            
            # 芳香
            "Cb-H": Chem.MolFromSmarts("c"), # 芳香碳
            "Cb-(C)": Chem.MolFromSmarts("c-[#6;!$(c)]"),
            
            # 氧
            "O-(H)(C)": Chem.MolFromSmarts("[OH][#6]"),
            "O-(C)2": Chem.MolFromSmarts("[#6][O][#6]"),
            
            # 羰基
            "CO-(H)(C)": Chem.MolFromSmarts("[#6][C](=[O])[H]"), # 醛
            "CO-(C)2": Chem.MolFromSmarts("[#6][C](=[O])[#6]"), # 酮
            
            # 羧酸
            "COO-(H)(C)": Chem.MolFromSmarts("[#6][C](=[O])[OH]"),
            "COO-(C)2": Chem.MolFromSmarts("[#6][C](=[O])[O][#6]"),
        }

    def analyze(self, mol):
        """
        分析分子并返回总焓值 (kJ/mol)
        """
        if mol is None:
            return None, "Invalid Molecule"

        total_h_kcal = 0.0
        groups_found = []

        # 这是一个简化的匹配逻辑。
        # 真正的 Benson 分析非常复杂，需要区分原子环境。
        # 这里使用 SMARTS 匹配作为近似。
        
        # 为了防止重复计数，我们通常需要从原子层面遍历，而不是子结构匹配
        # 这里为了演示，采用简单的子结构计数（存在重叠风险，但在简单分子中有效）
        
        # 实际应用中，建议基于原子的邻居列表手动构建基团字符串
        
        # 简单的原子遍历逻辑 (更准确)
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            num_H = atom.GetTotalNumHs()
            neighbors = atom.GetNeighbors()
            num_heavy = len(neighbors)
            
            # 确定杂化
            hyb = atom.GetHybridization()
            
            # 构建基团键 (简化版)
            group_key = None
            
            # --- 碳逻辑 ---
            if symbol == 'C':
                if hyb == Chem.rdchem.HybridizationType.SP3:
                    if num_heavy == 1: group_key = "C-(H)3(C)"
                    elif num_heavy == 2: group_key = "C-(H)2(C)2"
                    elif num_heavy == 3: group_key = "C-(H)(C)3"
                    elif num_heavy == 4: group_key = "C-(C)4"
                
                elif hyb == Chem.rdchem.HybridizationType.SP2:
                    # 检查是否在芳香环中
                    if atom.GetIsAromatic():
                         # 简单的芳香碳处理
                         if num_H > 0: group_key = "Cb-H"
                         else: group_key = "Cb-(C)" # 假设连接烷基
                    else:
                        # 烯烃
                        # 检查双键
                        is_double = any(bond.GetBondType() == Chem.rdchem.BondType.DOUBLE for bond in atom.GetBonds())
                        if is_double:
                            if num_H == 2: group_key = "Cd-(H)2"
                            elif num_H == 1: group_key = "Cd-(H)(C)"
                            elif num_H == 0: group_key = "Cd-(C)2"
                            
            # --- 氧逻辑 ---
            elif symbol == 'O':
                # 醇/酚
                if num_H > 0:
                    group_key = "O-(H)(C)"
                # 醚
                elif num_heavy == 2:
                    group_key = "O-(C)2"
                # 羰基氧 (双键氧) - 贡献值通常包含在 C=O 基团中，或者单独计算
                # 这里为了简化，如果 O 是双键，我们暂时忽略，假设由 C=O 基团处理
                # 或者需要更复杂的逻辑
                
            # --- 氮逻辑 ---
            elif symbol == 'N':
                 if hyb == Chem.rdchem.HybridizationType.SP3:
                    if num_H == 2: group_key = "N-(H)2(C)"
                    elif num_H == 1: group_key = "N-(H)(C)2"
                    elif num_H == 0: group_key = "N-(C)3"

            # 累加
            if group_key and group_key in GROUP_CONTRIBUTIONS:
                val = GROUP_CONTRIBUTIONS[group_key]
                # 避免重复计算某些复杂基团（如酯基），这里仅作演示
                # 实际代码需要更严格的去重逻辑
                total_h_kcal += val
                groups_found.append(group_key)
            elif group_key:
                # 参数缺失警告
                pass 

        # 修正：简单的 C=O 检测 (如果上面没处理)
        # 这是一个非常粗糙的实现，仅用于演示流程
        # 严谨的实现需要处理 C=O 作为一个整体单元
        
        final_h_kj = total_h_kcal * KCAL_TO_KJ
        return final_h_kj, "OK"

# =========================================================
# 3. 数据加载与处理
# =========================================================

def load_dataset(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = None
    # 尝试不同编码
    for enc in ["utf-8", "gbk", "latin1"]:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except:
            pass

    if df is None:
        raise RuntimeError("Cannot read CSV")

    # 自动寻找列
    smiles_col = None
    value_col = None

    for c in df.columns:
        lc = c.lower()
        if "smiles" in lc: smiles_col = c
        # 寻找生成焓相关的列
        if "enthalpy" in lc or "formation" in lc or "hf" in lc or "value" in lc:
            value_col = c

    if smiles_col is None or value_col is None:
        # 如果找不到，默认取前两列
        if len(df.columns) >= 2:
            smiles_col = df.columns[0]
            value_col = df.columns[1]
        else:
            raise RuntimeError("Cannot find required columns")

    data = pd.DataFrame()
    data["smiles"] = df[smiles_col].astype(str)
    # 确保数值列是数字
    data["exp"] = pd.to_numeric(df[value_col], errors="coerce")
    
    # 清洗数据
    data = data.dropna()
    data = data[data["smiles"].str.len() > 2]
    data = data.reset_index(drop=True)

    return data

# =========================================================
# 4. 预测主循环
# =========================================================

def run_prediction(df):
    analyzer = BensonGroupAnalyzer()
    
    predictions = []
    experimental = []
    valid_smiles = []
    fail_count = 0

    for i, row in df.iterrows():
        smiles = row["smiles"]
        exp_val = row["exp"]
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            fail_count += 1
            continue
            
        pred_val, status = analyzer.analyze(mol)
        
        if pred_val is not None:
            predictions.append(pred_val)
            experimental.append(exp_val)
            valid_smiles.append(smiles)
        else:
            fail_count += 1

    return np.array(experimental), np.array(predictions), fail_count

# =========================================================
# 5. 统计与绘图
# =========================================================

def compute_metrics(exp, pred):
    r2 = r2_score(exp, pred)
    mae = mean_absolute_error(exp, pred)
    rmse = np.sqrt(mean_squared_error(exp, pred))
    return r2, mae, rmse

def plot_results(exp, pred, out_file):
    plt.figure(figsize=(8, 8))
    
    # 计算误差用于颜色映射
    error = np.abs(pred - exp)
    
    plt.scatter(exp, pred, c=error, cmap="viridis", s=60, alpha=0.7, edgecolors="k")
    
    # 对角线
    min_v = min(exp.min(), pred.min())
    max_v = max(exp.max(), pred.max())
    # 稍微扩大一点范围
    margin = (max_v - min_v) * 0.1
    plt.plot([min_v - margin, max_v + margin], [min_v - margin, max_v + margin], "r--", label="Ideal Fit")
    
    plt.xlabel("Experimental $\Delta H_f^\circ$ (kJ/mol)")
    plt.ylabel("Predicted $\Delta H_f^\circ$ (kJ/mol)")
    plt.title("Ideal Gas Enthalpy of Formation Prediction\n(Benson Group Contribution Method)")
    plt.legend()
    
    # 添加颜色条
    cbar = plt.colorbar()
    cbar.set_label("Absolute Error (kJ/mol)")
    
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)

def analyze(csv_path, output_prefix="joback_formation_enthalpy"):
    """
    主分析流程：加载数据 -> 预测 -> 统计 -> 绘图
    """
    print("Loading dataset...")
    try:
        df = load_dataset(csv_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("Total molecules loaded:", len(df))

    print("Running Joback prediction...")
    exp, pred, fail = run_prediction(df)

    success = len(pred)

    print("\nPrediction summary")
    print("Successful:", success)
    print("Failed:", len(df) - success)

    if success < 5:
        print("Too few predictions to calculate statistics.")
        return

    r2, mae, rmse = compute_metrics(exp, pred)

    print("\nStatistics")
    print("R2  :", round(r2, 4))
    print("MAE :", round(mae, 2), "kJ/mol")
    print("RMSE:", round(rmse, 2), "kJ/mol")

    # 保存结果到CSV
    result_df = pd.DataFrame(
        {
            "Experimental": exp,
            "Predicted": pred,
            "Error": pred - exp
        }
    )

    out_csv = output_prefix + "_results.csv"
    result_df.to_csv(out_csv, index=False)
    print("\nSaved results to:", out_csv)

    # 绘图
    plot_file = output_prefix + "_scatter.png"
    plot_results(exp, pred, plot_file)
    print("Saved plot to:", plot_file)
  
# =========================================================
# Main Entry Point
# =========================================================

if __name__ == "__main__":
    # 请在此处修改为你实际的数据文件路径
    # 确保CSV中包含 'SMILES' 列和 'PVCValue' (或包含 'value') 列
    input_csv_path = "./data/data/理想气体生成焓.csv"
    
    # 输出文件的前缀名称
    output_name = "joback_formation_enthalpy"

    try:
        print(f"Starting Joback analysis for: {input_csv_path}")
        analyze(input_csv_path, output_prefix=output_name)
    except Exception as e:
        print(f"Error occurred: {e}")