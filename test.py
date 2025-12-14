from atomprop.dataloader.splitter import ScaffoldKFoldSplitter
import deepchem as dc
import numpy as np

smiles_list = [
    "CC(C)Cl", "CCC(C)CO", "CCCCCCCO", "CCCCCCCC(=O)OC", 
    "c3ccc2nc1ccccc1cc2c3", "Nc2cccc3nc1ccccc1cc23", "C1CCCCCC1",
    "CCN(CC)CC", "CCOC(=O)C", "CC(=O)OC", "CCCCN", "CCCCCC",
    "c1ccccc1", "c1ccccc1O", "c1ccccc1Cl", "c1ccccc1Br",
    "CC(=O)N", "CCN", "CCCC(=O)O", "CC(C)C"
] * 3  # 扩展到60个分子

Xs = np.zeros(len(smiles_list))
Ys = np.ones(len(smiles_list))
dataset = dc.data.DiskDataset.from_numpy(
    X=Xs, y=Ys, w=np.zeros(len(smiles_list)), ids=smiles_list
)

# 创建分割器
splitter = ScaffoldKFoldSplitter(fold=3, frac_test=0.1)

# 测试所有fold
print("Testing K-Fold splits:")
for fold_idx, (train_inds, valid_inds, test_inds) in enumerate(splitter.k_fold_split(dataset)):
    print(f"Fold {fold_idx + 1}: Train={len(train_inds)}, Valid={len(valid_inds)}, Test={len(test_inds)}")
    print(f"  Train+Valid ratio: {len(train_inds)/len(valid_inds):.2f}:1")

# 测试第一个fold的数据集
print("\nFirst fold datasets:")
train_ds, valid_ds, test_ds = splitter.train_valid_test_split(dataset)
print(f"Train dataset size: {len(train_ds)}")
print(f"Valid dataset size: {len(valid_ds)}")
print(f"Test dataset size: {len(test_ds)}")