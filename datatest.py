from atomprop.tasks.tasks import NodeAttrPrediction, MaskedNodePrediction, GraphMaskContrast, BatchContrast, BondAnglePrediction, DihedralAnglePrediction
from atomprop.dataloader.dataloader import SMILESToInputs, PyGChunkDataListLoader, xyzBatchLoader, xyzBatchLoaderContext
from atomprop.models.GNNs import Embedder, GNN, GNNAggr
from atomprop.utils.mlp import MLP
from atomprop.utils.mask import MolGraphMask
from atomprop.utils.groups import TripletGroup, QuadrupletGroup
from atomprop.embeddings.AtomEmbedding import BondTypes
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data, Batch
from torch.utils.tensorboard import SummaryWriter
import os
from rdkit import Chem

smiles = "CC(C)(C)[Si](C)(C)[Si]1=[Si]([Si](C(C)(C)C)(C(C)(C)C)C(C)(C)C)[Si+]1[Si](C(C)(C)C)(C(C)(C)C)C(C)(C)C"

from rdkit.Chem import FunctionalGroups

def print_all_functional_groups():
    """打印所有预定义的官能团模式"""
    fg_list = FunctionalGroups.BuildFuncGroupHierarchy()
    for fg in fg_list:
        print(f"Label: {fg.label}, Pattern: {fg.pattern}")

def detect_with_rdkit_fg(mol):
    """使用RdKit内置的FunctionalGroups模块"""
    results = {}
    
    # 获取所有预定义的官能团模式
    fg_list = FunctionalGroups.BuildFuncGroupHierarchy()
    
    for fg in fg_list:
        pattern = fg.pattern
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            results[fg.label] = len(matches)
    
    return results

# 使用示例
print_all_functional_groups()

mol = Chem.MolFromSmiles("CC(=O)NC1=CC=C(C=C1)O")  # 对乙酰氨基酚
fg_results = detect_with_rdkit_fg(mol)
print("RdKit内置官能团检测:")
for group, count in fg_results.items():
    print(f"{group}: {count}个")