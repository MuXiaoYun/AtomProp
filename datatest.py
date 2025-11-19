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
mol = Chem.MolFromSmiles(smiles)
# print num of atoms
print("Number of atoms:", mol.GetNumAtoms())

try:
    mol = Chem.RemoveHs(mol)
except:
    pass

print("Molecule after removing Hs has", mol.GetNumAtoms(), "atoms.")

try:
    mol = Chem.RemoveHs(mol)
except:
    pass

print("Molecule after removing Hs has", mol.GetNumAtoms(), "atoms.")

try:
    mol = Chem.RemoveHs(mol)
except:
    pass

print("Molecule after removing Hs has", mol.GetNumAtoms(), "atoms.")

try:
    mol = Chem.RemoveHs(mol)
except:
    pass

print("Molecule after removing Hs has", mol.GetNumAtoms(), "atoms.")

try:
    mol = Chem.RemoveHs(mol)
except:
    pass

print("Molecule after removing Hs has", mol.GetNumAtoms(), "atoms.")

try:
    mol = Chem.RemoveHs(mol)
except:
    pass

print("Molecule after removing Hs has", mol.GetNumAtoms(), "atoms.")