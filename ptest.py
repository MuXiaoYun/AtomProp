from atomprop.tasks.tasks import NodeAttrPrediction
from atomprop.dataloader.dataloader import SMILESToInputs
from atomprop.models.GNNs import Embedder, GNN
from atomprop.utils.mlp import MLP
from atomprop.utils.mask import MolGraphMask
from atomprop.embeddings.AtomEmbedding import BondTypes
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch_geometric.data import Data, Batch
from torch.nn import DataParallel
from torch_geometric.nn import DataParallel as GeoDataParallel

hap = [[1,2], [3,4]]
data = Data(x=hap, edge_index=[[0,1]])
hap[0] = [5, 6]
print(data.x)
