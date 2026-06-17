import inspect
import sys
import torch

# configs/config.py

# Training
from_scratch = True
from_model_path = "trained_models/pretrain_final/model_epoch7.pth" # if not from_scratch

# Dataset and I/O
data_path = "data/zinc15/dataset/zinc_standard_agent/processed/smiles.csv"
# data_path = "data/pubchem/pubchem-10m.txt.clean"
pretrain_file_type = 'txt'
# Shuffle mode: "none" = sequential, "chunk" = shuffle chunks + within chunk (I/O efficient),
#               "full" = full random permutation (slow random disk access)
shuffle = "chunk"
random_state = 42

logdir = "pretrain"

# Model settings
geat_num_layers = 12
aggr_num_layers = 2
num_heads = 16
global_num_heads = 16
output_negative_slope = 0.2
geat_dropout = 0.1
head_dropout = 0.3

FFN_type = "MLP"
FFN_num_layers = 2
FFN_hidden_dim = 4096
FFN_num_experts = 8
FFN_top_k = 2
use_edge_embedding = False

# Per-layer FFN settings (new Transformer block)
per_layer_FFN_type = "MLP"
per_layer_FFN_num_layers = 2
per_layer_FFN_hidden_dim = 4096
per_layer_FFN_dropout = 0.1
per_layer_FFN_num_experts = 8
per_layer_FFN_top_k = 2

# Low-rank bilinear attention (64 = ~8x parameter reduction vs full)
attention_rank = 64

# Data settings
dataset_size = -1
chunk_size = 65536
max_atom_num = 128
batch_size = 128

# Weight settings
fix_uncertainty = False
fixed_log_vars = torch.tensor([-3.2157, 0.5324, -0.4459, -0.6287, -1.8934, 2.7381], dtype=torch.float32)

# Training settings
num_epochs = 20
weight_type = "UW"
record_freq = 100
skip_batch = 0  # skip the first N batches (set >0 to resume after OOM crash)

# Masking rates
less_rate = 0.1
more_rate = 0.3

# Model dimensions
embed_dim = 1024

# Functional groups
fg_list = None  # if None, use default RDKit functional groups

# Optimizer & Scheduler base settings
embedding_layer_lr = 1e-4  # embedding layer
backbone_lr = 1e-4  # geat
head_lr = 5e-4  # classification head
weight_strategy_lr = 1e-3

# Layer-wise learning rate decay
use_layer_decay = True  # Set to False to disable
layer_decay_rate = 0.9  # Each layer has this rate of LR of previous layer

embedding_layer_wd = 5e-5  # embedding layer
backbone_wd = 5e-5  # geat
head_wd = 1e-5  # classification head
weight_strategy_wd = 0.0

# Scheduler settings
embedding_layer_eta_min = 1e-5
backbone_eta_min = 1e-5
head_eta_min = 1e-5
weight_strategy_eta_min = 1e-5

weight_strategy_pct_start = 0.05
weight_strategy_div_factor = 10.0

def print_all_params():
    """Print all configuration parameters defined in this module."""
    # Get all variables in the current module (globals)
    current_module = sys.modules[__name__]
    attrs = {
        name: value
        for name, value in inspect.getmembers(current_module)
        if not name.startswith("_")  # skip private/dunder names
        and not inspect.isfunction(value)  # skip functions (including this one)
        and not inspect.ismodule(value)   # skip imported modules
    }

    print("=== Configuration Parameters ===")
    for key, val in sorted(attrs.items()):
        print(f"{key} = {repr(val)}")
    print("================================")


# Auto-load local overrides (config_local.py) if it exists
import os as _os
_local_file = __file__.replace('.py', '_local.py')
if _os.path.exists(_local_file):
    import importlib.util as _importlib_util
    _spec = _importlib_util.spec_from_file_location("_config_local", _local_file)
    _local_mod = _importlib_util.module_from_spec(_spec)
    _spec.loader.exec_module(_local_mod)
    _current_mod = sys.modules[__name__]
    for _attr in dir(_local_mod):
        if not _attr.startswith('_'):
            setattr(_current_mod, _attr, getattr(_local_mod, _attr))
    print(f"[CONFIG] Loaded local overrides from {_local_file}")