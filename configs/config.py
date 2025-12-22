import inspect
import sys
import torch

# configs/config.py

# Training
from_scratch = False
from_model_path = "trained_models/pretrain_pubchem_geat/model_epoch7.pth"

# Dataset and I/O
data_path = "data/zinc15/dataset/zinc_standard_agent/processed/smiles.csv"
# data_path = "data/pubchem/pubchem-10m.txt"
pretrain_file_type = 'txt'

logdir = "pretrain_pubchem_geat_continue"

# Data settings
dataset_size = -1
chunk_size = 65536
max_atom_num = 128
batch_size = 224

# Weight settings
fixed_log_vars = torch.tensor([-3.0956, -10.0055, -0.0008, 0.875, -1.7909, 2.7035], dtype=torch.float32)

# Training settings
num_epochs = 8
record_freq = 100

# Masking rates
less_rate = 0.1
more_rate = 0.3

# Model dimensions
embed_dim = 384

# Functional groups
fg_list = None  # if None, use default RDKit functional groups

# Device (will be overridden in main script based on availability)
device_str = "cuda:6"  # fallback to cpu if not available

# Optimizer & Scheduler base settings
backbone_lr = 5e-4
neck_lr = 5e-4
head_lr = 5e-4
weight_strategy_lr = 1e-3

backbone_wd = 5e-5
neck_wd = 5e-5
head_wd = 1e-5
weight_strategy_wd = 0.0

neck_scheduler_max_lr = 1e-3  # special for neck

# OneCycleLR settings
pct_start = 0.1
anneal_strategy = "cos"
div_factor = 25.0
final_div_factor = 1e4

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