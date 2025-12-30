import inspect
import sys

# config_finetune.py

# Device
device_str = "cuda:5"

# Model settings
geat_num_layers = 6
aggr_num_layers = 1

# Data settings
data_path = "./data/moleculenet/tox21/tox21.csv"
x_col = "smiles"
exclude_list = ["mol_id", "name", "num"]

# Training settings
no_pretrain = True
pretrained_path = 'trained_models/pretrain_pubchem_geat_continue/model_epoch4.pth'
logdir = "finetune_geat_toxcast_52"

batch_size = 32
test_batch_size = 32
num_epochs = 200
random_state = 42

# Cross-validation
k_folds = 5
frac_test = 0.1

# Model architecture
embed_dim = 512
aggr = 'mean'  # options: 'mean', 'sum', 'max', 'attention'

# Optimizer settings
lr_backbone_neck = 5e-6
lr_head = 2e-4
lr_aggr = 2e-4

# Scheduler settings
T_max = num_epochs
eta_min_backbone_neck = 1e-7
eta_min_head = 2e-6
eta_min_aggr = 2e-6

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
        
config_dict = {
    "tox21": "./data/moleculenet/tox21/tox21.csv",
    "toxcast": "./data/moleculenet/toxcast/toxcast_data.csv",
    "sider": "./data/moleculenet/sider/sider.csv",
    "clintox": "./data/moleculenet/clintox/clintox.csv",
    "bbbp": "./data/moleculenet/bbbp/BBBP.csv"
}

def set_data_path(dataset_name):
    global data_path, logdir
    data_path = config_dict[dataset_name]
    logdir += '_'
    logdir += dataset_name
    