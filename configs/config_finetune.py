import inspect
import sys

# config_finetune.py

# Device
device_str = "cuda:6"

# Model settings
geat_num_layers = 4
aggr_num_layers = 2
num_heads = 8
global_num_heads = 8
output_negative_slope = 0.2
geat_dropout = 0.1
FFN_type = "MOE"
FFN_num_layers = 2
FFN_hidden_dim = 1024
FFN_num_experts = 8
FFN_top_k = 2
use_edge_embedding = False

# Regularization settings
norm_lambda = 0

# Data settings
data_path = "./data/moleculenet/tox21/tox21.csv"
x_col = "smiles"
exclude_list = ["mol_id", "name", "num"]

# Training settings
no_pretrain = True
pretrained_path = 'trained_models/pretrain_zinc_geat_moe/model_epoch3.pth'
logdir = "finetune_MOE82"

batch_size = 128
test_batch_size = 128
num_epochs = 200
random_state = 42
head_dropout = 0.1

# Cross-validation
k_folds = 5
frac_test = 0.1

# Model architecture
embed_dim = 512
aggr = 'attention'  # options: 'mean', 'sum', 'max', 'attention'
head_hidden_dim = 512

# Optimizer settings
lr_embedding_layer_backbone = 2e-5
lr_head = 1e-3
lr_aggr = 1e-3

# Scheduler settings
T_max = num_epochs
eta_min_embedding_layer_backbone = 5e-7
eta_min_head = 1e-5
eta_min_aggr = 1e-5

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
    