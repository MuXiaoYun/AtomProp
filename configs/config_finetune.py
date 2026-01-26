import inspect
import sys

# config_finetune.py
device = "cuda:0"
num_runs = 10

# Model settings
geat_num_layers = 4
aggr_num_layers = 2
num_heads = 8
global_num_heads = 8
output_negative_slope = 0.2
geat_dropout = 0.2
head_dropout = 0.3

downstream_head_attn_num_layers = 2

FFN_type = "MLP"
FFN_num_layers = 2
FFN_hidden_dim = 1024
FFN_num_experts = 8
FFN_top_k = 2
use_edge_embedding = False

# Data settings
data_path = "./data/moleculenet/tox21/tox21.csv"
x_col = "smiles"
exclude_list = ["mol_id", "name", "num", "iupac", "Compound ID", "CMPD_CHEMBLID"] # columns that are not x or y

# Training settings
no_pretrain = False
pretrained_path = 'trained_models/model_epoch15.pth'
logdir = "finetune_weights"
logdir += "_nopre" if no_pretrain else "_pre"

batch_size = 128
test_batch_size = 128
num_epochs = 100
tolerance = 20
random_state = 0

# Cross-validation
gamma = 1.0

# Model architecture
embed_dim = 512
aggr = 'mean'  # options: 'mean', 'sum', 'max', 'attention'
head_layers = 2
head_hidden_dim = 1024

# Optimizer settings
lr_embedding_layer_backbone = 1e-5
lr_head = 1e-3

# Scheduler settings
T_max = 30
freeze = 20  # number of epochs to freeze embedding layer and backbone
eta_min_embedding_layer_backbone = 5e-6
eta_min_head = 5e-4

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
    ## classification tasks
    "tox21": "./data/moleculenet/tox21/tox21.csv",
    "toxcast": "./data/moleculenet/toxcast/toxcast_data.csv",
    "sider": "./data/moleculenet/sider/sider.csv",
    "clintox": "./data/moleculenet/clintox/clintox.csv",
    "bbbp": "./data/moleculenet/bbbp/BBBP.csv",
    ## regression tasks
    "freesolv": "./data/moleculenet/freesolv/SAMPL.csv",
    "esol": "./data/moleculenet/esol/delaney-processed.csv",
    "lipo": "./data/moleculenet/lipo/Lipophilicity.csv",
}

def set_data_path(dataset_name):
    global data_path, logdir
    data_path = config_dict[dataset_name]
    logdir += '_'
    logdir += dataset_name
    