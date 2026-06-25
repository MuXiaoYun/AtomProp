import inspect
import sys

# config_finetune.py
device = "cuda:0"
num_runs = 3

# Model settings
geat_num_layers = 12
aggr_num_layers = 2
num_heads = 16
global_num_heads = 16
output_negative_slope = 0.2
geat_dropout = 0.2

FFN_type = "MLP"
FFN_num_layers = 2
FFN_hidden_dim = 4096
FFN_num_experts = 8
FFN_top_k = 2
use_edge_embedding = False

# Per-layer FFN settings
per_layer_FFN_type = "MLP"
per_layer_FFN_num_layers = 2
per_layer_FFN_hidden_dim = 4096
per_layer_FFN_dropout = 0.1
per_layer_FFN_num_experts = 8
per_layer_FFN_top_k = 2
attention_rank = 64

# Data settings
data_path = "./data/moleculenet/tox21/tox21.csv"
x_col = "smiles"
exclude_list = ["mol_id", "name", "num", "iupac", "Compound ID", "CMPD_CHEMBLID", "CID", "activity"] # columns that are not x or y

# Training settings
no_pretrain = True
pretrained_path = 'trained_models/pretrain_final_true/model_epoch15.pth'
logdir = "finetune_0127"
logdir += "_nopre" if no_pretrain else "_pre"

batch_size = 128
test_batch_size = 128
num_epochs = 100
tolerance = 10
random_state = 0

# Cross-validation
gamma = 1.0

# Model architecture
embed_dim = 1024
aggr = 'mean'  # options: 'mean', 'sum', 'max', 'attention'
# Head settings — always use simple MLP (DownstreamHead is deprecated)
head_type = "mlp"
head_dropout = 0.3
head_layers = 2
head_hidden_dim = 2048

# ---- LoRA (Low-Rank Adaptation) configuration ----
use_lora = True             # Use LoRA for efficient fine-tuning
lora_rank = 8                 # LoRA rank (typical: 4-64)
lora_alpha = 8.0              # LoRA scaling factor
lora_dropout = 0.0            # Dropout on LoRA branch input
lora_include_ffn = False      # Also adapt per-layer + final FFN weights
lora_include_global_attn = False  # Also adapt neck global attention

# Optimizer settings
lr_embedding_layer_backbone = 1e-5
wd_emb_backbone = 1e-4
lr_head = 1e-4
wd_head = 1e-4

# Scheduler settings
T_max = 100
freeze = 0  # number of epochs to freeze embedding layer and backbone
eta_min_embedding_layer_backbone = 5e-6
eta_min_head = 5e-5

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


# Auto-load local overrides (config_finetune_local.py) if it exists
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

config_dict = {
    "tox21": "./data/moleculenet/tox21/tox21.csv",
    "toxcast": "./data/moleculenet/toxcast/toxcast_data.csv",
    "sider": "./data/moleculenet/sider/sider.csv",
    "clintox": "./data/moleculenet/clintox/clintox.csv",
    "bbbp": "./data/moleculenet/bbbp/BBBP.csv",
    "bace": "./data/moleculenet/bace/bace.csv",
    "hiv": "./data/moleculenet/hiv/HIV.csv",
    "muv": "./data/moleculenet/muv/muv.csv",
}

def set_data_path(dataset_name):
    global data_path, logdir
    data_path = config_dict[dataset_name]
    logdir += '_'
    logdir += dataset_name
    