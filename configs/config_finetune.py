# config_finetune.py

# Data settings
data_path = "./data/moleculenet/tox21/tox21.csv"
x_col = "smiles"
exclude_list = ["mol_id", "name", "num"]

# Training settings
no_pretrain = True
pretrained_path = 'trained_models/pretrain_pubchem_geat/model_epoch2.pth'
logdir = "finetune_geat_tox21"

batch_size = 32
test_batch_size = 32
num_epochs = 100
random_state = 42

# Cross-validation
k_folds = 3
frac_test = 0.1

# Model architecture
embed_dim = 384
aggr = 'attention'  # options: 'mean', 'sum', 'max', 'attention'

# Optimizer settings
lr_backbone_neck = 5e-6
lr_head = 2e-4
lr_aggr = 2e-4

# Scheduler settings
T_max = num_epochs
eta_min_backbone_neck = 1e-7
eta_min_head = 2e-6
eta_min_aggr = 2e-6

# Device
device_id = 5  # use cuda:5; set to None or negative to auto-select or use CPU