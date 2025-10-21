from atomprop.models.GeAT import GeATNet
import configs.config as config
import rdkit.Chem as Chem
import torch

# test_split_big_graph.py
"""
完整测试：8 张小图 → 合并为一张大图（Data 形式）→ 还原回 8 张小图
前置代码不允许改动，只能在后面追加。
"""

from torch_geometric.data import Data, Batch
import torch

def big_data_to_data_list(data, batch):
    """
    Convert batched graph data back to list of individual graphs.
    
    Args:
        data: Data object containing batched node features and edge indices
        batch: Batch vector indicating which graph each node belongs to
    
    Returns:
        List[Data]: List of individual graph Data objects
    """
    # Calculate number of nodes per graph
    node_counts = torch.bincount(batch)
    # Calculate node offsets for each graph
    node_offsets = torch.cat([torch.tensor([0], device=batch.device), 
                             torch.cumsum(node_counts, 0)[:-1]])
    
    # Precompute which graph each edge belongs to
    edge_batch = batch[data.edge_index[0]]
    
    small_graphs = []
    num_graphs = len(node_counts)
    
    for i in range(num_graphs):
        start_idx = node_offsets[i]
        end_idx = start_idx + node_counts[i]
        
        # Extract node features for current graph
        x = data.x[start_idx:end_idx]
        
        # Extract edges for current graph and adjust indices
        edge_mask = edge_batch == i
        edge_index = data.edge_index[:, edge_mask] - start_idx
        
        small_graphs.append(Data(x=x, edge_index=edge_index))
    
    return small_graphs

# ===== Generate 8 different graphs with varying structures =====
torch.manual_seed(42)  # For reproducibility

small_graphs = []
node_configs = [3, 4, 5, 4, 6, 3, 5, 4]  # Different number of nodes for each graph
edge_configs = [4, 6, 8, 5, 10, 3, 7, 6]  # Different number of edges for each graph

print("=== Original Graphs ===")
for i in range(8):
    num_nodes = node_configs[i]
    num_edges = edge_configs[i]
    
    # Generate random node features with different dimensions
    x = torch.randn(num_nodes, 2) + i * 0.1  # Add offset to make features different
    
    # Generate random edges (avoid self-loops for simplicity)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    graph = Data(x=x, edge_index=edge_index)
    small_graphs.append(graph)
    
    print(f"Graph {i}: nodes={num_nodes}, edges={num_edges}, "
          f"x_shape={x.shape}, edge_shape={edge_index.shape}")

# ===== Batch into one big graph =====
print("\n=== Batched Graph ===")
big_batch = Batch.from_data_list(small_graphs)
print(f"Batched graph: {big_batch}")

# Extract components (simulating the given scenario)
big_data = Data(x=big_batch.x, edge_index=big_batch.edge_index)
batch_vector = big_batch.batch
big_batch = None
small_graphs = None

# ===== Restore using our function =====
print("\n=== Restored Graphs ===")
restored_graphs = big_data_to_data_list(big_data, batch_vector)

# Verify correctness
all_correct = True
for i, (original_config, restored_graph) in enumerate(zip(zip(node_configs, edge_configs), restored_graphs)):
    expected_nodes, expected_edges = original_config
    actual_nodes = restored_graph.num_nodes
    actual_edges = restored_graph.num_edges
    
    nodes_match = expected_nodes == actual_nodes
    edges_match = expected_edges == actual_edges
    
    print(f"Graph {i}: nodes({expected_nodes}→{actual_nodes}) {'✓' if nodes_match else '✗'}, "
          f"edges({expected_edges}→{actual_edges}) {'✓' if edges_match else '✗'}")
    
    if not (nodes_match and edges_match):
        all_correct = False

print(f"\n=== Overall Result: {'SUCCESS' if all_correct else 'FAILED'} ===")

# Additional verification: check if we can batch the restored graphs again
print("\n=== Additional Verification ===")
re_batched = Batch.from_data_list(restored_graphs)
print(f"Re-batched graph matches original structure: "
      f"nodes={re_batched.num_nodes == big_data.num_nodes}, "
      f"edges={re_batched.num_edges == big_data.num_edges}")

# data_path = "data/moleculenet/qm9/gdb9.sdf"
# check_data = False
# target_label = "DFT TOTAL ENERGY"

# if __name__ == "__main__":
#     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # print(f"Using device: {device}")

#     # Use the default configurations
#     geatnet = GeATNet(atom_embedding_dim=config.atom_embedding_dim,
#                      num_atom_types=config.num_atom_types,
#                      num_bond_types=config.num_bond_types,
#                      num_heads=config.num_heads,
#                      global_num_heads=config.global_num_heads,
#                      backbone_dropout=config.backbone_dropout,
#                      neck_dropout=config.neck_dropout,
#                      head_dropout=config.head_dropout,
#                      mlp_hidden_dim=config.geatnet_hidden_dim,
#                      output_negative_slope=config.edge_attetion_output_negative_slope,
#                      parallel_between_bondtypes=config.parallel_between_bondtypes,
#                      )
#     print(geatnet)
#     # Print param numbers
#     total_params = sum(p.numel() for p in geatnet.parameters() if p.requires_grad)
#     print(f"Total trainable parameters: {total_params}")
#     # Print param numbers of each part
#     # Backbone
#     backbone_params = sum(p.numel() for p in geatnet.backbone.parameters() if p.requires_grad)
#     print(f"Backbone trainable parameters: {backbone_params}")
#     # Neck
#     neck_params = sum(p.numel() for p in geatnet.neck.parameters() if p.requires_grad)
#     print(f"Neck trainable parameters: {neck_params}")
#     # Head
#     head_params = sum(p.numel() for p in geatnet.head.parameters() if p.requires_grad)
#     print(f"Head trainable parameters: {head_params}")

#     if not check_data:
#         exit(0)
#     # Judge file type
#     if data_path.endswith(".csv"):
#         # Load data from the CSV file
#         # smiles for input and the rest for regression targets
        
#         import pandas as pd
#         df = pd.read_csv(data_path)
#         print(df.head())
#         # Print the columns
#         print("Columns in the dataset:", df.columns.tolist())
#         # Use rdkit.chem to handle SMILES strings. Count how many kinds of atom types and bond types are in the dataset
#         smiles_list = df['SMILES'].tolist()
#         target_label_list = df[target_label].tolist()
#         print("Number of SMILES strings:", len(smiles_list))

#         atom_types = set()
#         bond_types = set()

#         max_num = -1

#         for i, smiles in enumerate(smiles_list):
#             if i % 100 == 0:
#                 print(f"Processing {i}th SMILES: {smiles}")
#             mol = Chem.MolFromSmiles(smiles)
#             if mol is not None:
#                 for atom in mol.GetAtoms():
#                     atom_types.add(atom.GetSymbol())
#                 for bond in mol.GetBonds():
#                     bond_types.add(bond.GetBondTypeAsDouble())
#                 atom_num = mol.GetNumAtoms()
#                 if atom_num > max_num:
#                     max_num = atom_num
#             else:
#                 print(f"Invalid SMILES string at index {i}: {smiles}")

#         max_label = max(target_label_list)
#         min_label = min(target_label_list)
        
#         print("Number of atom types:", len(atom_types))
#         print("Atom types:", atom_types)
#         print("Number of bond types:", len(bond_types))
#         print("Bond types:", bond_types)
#         print("Maximum number of atoms in a molecule:", max_num)
#         print(f"Max {target_label}: {max_label}, Min {target_label}: {min_label}")

#     elif data_path.endswith(".sdf"):
#         # Load data from the SDF file
#         from atomprop.dataloader.dataloader import SDFToInputs
#         sdf_path = data_path
#         results = SDFToInputs.convert(sdf_path, context_length=config.context_length)
#         print(f"Number of molecules in the SDF file: {len(results)}")

#         atom_types = set()
#         bond_types = set()

#         max_num = -1

#         for i, (atom_type_indices, adj_matrix, mol) in enumerate(results):
#             if i % 100 == 0:
#                 print(f"Processing {i}th molecule")
#             if mol is not None:
#                 for atom in mol.GetAtoms():
#                     atom_types.add(atom.GetSymbol())
#                 for bond in mol.GetBonds():
#                     bond_types.add(bond.GetBondTypeAsDouble())
#                 atom_num = mol.GetNumAtoms()
#                 if atom_num > max_num:
#                     max_num = atom_num
#             else:
#                 print(f"Invalid molecule at index {i}")
        
#         print("Number of atom types:", len(atom_types))
#         print("Atom types:", atom_types)
#         print("Number of bond types:", len(bond_types))
#         print("Bond types:", bond_types)
#         print("Maximum number of atoms in a molecule:", max_num)
    