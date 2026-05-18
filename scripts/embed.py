import argparse
import heapq
import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from atomprop.dataloader.dataloader import SMILESToInputs
from atomprop.models.GNNs import Embedder
from atomprop.models.GeAT import GeATNet
from atomprop.utils.utils import remove_module_prefix
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def load_geat_encoder(checkpoint_path, config, device):
    """
    Load only the embedding layer and GeAT backbone from a checkpoint.
    """
    embed_dim = config.embed_dim

    embedding_layer = Embedder(num_atom_types=120, embed_dim=embed_dim)
    backbone = GeATNet(
        embed_dim=embed_dim,
        num_heads=config.num_heads,
        global_num_heads=config.global_num_heads,
        output_negative_slope=config.output_negative_slope,
        dropout=config.geat_dropout,
        geat_num_layers=config.geat_num_layers,
        aggr_num_layers=config.aggr_num_layers,
        FFN_type=config.FFN_type,
        FFN_hidden_dim=config.FFN_hidden_dim,
        FFN_num_experts=config.FFN_num_experts,
        FFN_num_layers=config.FFN_num_layers,
        FFN_top_k=config.FFN_top_k,
        use_edge_embedding=config.use_edge_embedding
    )

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    embedding_layer.load_state_dict(remove_module_prefix(ckpt['embedding_layer_state_dict']))
    backbone.load_state_dict(remove_module_prefix(ckpt['backbone_state_dict']))

    embedding_layer.to(device)
    backbone.to(device)
    embedding_layer.eval()
    backbone.eval()

    return embedding_layer, backbone


def smiles_to_graph(smiles, sanitize=False):
    """Convert a single SMILES to PyG Data object."""
    atom_info, edge_info, mol = SMILESToInputs.convert(smiles, sanitize=sanitize)
    if atom_info is None or edge_info is None:
        return None

    if edge_info.dim() == 2 and edge_info.size(1) == 4:
        edge_index = edge_info[:, :2].t().contiguous()
        edge_attr = edge_info[:, 2:]
    else:
        edge_index = torch.tensor([[], []], dtype=torch.long)
        edge_attr = torch.tensor([], dtype=torch.long).view(0, 2)

    data = Data(x=atom_info, edge_index=edge_index, edge_attr=edge_attr)
    return data


@torch.no_grad()
def encode_molecule(embedding_layer, backbone, data, device):
    """Encode a single molecule graph into a fixed-size representation."""
    data = data.to(device)
    # Ensure atom types are 1D: [N]
    x_input = data.x.squeeze(-1) if data.x.dim() > 1 else data.x
    emb = embedding_layer(x_input)

    # Create batch vector: all atoms belong to graph 0
    batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

    graph_emb = backbone(
        Data(x=emb, edge_index=data.edge_index, edge_attr=data.edge_attr),
        batch=batch
    )  # [N, D]

    # Global mean pooling
    rep = torch.mean(graph_emb, dim=0, keepdim=True)  # [1, D]
    return rep.cpu().numpy()


def read_smiles_file(path):
    """Generator that yields SMILES strings from a file (one per line or CSV)."""
    if path.endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(path)
        if 'smiles' not in df.columns:
            raise ValueError("CSV must contain a 'smiles' column.")
        for smi in df['smiles'].dropna():
            yield str(smi).strip()
    else:
        with open(path, 'r') as f:
            for line in f:
                smi = line.strip()
                if smi:
                    yield smi


def main():
    parser = argparse.ArgumentParser(description="Find top-K most similar molecules using GeAT representations (memory-efficient).")
    parser.add_argument("--query_smiles", type=str, required=True, help="Input SMILES string to query.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset (CSV with 'smiles' column or plain text, one SMILES per line).")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained GeAT checkpoint (.pth).")
    parser.add_argument("--config_module", type=str, default="configs.config_finetune", help="Python config module (e.g., configs.config_finetune).")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top similar molecules to return.")
    args = parser.parse_args()

    # Dynamically import config
    import importlib
    cfg = importlib.import_module(args.config_module)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model encoder
    print("Loading GeAT encoder...")
    embedding_layer, backbone = load_geat_encoder(args.checkpoint, cfg, device)

    # Encode query molecule
    print(f"Encoding query SMILES: {args.query_smiles}")
    query_graph = smiles_to_graph(args.query_smiles)
    if query_graph is None:
        raise ValueError("Invalid query SMILES.")
    query_rep = encode_molecule(embedding_layer, backbone, query_graph, device)  # [1, D]

    # Stream through reference dataset and maintain top-K similarities
    print(f"Scanning reference molecules in '{args.dataset_path}' to find top-{args.top_k} similar...")
    
    heap = []  # min-heap of (similarity, smiles); keeps smallest at root
    total_processed = 0
    valid_processed = 0

    for smi in read_smiles_file(args.dataset_path):
        total_processed += 1
        g = smiles_to_graph(smi)
        if g is None:
            continue

        try:
            ref_rep = encode_molecule(embedding_layer, backbone, g, device)
            sim = float(cosine_similarity(query_rep, ref_rep)[0, 0])
        except Exception as e:
            print(f"Warning: Failed to encode SMILES '{smi}': {e}")
            continue

        valid_processed += 1

        # Maintain top-K using heap
        if len(heap) < args.top_k:
            heapq.heappush(heap, (sim, smi))
        else:
            if sim > heap[0][0]:
                heapq.heapreplace(heap, (sim, smi))

        # Progress indicator
        if total_processed % 10000 == 0:
            print(f"Processed {total_processed} molecules ({valid_processed} valid)...")

    # Final results
    top_results = sorted(heap, key=lambda x: x[0], reverse=True)

    print("\n" + "="*60)
    print(f"Top {len(top_results)} most similar molecules (by cosine similarity):")
    print("="*60)
    for i, (sim, smi) in enumerate(top_results, 1):
        print(f"{i}. Similarity: {sim:.4f} | SMILES: {smi}")

    print(f"\nSummary: Scanned {total_processed} lines, {valid_processed} valid molecules.")


if __name__ == "__main__":
    main()