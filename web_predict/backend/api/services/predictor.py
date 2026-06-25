"""SMILES property prediction service (shared logic from predict_gui.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

# AtomProp project root (web_predict/backend -> web_predict -> AtomProp)
ATOMPROP_ROOT = Path(__file__).resolve().parents[4]
if str(ATOMPROP_ROOT) not in sys.path:
    sys.path.insert(0, str(ATOMPROP_ROOT))

import configs.config_reg as cfg  # noqa: E402
from atomprop.dataloader.dataloader import SMILESToInputs  # noqa: E402
from atomprop.models.gnns import Embedder, GNNAggr  # noqa: E402
from atomprop.models.geat import GeATNet  # noqa: E402
from atomprop.utils.mlp import MLP  # noqa: E402


def get_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _build_base_model(device: torch.device):
    """Build the base GeATNet model (without head) and load pretrained weights from config."""
    embedding_layer = Embedder(num_atom_types=120, embed_dim=cfg.embed_dim).to(device)
    backbone = GeATNet(
        embed_dim=cfg.embed_dim,
        num_heads=cfg.num_heads,
        global_num_heads=cfg.global_num_heads,
        output_negative_slope=cfg.output_negative_slope,
        dropout=cfg.geat_dropout,
        geat_num_layers=cfg.geat_num_layers,
        aggr_num_layers=cfg.aggr_num_layers,
        FFN_type=cfg.FFN_type,
        FFN_hidden_dim=cfg.FFN_hidden_dim,
        FFN_num_experts=cfg.FFN_num_experts,
        FFN_num_layers=cfg.FFN_num_layers,
        FFN_top_k=cfg.FFN_top_k,
        use_edge_embedding=cfg.use_edge_embedding,
    ).to(device)
    aggrmodel = GNNAggr(embed_dim=cfg.embed_dim, aggr=cfg.aggr, layers=1).to(device)

    # Load base pretrained weights from config
    base_path = Path(ATOMPROP_ROOT) / cfg.pretrained_path
    if base_path.is_file():
        base_ckpt = torch.load(base_path, map_location=device, weights_only=False)
        from atomprop.utils.utils import remove_module_prefix
        embedding_layer.load_state_dict(
            remove_module_prefix(base_ckpt["embedding_layer_state_dict"]))
        backbone.load_state_dict(
            remove_module_prefix(base_ckpt["backbone_state_dict"]))
    else:
        print(f"[Predictor] Base model not found at {base_path}, using random init.")

    return embedding_layer, backbone, aggrmodel


def predict_smiles(smiles_list: list[str], model_path: str | Path) -> list[dict]:
    """
    Run batch prediction on a list of SMILES strings.

    Accepts both full-model checkpoints and LoRA-only checkpoints.
    LoRA checkpoints must contain ``lora_state_dict`` and ``head_state_dict``;
    the base backbone + embedder are loaded from the config's pretrained_path.

    Returns list of dicts: {"smiles": str, "predicted_value": float}
    Invalid SMILES are skipped.
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    cleaned = [s.strip() for s in smiles_list if s and str(s).strip()]
    if not cleaned:
        return []

    device = get_device()
    embedding_layer, backbone, aggrmodel = _build_base_model(device)

    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    # Detect LoRA vs full checkpoint
    is_lora = "lora_state_dict" in ckpt

    if is_lora:
        # LoRA checkpoint: apply LoRA to backbone, load adapter weights, then merge
        lora_cfg = ckpt.get("lora_config", {})
        backbone.apply_lora(
            rank=lora_cfg.get("rank", 8),
            alpha=lora_cfg.get("alpha", 8.0),
            dropout=lora_cfg.get("dropout", 0.0),
            include_ffn=lora_cfg.get("include_ffn", False),
            include_global_attn=lora_cfg.get("include_global_attn", False),
        )
        backbone.load_lora_state_dict(ckpt["lora_state_dict"])
        backbone.merge_lora()
    else:
        # Full checkpoint: load backbone state dict directly
        backbone.load_state_dict(ckpt["backbone_state_dict"])
        # Also load embedding if present (full checkpoints include it)
        if "embedding_layer_state_dict" in ckpt:
            embedding_layer.load_state_dict(ckpt["embedding_layer_state_dict"])

    # Head: always create fresh from task output dim, load from checkpoint
    output_dim = ckpt["head_state_dict"]["layers.%d.weight" % (cfg.head_layers - 1)].shape[0]
    head = MLP(
        input_dim=cfg.embed_dim,
        hidden_dim=cfg.head_hidden_dim,
        output_dim=output_dim,
        num_layers=cfg.head_layers,
        dropout=cfg.head_dropout,
        batch_norm=True,
        output_activation=None,
    ).to(device)
    head.load_state_dict(ckpt["head_state_dict"])

    if cfg.aggr == "attention" and ckpt.get("aggr_state_dict"):
        aggrmodel.load_state_dict(ckpt["aggr_state_dict"])

    scaler_stats = ckpt.get("scaler_stats")
    embedding_layer.eval()
    backbone.eval()
    head.eval()
    aggrmodel.eval()

    dataset = []
    valid_smiles = []
    for smi in cleaned:
        atom_info, edge_info, _ = SMILESToInputs.convert(smi, sanitize=False)
        if atom_info is None or edge_info is None:
            continue
        if edge_info.dim() == 2 and edge_info.size(1) == 4:
            edge_index = edge_info[:, :2].t().contiguous()
            edge_attr = edge_info[:, 2:]
        else:
            edge_index = torch.tensor([[], []], dtype=torch.long)
            edge_attr = torch.tensor([], dtype=torch.long).view(0, 2)
        dataset.append(Data(x=atom_info, edge_index=edge_index, edge_attr=edge_attr))
        valid_smiles.append(smi)

    if not dataset:
        return []

    loader = DataLoader(
        dataset, batch_size=8, shuffle=False, collate_fn=Batch.from_data_list
    )
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            emb = embedding_layer(batch.x.squeeze())
            emb = backbone(
                Data(x=emb, edge_index=batch.edge_index, edge_attr=batch.edge_attr),
                batch=batch.batch,
            )
            g_emb = aggrmodel(emb, batch.batch)
            out = head(g_emb).cpu().numpy().flatten()
            preds.extend(out)

    if scaler_stats is not None:
        preds = np.array(preds) * scaler_stats["scale"][0] + scaler_stats["mean"][0]

    return [
        {"smiles": smi, "predicted_value": float(val)}
        for smi, val in zip(valid_smiles, preds)
    ]
