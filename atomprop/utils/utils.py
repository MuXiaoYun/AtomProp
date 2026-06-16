"""
Useful tools.
"""


def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from keys in state_dict (used when saving DDP model)."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  # remove 'module.'
        else:
            new_state_dict[k] = v
    return new_state_dict


def convert_old_geat_checkpoint(old_backbone_state, new_num_layers, old_num_layers=5):
    """
    Convert a state_dict from the old GeAT architecture (Post-LN, no per-layer FFN,
    separate norm_layers) to the new architecture (Pre-LN, per-layer FFN, norms inside layer).

    Mappings:
        - Q_w/K_w/V_w, project, edge_attention params: direct copy
        - norm_after_attn -> norm1
        - norm_layers (in GeATConv) -> removed (norms now inside layers)
        - norm2 (new) and per-layer FFN (new): not present in old checkpoint

    Returns:
        new_state_dict: dict with mapped keys
    """
    new_dict = {}

    # Directly mappable keys from old structure
    DIRECT_KEYS = [
        'edge_type_embedding.weight',
        'edge_direction_embedding.weight',
        'ffn.layers',
        'FFN_norm.weight',
        'FFN_norm.bias',
    ]

    # Global attention module keys
    GLOBAL_ATTN_PREFIXES = [
        'neck.global_attns.',
        'neck.norm_layers.',
    ]

    for old_key, value in old_backbone_state.items():
        # ----- Map GeAT layers -----
        # Old: backbone.geat_layers.{i}.xxx
        # New: backbone.geat_layers.{i}.xxx (same for Q/K/V/project/edge_attention)
        # Old: backbone.geat_layers.{i}.norm_after_attn.xxx -> norm1.xxx
        # Old: backbone.norm_layers.{i}.xxx -> REMOVED (now inside layer)

        if old_key.startswith('backbone.geat_layers.'):
            parts = old_key.split('.')
            layer_idx = int(parts[2])
            if layer_idx >= new_num_layers:
                continue  # skip layers beyond new model depth

            suffix = '.'.join(parts[3:])

            if suffix.startswith('norm_after_attn'):
                # Map old single norm to new norm1
                new_suffix = suffix.replace('norm_after_attn', 'norm1')
                new_key = f'backbone.geat_layers.{layer_idx}.{new_suffix}'
                new_dict[new_key] = value
            elif not suffix.startswith('norm1.') and not suffix.startswith('norm2.') and not suffix.startswith('ffn.'):
                # Directly mappable: Q_w, K_w, V_w, project, edge_attention.*
                new_dict[old_key] = value

        # ----- Skip old norm_layers -----
        elif old_key.startswith('backbone.norm_layers.'):
            continue  # norms are now inside each GeATLayer

        # ----- Direct mappable (edge embeddings, global FFN) -----
        elif any(old_key.startswith(k) for k in DIRECT_KEYS):
            new_dict[old_key] = value

        # ----- Global attention layers (unchanged structure) -----
        elif any(old_key.startswith(p) for p in GLOBAL_ATTN_PREFIXES):
            new_dict[old_key] = value

        # Catch any remaining keys
        else:
            # Only keep if it matches the new model's expected key pattern
            new_dict[old_key] = value

    return new_dict
