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