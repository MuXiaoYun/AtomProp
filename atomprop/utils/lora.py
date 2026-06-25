"""
LoRA (Low-Rank Adaptation) module for efficient fine-tuning of GeATNet.

Provides LoraLinear wrapper, injection/merge utilities, and state dict helpers.
Designed to work with any nn.Linear layer in the model.
"""

from __future__ import annotations

import math
from typing import Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoraLinear(nn.Module):
    """
    Wraps a frozen nn.Linear with trainable low-rank adapters A and B.

    Original weight W_0 (frozen):  shape [out_features, in_features]
    Adapter A:                      shape [rank, in_features]
    Adapter B:                      shape [out_features, rank]

    Forward:  y = W_0 @ x + (alpha / rank) * B @ A @ x

    A is Kaiming-uniform initialized; B is zero-initialized so the adapter
    starts as an identity transform (no effect on the pretrained output).

    merge() bakes the adapter into W_0 for zero-overhead inference.
    unmerge() restores the original W_0 for continued training.
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 1.0

        # Freeze the original weight & bias
        self.register_buffer("original_weight", original.weight.data.clone())
        if original.bias is not None:
            self.register_buffer("original_bias", original.bias.data.clone())
        else:
            self.original_bias = None
        original.weight.requires_grad_(False)
        if original.bias is not None:
            original.bias.requires_grad_(False)

        # Keep a reference to the original layer for merge/unmerge
        self.original = original

        out_features, in_features = original.weight.shape
        actual_rank = min(rank, in_features, out_features)
        if actual_rank != rank:
            print(
                f"[LoraLinear] Rank reduced from {rank} to {actual_rank} "
                f"(in={in_features}, out={out_features})"
            )
            self.rank = actual_rank
            self.scaling = alpha / actual_rank if actual_rank > 0 else 1.0

        # A: down-projection  (rank, in_features)
        self.lora_A = nn.Parameter(torch.zeros(actual_rank, in_features))
        # B: up-projection    (out_features, rank)
        self.lora_B = nn.Parameter(torch.zeros(out_features, actual_rank))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._merged = False

        self._reset_parameters()

    def _reset_parameters(self):
        """Kaiming-uniform init for A, zero init for B (adapter starts as identity)."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._merged:
            # Merged mode: original weight already contains adapter
            return F.linear(x, self.original.weight, self.original.bias)

        # Frozen linear + LoRA branch
        base = F.linear(x, self.original.weight, self.original.bias)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base + lora_out * self.scaling

    def merge(self):
        """Bake LoRA weights into the original weight. Idempotent."""
        if self._merged:
            return
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.original.weight.data = self.original_weight + delta
        self._merged = True

    def unmerge(self):
        """Restore the original (frozen) weight. Idempotent."""
        if not self._merged:
            return
        self.original.weight.data = self.original_weight.clone()
        self._merged = False

    @property
    def merged(self) -> bool:
        return self._merged

    def extra_repr(self) -> str:
        return (
            f"in={self.original.in_features}, out={self.original.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, merged={self._merged}"
        )


# ---------------------------------------------------------------------------
# Injection / extraction utilities
# ---------------------------------------------------------------------------

def inject_lora(
    module: nn.Module,
    target_names: Set[str],
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
    include_ffn: bool = False,
    include_global_attn: bool = False,
) -> nn.Module:
    """
    Recursively replace nn.Linear children (whose attribute name is in
    *target_names*) with LoraLinear wrappers.  Modifies the module in-place.

    Args:
        module: The root module to inject into.
        target_names: Attribute names to replace (e.g. {"Q_w","K_w","V_w","project"}).
        rank: LoRA rank.
        alpha: LoRA scaling factor (effective scale = alpha / rank).
        dropout: Dropout applied to the LoRA branch input.
        include_ffn: If True, also walk into submodules named 'ffn' (MLP/MoE)
            and wrap every nn.Linear inside them.
        include_global_attn: If True, walk into 'global_attention' submodules
            (nn.MultiheadAttention) and wrap their out_proj Linear layer.

    Returns:
        The module (same object, modified in-place).
    """
    _inject_impl(module, target_names, rank, alpha, dropout,
                 include_ffn, include_global_attn)
    return module


def _inject_impl(
    module: nn.Module,
    target_names: Set[str],
    rank: int,
    alpha: float,
    dropout: float,
    include_ffn: bool,
    include_global_attn: bool,
):
    """Recursive implementation of inject_lora."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in target_names:
            setattr(module, name, LoraLinear(child, rank=rank, alpha=alpha, dropout=dropout))
        elif include_ffn and name == "ffn":
            # ffn can be MLP (has .layers) or MoE (has .gate, .experts)
            _wrap_ffn(child, rank, alpha, dropout)
        elif include_global_attn and hasattr(child, "global_attention"):
            _wrap_global_attn(child.global_attention, rank, alpha, dropout)
        else:
            # Recurse into children that are not special-cased
            _inject_impl(child, target_names, rank, alpha, dropout,
                         include_ffn, include_global_attn)


def _wrap_ffn(ffn_module: nn.Module, rank: int, alpha: float, dropout: float):
    """Wrap all nn.Linear layers inside a FFN (MLP or MoE) module."""
    if hasattr(ffn_module, "gate") and isinstance(ffn_module.gate, nn.Linear):
        ffn_module.gate = LoraLinear(ffn_module.gate, rank=rank, alpha=alpha, dropout=dropout)
    if hasattr(ffn_module, "experts"):
        for expert in ffn_module.experts:
            if hasattr(expert, "layers"):
                for i, layer in enumerate(expert.layers):
                    if isinstance(layer, nn.Linear):
                        expert.layers[i] = LoraLinear(layer, rank=rank, alpha=alpha, dropout=dropout)
    if hasattr(ffn_module, "layers"):
        for i, layer in enumerate(ffn_module.layers):
            if isinstance(layer, nn.Linear):
                ffn_module.layers[i] = LoraLinear(layer, rank=rank, alpha=alpha, dropout=dropout)


def _wrap_global_attn(global_attn: nn.Module, rank: int, alpha: float, dropout: float):
    """Wrap out_proj Linear in nn.MultiheadAttention (if present)."""
    if hasattr(global_attn, "out_proj") and isinstance(global_attn.out_proj, nn.Linear):
        global_attn.out_proj = LoraLinear(global_attn.out_proj, rank=rank, alpha=alpha, dropout=dropout)


# ---------------------------------------------------------------------------
# State dict helpers
# ---------------------------------------------------------------------------

def get_lora_state_dict(model: nn.Module) -> dict:
    """
    Return a state dict containing only LoRA parameters (lora_A, lora_B).
    Keys are the full dotted path within the model (e.g. ``backbone.geat_layers.0.Q_w.lora_A``).
    """
    lora_sd = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_sd[name] = param.data.clone()
    return lora_sd


def load_lora_state_dict(model: nn.Module, lora_state: dict, strict: bool = True) -> None:
    """
    Load a LoRA state dict into the model. Only touches LoRA-specific keys.

    Args:
        model: The model with LoRA adapters already injected.
        lora_state: State dict from ``get_lora_state_dict()``.
        strict: If True, raise if any LoRA key in the model is missing from
            ``lora_state`` or vice versa.
    """
    model_lora_keys = set(get_lora_state_dict(model).keys())
    provided_keys = set(lora_state.keys())

    if strict:
        missing = model_lora_keys - provided_keys
        unexpected = provided_keys - model_lora_keys
        if missing:
            raise RuntimeError(f"Missing LoRA keys in provided state dict: {missing}")
        if unexpected:
            raise RuntimeError(f"Unexpected LoRA keys in provided state dict: {unexpected}")

    model_state = model.state_dict()
    for key in provided_keys & model_lora_keys:
        model_state[key].copy_(lora_state[key])


def merge_lora_weights(model: nn.Module) -> None:
    """Merge all LoraLinear layers in the model for inference."""
    for module in model.modules():
        if isinstance(module, LoraLinear):
            module.merge()


def unmerge_lora_weights(model: nn.Module) -> None:
    """Unmerge all LoraLinear layers in the model for continued training."""
    for module in model.modules():
        if isinstance(module, LoraLinear):
            module.unmerge()
