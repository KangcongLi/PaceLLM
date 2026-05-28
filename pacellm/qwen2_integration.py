"""Helpers for applying PaceLLM components to Hugging Face Qwen2 models."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

from torch import nn

from .activation_memory_bank import AMBFeedForward, ActivationMemoryConfig


def enable_amb_for_qwen2(
    model: nn.Module,
    layers: Iterable[int],
    memory_config: Optional[ActivationMemoryConfig] = None,
) -> nn.Module:
    """Replace selected Qwen2 MLP layers with AMB-enabled FFNs in-place.

    The target model is expected to follow the Hugging Face layout:
    `model.model.layers[i].mlp` with `gate_proj`, `up_proj`, `down_proj`,
    and `act_fn` attributes.
    """

    layer_ids = set(layers)
    decoder_layers = model.model.layers
    for layer_id in layer_ids:
        mlp = decoder_layers[layer_id].mlp
        decoder_layers[layer_id].mlp = AMBFeedForward(
            gate_proj=mlp.gate_proj,
            up_proj=mlp.up_proj,
            down_proj=mlp.down_proj,
            act_fn=mlp.act_fn,
            memory_config=memory_config,
        )
    return model


def reset_amb_memory(model: nn.Module) -> None:
    """Reset every AMB module in a patched model."""

    for module in model.modules():
        if isinstance(module, AMBFeedForward):
            module.reset_memory()
