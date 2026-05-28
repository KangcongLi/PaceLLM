"""Activation Memory Bank for FFN-level working memory.

The module stores FFN intermediate activations rather than tokens or KV cache
entries. During inference, new activations retrieve similar historical states
and then update the bank with reuse / fusion / replacement rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class ActivationMemoryConfig:
    """Hyperparameters for the Activation Memory Bank."""

    bank_size: int = 100
    theta_high: float = 0.75
    theta_low: float = 0.25
    retrieve_topk: int = 5
    negative_topk: int = 2
    store_topk: int = 5
    fusion_alpha: float = 0.2
    high_reuse_weight: float = 1.0
    mid_reuse_weight: float = 0.5
    negative_weight: float = 0.1
    update_momentum: float = 0.5
    chunk_size: int = 1024

    def __post_init__(self) -> None:
        if self.bank_size <= 0:
            raise ValueError("bank_size must be positive.")
        if not 0 <= self.theta_low <= self.theta_high <= 1:
            raise ValueError("thresholds must satisfy 0 <= theta_low <= theta_high <= 1.")
        if not 0 <= self.fusion_alpha <= 1:
            raise ValueError("fusion_alpha must be in [0, 1].")
        if not 0 <= self.update_momentum <= 1:
            raise ValueError("update_momentum must be in [0, 1].")


class ActivationMemoryBank(nn.Module):
    """A fixed-size activation-level memory bank.

    Keys represent raw FFN intermediate activations. Values represent enhanced
    FFN activations that can be reused in later tokens. Buffers are registered
    so the bank follows `.to(device)` / `.half()` with the parent model.
    """

    def __init__(
        self,
        embed_dim: int,
        config: Optional[ActivationMemoryConfig] = None,
        *,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.config = config or ActivationMemoryConfig()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.register_buffer(
            "memory_keys",
            torch.zeros(self.config.bank_size, embed_dim, **factory_kwargs),
            persistent=False,
        )
        self.register_buffer(
            "memory_values",
            torch.zeros(self.config.bank_size, embed_dim, **factory_kwargs),
            persistent=False,
        )
        self.register_buffer(
            "usage_counts",
            torch.zeros(self.config.bank_size, device=device, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "valid_mask",
            torch.zeros(self.config.bank_size, device=device, dtype=torch.bool),
            persistent=False,
        )

    @torch.no_grad()
    def reset(self) -> None:
        """Clear all remembered activations before starting a new document."""

        self.memory_keys.zero_()
        self.memory_values.zero_()
        self.usage_counts.zero_()
        self.valid_mask.zero_()

    def retrieve(self, queries: torch.Tensor) -> torch.Tensor:
        """Retrieve reusable activation states for flattened FFN activations.

        Args:
            queries: Tensor with shape `[num_tokens, intermediate_size]`.

        Returns:
            Tensor with the same shape as `queries`.
        """

        if queries.ndim != 2 or queries.size(-1) != self.embed_dim:
            raise ValueError(f"queries must have shape [N, {self.embed_dim}].")
        if self.memory_keys.device != queries.device:
            self.to(device=queries.device)
        if not bool(self.valid_mask.any()):
            return queries

        outputs = []
        for start in range(0, queries.size(0), self.config.chunk_size):
            chunk = queries[start : start + self.config.chunk_size]
            outputs.append(self._retrieve_chunk(chunk))
        return torch.cat(outputs, dim=0)

    def _retrieve_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        keys = self.memory_keys[self.valid_mask].to(dtype=chunk.dtype)
        values = self.memory_values[self.valid_mask].to(dtype=chunk.dtype)
        sim = F.cosine_similarity(chunk.unsqueeze(1), keys.unsqueeze(0), dim=-1)

        topk = min(self.config.retrieve_topk, values.size(0))
        negk = min(self.config.negative_topk, values.size(0))
        topk_sim, topk_idx = sim.topk(topk, dim=-1)
        pos_mean = values[topk_idx].mean(dim=1)

        if negk > 0:
            neg_idx = sim.topk(negk, dim=-1, largest=False).indices
            neg_mean = values[neg_idx].mean(dim=1)
        else:
            neg_mean = torch.zeros_like(pos_mean)

        max_sim = topk_sim.max(dim=-1).values
        high = max_sim > self.config.theta_high
        mid = (max_sim > self.config.theta_low) & ~high

        retrieved = chunk.clone()
        retrieved[high] = (
            self.config.high_reuse_weight * pos_mean[high]
            + self.config.negative_weight * neg_mean[high]
        )
        retrieved[mid] = (
            self.config.mid_reuse_weight * pos_mean[mid]
            + self.config.negative_weight * neg_mean[mid]
        )
        return retrieved

    @torch.no_grad()
    def store(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        """Update the bank using high / mid / low similarity policies."""

        if keys.shape != values.shape:
            raise ValueError("keys and values must have the same shape.")
        if keys.ndim != 2 or keys.size(-1) != self.embed_dim:
            raise ValueError(f"keys must have shape [N, {self.embed_dim}].")

        keys = keys.detach().to(device=self.memory_keys.device, dtype=self.memory_keys.dtype)
        values = values.detach().to(device=self.memory_values.device, dtype=self.memory_values.dtype)

        for start in range(0, keys.size(0), self.config.chunk_size):
            end = start + self.config.chunk_size
            self._store_chunk(keys[start:end], values[start:end])

    def _store_chunk(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        if not bool(self.valid_mask.any()):
            self._replace_slots(keys, values)
            return

        active_keys = self.memory_keys[self.valid_mask]
        sim = F.cosine_similarity(keys.unsqueeze(1), active_keys.unsqueeze(0), dim=-1)
        topk = min(self.config.store_topk, active_keys.size(0))
        topk_sim, topk_idx_active = sim.topk(topk, dim=-1)
        mean_sim = topk_sim.mean(dim=-1)

        active_slots = torch.where(self.valid_mask)[0]
        topk_slots = active_slots[topk_idx_active]
        mid = (mean_sim > self.config.theta_low) & (mean_sim <= self.config.theta_high)
        low = mean_sim <= self.config.theta_low

        if mid.any():
            self._merge_slots(topk_slots[mid], keys[mid], values[mid])
        if low.any():
            self._replace_slots(keys[low], values[low])

    def _merge_slots(self, slots: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> None:
        unique_slots = torch.unique(slots.reshape(-1))
        if unique_slots.numel() == 0:
            return
        key_update = keys.mean(dim=0)
        value_update = values.mean(dim=0)
        m = self.config.update_momentum
        self.memory_keys[unique_slots] = m * self.memory_keys[unique_slots] + (1 - m) * key_update
        self.memory_values[unique_slots] = m * self.memory_values[unique_slots] + (1 - m) * value_update
        self.usage_counts[unique_slots] += 1
        self.valid_mask[unique_slots] = True

    def _replace_slots(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        if keys.numel() == 0:
            return
        n_items = min(keys.size(0), self.config.bank_size)
        free_slots = torch.where(~self.valid_mask)[0]
        if free_slots.numel() < n_items:
            used_slots = torch.argsort(self.usage_counts.masked_fill(~self.valid_mask, float("inf")))
            slots = torch.cat([free_slots, used_slots])[:n_items]
        else:
            slots = free_slots[:n_items]

        self.memory_keys[slots] = keys[: slots.numel()]
        self.memory_values[slots] = values[: slots.numel()]
        self.usage_counts[slots] = 1
        self.valid_mask[slots] = True


class AMBFeedForward(nn.Module):
    """Qwen/LLaMA-style gated FFN with Activation Memory Bank.

    This class is intentionally lightweight: pass existing `gate_proj`,
    `up_proj`, `down_proj`, and activation function to wrap an FFN block.
    """

    def __init__(
        self,
        gate_proj: nn.Module,
        up_proj: nn.Module,
        down_proj: nn.Module,
        act_fn: Callable[[torch.Tensor], torch.Tensor],
        *,
        memory_config: Optional[ActivationMemoryConfig] = None,
    ) -> None:
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.act_fn = act_fn

        intermediate_size = gate_proj.out_features
        self.memory_bank = ActivationMemoryBank(intermediate_size, memory_config)

    @torch.no_grad()
    def reset_memory(self) -> None:
        self.memory_bank.reset()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        intermediate = self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        flat = intermediate.reshape(-1, intermediate.size(-1))
        if self.memory_bank.memory_keys.device != flat.device:
            self.memory_bank.to(device=flat.device)

        retrieved = self.memory_bank.retrieve(flat)
        alpha = self.memory_bank.config.fusion_alpha
        enhanced = (1 - alpha) * flat + alpha * retrieved

        if not self.training:
            self.memory_bank.store(flat, enhanced)

        return self.down_proj(enhanced.reshape(batch_size, seq_len, -1))
