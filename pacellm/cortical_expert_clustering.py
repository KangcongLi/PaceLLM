"""Cortical Expert Clustering for FFN neuron reordering.

The clustering step groups semantically similar FFN neurons into contiguous
blocks while preserving the original parameter values. For gated FFNs, the
same permutation must be applied to:

* rows of `gate_proj.weight`
* rows of `up_proj.weight`
* columns of `down_proj.weight`
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn.functional as F


@dataclass
class CorticalClusteringConfig:
    """Hyperparameters for balanced FFN neuron clustering."""

    n_experts: int = 64
    max_iter: int = 100
    seed: int = 0
    backend: str = "auto"
    cache_dir: Optional[str] = None

    def __post_init__(self) -> None:
        if self.n_experts <= 0:
            raise ValueError("n_experts must be positive.")
        if self.backend not in {"auto", "torch", "k_means_constrained"}:
            raise ValueError("backend must be one of: auto, torch, k_means_constrained.")


def balanced_cluster_neurons(
    gate_weight: torch.Tensor,
    config: Optional[CorticalClusteringConfig] = None,
    *,
    weight_name: str = "gate_proj",
) -> torch.Tensor:
    """Cluster FFN neurons from `gate_proj.weight` into equal-size experts.

    Args:
        gate_weight: Tensor shaped `[intermediate_size, hidden_size]`.
        config: Clustering settings.
        weight_name: Name used for cache files.

    Returns:
        Tensor of labels shaped `[intermediate_size]`, with values in
        `[0, n_experts)`.
    """

    config = config or CorticalClusteringConfig()
    if gate_weight.ndim != 2:
        raise ValueError("gate_weight must be a 2-D tensor.")
    num_neurons = gate_weight.size(0)
    if num_neurons % config.n_experts != 0:
        raise ValueError("intermediate_size must be divisible by n_experts.")

    cache_path = _cache_path(config, weight_name, tuple(gate_weight.shape))
    if cache_path is not None and cache_path.exists():
        return torch.load(cache_path, map_location="cpu")

    features = F.normalize(gate_weight.detach().float().cpu(), p=2, dim=-1)
    if config.backend in {"auto", "k_means_constrained"}:
        labels = _try_constrained_kmeans(features, config)
        if labels is None and config.backend == "k_means_constrained":
            raise ImportError("k_means_constrained is not installed.")
    else:
        labels = None

    if labels is None:
        labels = _torch_balanced_kmeans(features, config)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(labels, cache_path)
    return labels


def rearrange_mlp_weights(
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    labels: torch.Tensor,
    n_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the same neuron permutation to a gated FFN triplet."""

    order = _balanced_order(labels, n_experts).to(gate_weight.device)
    return gate_weight[order], up_weight[order], down_weight[:, order.to(down_weight.device)]


def cluster_and_rearrange_state_dict(
    state_dict: dict[str, torch.Tensor],
    config: Optional[CorticalClusteringConfig] = None,
    *,
    layer_prefixes: Optional[Iterable[str]] = None,
) -> dict[str, torch.Tensor]:
    """Return a copied state dict with clustered MLP weights.

    The function looks for Qwen/LLaMA-style keys:
    `{prefix}.gate_proj.weight`, `{prefix}.up_proj.weight`,
    `{prefix}.down_proj.weight`.
    """

    config = config or CorticalClusteringConfig()
    new_state = copy.copy(state_dict)
    prefixes = list(layer_prefixes) if layer_prefixes is not None else _discover_mlp_prefixes(state_dict)

    for prefix in prefixes:
        gate_key = f"{prefix}.gate_proj.weight"
        up_key = f"{prefix}.up_proj.weight"
        down_key = f"{prefix}.down_proj.weight"
        missing = [key for key in (gate_key, up_key, down_key) if key not in state_dict]
        if missing:
            raise KeyError(f"Missing MLP weights for {prefix}: {missing}")

        labels = balanced_cluster_neurons(
            state_dict[gate_key],
            config,
            weight_name=prefix.replace(".", "_"),
        )
        gate, up, down = rearrange_mlp_weights(
            state_dict[gate_key],
            state_dict[up_key],
            state_dict[down_key],
            labels,
            config.n_experts,
        )
        new_state[gate_key] = gate
        new_state[up_key] = up
        new_state[down_key] = down

    return new_state


def _discover_mlp_prefixes(state_dict: dict[str, torch.Tensor]) -> list[str]:
    suffix = ".gate_proj.weight"
    return sorted(key[: -len(suffix)] for key in state_dict if key.endswith(suffix))


def _cache_path(
    config: CorticalClusteringConfig,
    weight_name: str,
    shape: tuple[int, int],
) -> Optional[Path]:
    if config.cache_dir is None:
        return None
    safe_name = weight_name.replace("/", "_").replace(".", "_")
    return Path(config.cache_dir) / f"{safe_name}_clusters{config.n_experts}_shape{shape}.pth"


def _try_constrained_kmeans(
    features: torch.Tensor,
    config: CorticalClusteringConfig,
) -> Optional[torch.Tensor]:
    try:
        from k_means_constrained import KMeansConstrained
    except ImportError:
        return None

    expert_size = features.size(0) // config.n_experts
    estimator = KMeansConstrained(
        n_clusters=config.n_experts,
        size_min=expert_size,
        size_max=expert_size,
        random_state=config.seed,
        max_iter=config.max_iter,
    )
    estimator.fit(features.numpy())
    return torch.as_tensor(estimator.labels_, dtype=torch.long)


def _torch_balanced_kmeans(features: torch.Tensor, config: CorticalClusteringConfig) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    num_neurons = features.size(0)
    expert_size = num_neurons // config.n_experts
    init = torch.randperm(num_neurons, generator=generator)[: config.n_experts]
    centers = features[init].clone()

    labels = torch.zeros(num_neurons, dtype=torch.long)
    for _ in range(config.max_iter):
        previous = labels.clone()
        labels = _balanced_assign(features, centers, expert_size)
        for expert_id in range(config.n_experts):
            member = features[labels == expert_id]
            if member.numel() > 0:
                centers[expert_id] = F.normalize(member.mean(dim=0), p=2, dim=0)
        if torch.equal(labels, previous):
            break
    return labels


def _balanced_assign(features: torch.Tensor, centers: torch.Tensor, expert_size: int) -> torch.Tensor:
    scores = features @ centers.T
    sorted_scores, sorted_experts = scores.sort(dim=-1, descending=True)
    if centers.size(0) == 1:
        return torch.zeros(features.size(0), dtype=torch.long)
    margin = sorted_scores[:, 0] - sorted_scores[:, 1]
    token_order = torch.argsort(margin, descending=True)

    labels = torch.full((features.size(0),), -1, dtype=torch.long)
    capacity = torch.full((centers.size(0),), expert_size, dtype=torch.long)
    for neuron_idx in token_order.tolist():
        for expert_id in sorted_experts[neuron_idx].tolist():
            if capacity[expert_id] > 0:
                labels[neuron_idx] = expert_id
                capacity[expert_id] -= 1
                break
    if (labels < 0).any():
        raise RuntimeError("Balanced assignment failed; check n_experts and expert size.")
    return labels


def _balanced_order(labels: torch.Tensor, n_experts: int) -> torch.Tensor:
    labels = labels.detach().cpu().long()
    counts = torch.bincount(labels, minlength=n_experts)
    if counts.unique().numel() != 1:
        raise ValueError("labels must contain the same number of neurons per expert.")
    order = []
    for expert_id in range(n_experts):
        order.append(torch.where(labels == expert_id)[0])
    return torch.cat(order, dim=0)
