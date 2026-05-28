"""Core PaceLLM components.

This package contains the two paper-facing ideas distilled from the
experimental Qwen2 implementation:

* Activation Memory Bank (AMB)
* Cortical Expert Clustering (CEC)
"""

from .activation_memory_bank import ActivationMemoryBank, ActivationMemoryConfig, AMBFeedForward
from .cortical_expert_clustering import (
    CorticalClusteringConfig,
    balanced_cluster_neurons,
    cluster_and_rearrange_state_dict,
    rearrange_mlp_weights,
)
from .qwen2_integration import enable_amb_for_qwen2, reset_amb_memory

__all__ = [
    "ActivationMemoryBank",
    "ActivationMemoryConfig",
    "AMBFeedForward",
    "CorticalClusteringConfig",
    "balanced_cluster_neurons",
    "cluster_and_rearrange_state_dict",
    "rearrange_mlp_weights",
    "enable_amb_for_qwen2",
    "reset_amb_memory",
]
