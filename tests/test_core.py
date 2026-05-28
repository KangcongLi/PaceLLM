import torch
from torch import nn

from pacellm import (
    AMBFeedForward,
    ActivationMemoryBank,
    ActivationMemoryConfig,
    CorticalClusteringConfig,
    balanced_cluster_neurons,
    cluster_and_rearrange_state_dict,
)


def test_activation_memory_bank_roundtrip_shape():
    bank = ActivationMemoryBank(embed_dim=4, config=ActivationMemoryConfig(bank_size=3, chunk_size=2))
    queries = torch.randn(5, 4)

    assert bank.retrieve(queries).shape == queries.shape
    bank.store(queries, queries)
    assert bank.valid_mask.any()
    assert bank.retrieve(queries).shape == queries.shape

    bank.reset()
    assert not bank.valid_mask.any()


def test_amb_feed_forward_preserves_shape_in_eval():
    ffn = AMBFeedForward(
        gate_proj=nn.Linear(3, 6, bias=False),
        up_proj=nn.Linear(3, 6, bias=False),
        down_proj=nn.Linear(6, 3, bias=False),
        act_fn=torch.nn.functional.silu,
        memory_config=ActivationMemoryConfig(bank_size=4),
    )
    ffn.eval()
    hidden = torch.randn(2, 5, 3)

    output = ffn(hidden)

    assert output.shape == hidden.shape
    assert ffn.memory_bank.valid_mask.any()


def test_balanced_clustering_and_rearrangement():
    torch.manual_seed(0)
    state = {
        "model.layers.0.mlp.gate_proj.weight": torch.randn(8, 4),
        "model.layers.0.mlp.up_proj.weight": torch.randn(8, 4),
        "model.layers.0.mlp.down_proj.weight": torch.randn(4, 8),
    }
    config = CorticalClusteringConfig(n_experts=2, backend="torch", max_iter=3)

    labels = balanced_cluster_neurons(state["model.layers.0.mlp.gate_proj.weight"], config)
    assert torch.bincount(labels, minlength=2).tolist() == [4, 4]

    clustered = cluster_and_rearrange_state_dict(state, config)
    assert clustered["model.layers.0.mlp.gate_proj.weight"].shape == (8, 4)
    assert clustered["model.layers.0.mlp.up_proj.weight"].shape == (8, 4)
    assert clustered["model.layers.0.mlp.down_proj.weight"].shape == (4, 8)
