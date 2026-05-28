# PaceLLM

> Brain-Inspired Large Language Models for Long-Context Understanding

PaceLLM is an experimental toolkit for improving long-context understanding by modifying the Transformer FFN pathway instead of only enlarging the attention window or KV cache.

The repository distills two reusable components from the original experiments:

- **Activation Memory Bank (AMB)** stores and reuses FFN intermediate activations as a lightweight working memory.
- **Cortical Expert Clustering (CEC)** reorders FFN neurons into balanced contiguous expert regions without changing parameter values or adding a router.

## Repository Layout

```text
PaceLLM/
├── pacellm/                         # Reusable PaceLLM Python package
│   ├── activation_memory_bank.py    # AMB memory and FFN wrapper
│   ├── cortical_expert_clustering.py # CEC clustering and weight rearrangement
│   ├── qwen2_integration.py         # Helpers for Hugging Face Qwen2 models
│   └── cli/                         # Installable command-line entry points
├── scripts/
│   └── cluster_qwen2_mlp.py         # CLI for applying CEC to Qwen2 checkpoints
├── examples/
│   └── enable_amb_qwen2.py          # Minimal AMB usage example
├── tests/                           # Lightweight unit tests
├── legacy/                          # Original experimental Qwen2 reference implementation
├── pyproject.toml                   # Package metadata and dependencies
├── requirements.txt                 # Runtime dependency shortcut
└── .gitignore                       # Keeps caches, checkpoints and results out of Git
```

Local experiment folders such as `AMB/`, `lkc/`, downloaded checkpoints, benchmark outputs, `.cache/`, `__pycache__/` and editor/OS files are intentionally excluded from this clean GitHub-ready folder.

## Method Overview

### Activation Memory Bank (AMB)

AMB is inserted inside a gated FFN. It stores the intermediate activation

```text
intermediate = act(gate_proj(x)) * up_proj(x)
```

For each new token activation, AMB performs cosine-similarity retrieval against memory slots:

- high similarity: reuse historical activation;
- medium similarity: fuse current and historical activations;
- low similarity: write a new memory slot or replace a low-usage slot.

Implementation: `pacellm/activation_memory_bank.py` and `pacellm/qwen2_integration.py`.

### Cortical Expert Clustering (CEC)

CEC is a one-time FFN weight reordering pass. For Qwen2/LLaMA-style gated FFNs, the same neuron permutation is applied to:

- rows of `gate_proj.weight`;
- rows of `up_proj.weight`;
- columns of `down_proj.weight`.

This keeps the function shape compatible while placing semantically similar FFN neurons into contiguous expert blocks.

Implementation: `pacellm/cortical_expert_clustering.py` and `scripts/cluster_qwen2_mlp.py`.

## Installation

```bash
conda create -n pacellm python=3.10 -y
conda activate pacellm
pip install -e .
```

Or install only runtime dependencies:

```bash
pip install -r requirements.txt
```

Optional dependency for constrained balanced K-Means:

```bash
pip install k-means-constrained
```

If `k-means-constrained` is not installed, CEC falls back to the built-in PyTorch balanced clustering implementation.

## Quick Start: Enable AMB for Qwen2

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pacellm import ActivationMemoryConfig, enable_amb_for_qwen2, reset_amb_memory

model_id = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

memory_cfg = ActivationMemoryConfig(
    bank_size=100,
    theta_high=0.75,
    theta_low=0.25,
    fusion_alpha=0.2,
)

enable_amb_for_qwen2(model, layers=[12, 26], memory_config=memory_cfg)
model.eval()

messages = [{"role": "user", "content": "Summarize the following long document: ..."}]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

with torch.no_grad():
    reset_amb_memory(model)  # reset before each independent document/task
    output = model.generate(**inputs, max_new_tokens=256, do_sample=False)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Quick Start: Apply CEC to a Qwen2 Checkpoint

```bash
python scripts/cluster_qwen2_mlp.py \
  --model /path/to/qwen2-model \
  --output-dir /path/to/qwen2-pacellm-clustered \
  --n-experts 64 \
  --backend auto \
  --cache-dir .cache/pacellm_clusters \
  --trust-remote-code
```

Load the reordered checkpoint:

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "/path/to/qwen2-pacellm-clustered",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
```

AMB can also be enabled on top of a CEC-reordered model:

```python
from pacellm import enable_amb_for_qwen2

enable_amb_for_qwen2(model, layers=[12, 26])
```

## Core API

- `ActivationMemoryConfig`: AMB hyperparameters such as memory size, similarity thresholds and fusion weight.
- `ActivationMemoryBank`: fixed-size activation-level memory module.
- `AMBFeedForward`: wrapper for an existing gated FFN (`gate_proj`, `up_proj`, `down_proj`, `act_fn`).
- `enable_amb_for_qwen2`: in-place replacement helper for selected Qwen2 MLP layers.
- `reset_amb_memory`: clears all AMB modules before a new independent generation task.
- `CorticalClusteringConfig`: CEC clustering hyperparameters.
- `cluster_and_rearrange_state_dict`: applies CEC to all discovered gated-MLP triplets in a state dict.

## Development

Run the lightweight tests:

```bash
pip install -e ".[dev]"
pytest
```

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@article{li2026pacellm,
  title={PaceLLM: Brain-Inspired Large Language Models for Long-Context Understanding},
  author={Li, Kangcong and Ye, Peng and Tu, Chongjun and Zhang, Lin and Song, Chunfeng and Wu, Jiamin and Yang, Tao and Zheng, Qihao and Chen, Tao},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  pages={85647--85672},
  year={2026}
}
```

## License

This project is released under the Apache License 2.0. See `LICENSE` for details.
