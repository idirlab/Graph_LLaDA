"""
Factory for building graph-to-LLM projectors.

Projects graph encoder hidden states [num_nodes, graph_hidden_size]
into the LLaDA embedding space [num_nodes, llm_hidden_size].

See references/LLaDA-V/train/llava/model/multimodal_projector/ for projector patterns.
See references/GraphGPT/graphgpt/model/GraphLlama.py for the simple linear projector used there.
"""

import torch.nn as nn


def build_graph_projector(config):
    """Build a projector that maps graph_hidden_size -> hidden_size (LLM dim).

    Supported types:
        - "linear": Single linear layer (GraphGPT default)
        - "mlp" or "mlp2x_gelu": Two-layer MLP with GELU (LLaDA-V default)

    Args:
        config: Must have `graph_hidden_size` (int) and `hidden_size` (int).
                Optional: `graph_projector_type` (str, default "linear").
    """
    projector_type = getattr(config, "graph_projector_type", "linear")

    if projector_type == "linear":
        return nn.Linear(config.graph_hidden_size, config.hidden_size)
    elif projector_type in ("mlp", "mlp2x_gelu"):
        return nn.Sequential(
            nn.Linear(config.graph_hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
    else:
        raise ValueError(f"Unknown projector type '{projector_type}'. Use 'linear' or 'mlp'.")
