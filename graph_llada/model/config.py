"""Graph-LLaDA configuration."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GraphLLaDAConfig:
    """Configuration for Graph-LLaDA model.

    Attributes:
        llada_model_name_or_path: HuggingFace model ID or local path for LLaDA
        graph_tower: Encoder type ("graph_transformer" or "gcn")
        graph_hidden_size: Hidden dimension of graph encoder output
        hidden_size: LLM hidden dimension (set from LLaDA config)
        graph_projector_type: Projector type ("linear" or "mlp")
        use_graph_start_end: Whether to use <g_start>/<g_end> delimiters
        freeze_graph_tower: Whether to freeze graph encoder during training
        graph_num_layers: Number of GNN layers
        graph_num_heads: Number of attention heads (for Graph Transformer)
        graph_dropout: Dropout rate
        graph_edge_dim: Edge feature dimension (optional)
    """

    # Base LLaDA model
    llada_model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"

    # Graph encoder
    graph_tower: str = "graph_transformer"
    graph_hidden_size: int = 1024
    graph_num_layers: int = 2
    graph_num_heads: int = 4
    graph_dropout: float = 0.3
    graph_edge_dim: Optional[int] = None
    graph_concat_heads: bool = True

    # LLM hidden size (set from LLaDA config)
    hidden_size: int = 4096

    # Projector
    graph_projector_type: str = "linear"

    # Token insertion
    use_graph_start_end: bool = True

    # Training
    freeze_llada: bool = True
    freeze_graph_tower: bool = True

    def update_from_llada_config(self, llada_config):
        """Update hidden_size from loaded LLaDA model's config."""
        self.hidden_size = llada_config.hidden_size
