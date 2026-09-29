"""
Factory for building graph encoders using PyTorch Geometric standard layers.

Supports Graph Transformer (TransformerConv), GCN, and other PyG architectures.
All encoders are trained from scratch on WebNLG/KGQA data.
"""

import torch.nn as nn
from torch_geometric.nn import TransformerConv, GCNConv, Sequential


def build_graph_encoder(config):
    """Build a graph encoder based on config.graph_tower.

    Supported tower types:
        - "graph_transformer": Multi-layer Graph Transformer using TransformerConv
        - "gcn": Multi-layer Graph Convolutional Network using GCNConv

    Args:
        config: Must have `graph_tower` (str). May also have:
            - graph_hidden_size (int): Feature dimension (default 128)
            - graph_num_layers (int): Number of GNN layers (default 3)
            - graph_num_heads (int): Number of attention heads for Transformer (default 8)
            - graph_dropout (float): Dropout rate (default 0.1)
            - graph_edge_dim (int): Edge feature dimension (optional)

    Returns:
        nn.Module: Graph encoder with forward(g) -> [num_nodes, graph_hidden_size]
    """
    graph_tower = getattr(config, "graph_tower", None)
    if graph_tower is None:
        raise ValueError("config.graph_tower must be specified")

    hidden_size = getattr(config, "graph_hidden_size", 1024)
    num_layers = getattr(config, "graph_num_layers", 2)
    dropout = getattr(config, "graph_dropout", 0.3)
    edge_dim = getattr(config, "graph_edge_dim", None)

    if graph_tower == "graph_transformer":
        num_heads = getattr(config, "graph_num_heads", 4)
        concat = getattr(config, "graph_concat_heads", True)
        return build_graph_transformer(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            edge_dim=edge_dim,
            dropout=dropout,
            concat=concat,
        )

    elif graph_tower == "gcn":
        return build_gcn(
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )

    else:
        raise ValueError(
            f"Unknown graph_tower '{graph_tower}'. "
            "Supported: 'graph_transformer', 'gcn'"
        )


def build_graph_transformer(hidden_size, num_layers, num_heads, edge_dim, dropout, concat):
    """Build Graph Transformer using PyG's TransformerConv.

    Args:
        hidden_size: Node feature dimension
        num_layers: Number of TransformerConv layers
        num_heads: Number of attention heads
        edge_dim: Edge feature dimension (None if no edge features)
        dropout: Dropout rate
        concat: If True, concatenate heads; if False, average them

    Returns:
        Sequential model with TransformerConv layers
    """
    # Determine output dimension per layer based on concat strategy
    out_dim = hidden_size // num_heads if concat else hidden_size

    layers = []
    for i in range(num_layers):
        # Input dimension: first layer uses hidden_size, subsequent layers use layer output
        in_dim = hidden_size if i == 0 else (out_dim * num_heads if concat else out_dim)

        # TransformerConv layer
        layers.append((
            TransformerConv(
                in_channels=in_dim,
                out_channels=out_dim,
                heads=num_heads,
                concat=concat,
                dropout=dropout,
                edge_dim=edge_dim,
            ),
            'x, edge_index, edge_attr -> x'
        ))

        # Add normalization and activation for all but last layer.
        # LayerNorm is stable for very small graphs; BatchNorm1d fails when a
        # sample has only one node, which does occur in WebNLG.
        if i < num_layers - 1:
            actual_out = out_dim * num_heads if concat else out_dim
            layers.append(nn.LayerNorm(actual_out))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))

    # Final projection to ensure output is exactly hidden_size
    final_in = out_dim * num_heads if concat else out_dim
    if final_in != hidden_size:
        layers.append((nn.Linear(final_in, hidden_size), 'x -> x'))

    return GraphEncoderWrapper(Sequential('x, edge_index, edge_attr', layers), expects_edge_attr=True)


def build_gcn(hidden_size, num_layers, dropout):
    """Build GCN using PyG's GCNConv.

    Args:
        hidden_size: Node feature dimension
        num_layers: Number of GCN layers
        dropout: Dropout rate

    Returns:
        Sequential model with GCNConv layers
    """
    layers = []
    for i in range(num_layers):
        layers.append((
            GCNConv(
                in_channels=hidden_size,
                out_channels=hidden_size,
                normalize=True,
                add_self_loops=True,
            ),
            'x, edge_index -> x'
        ))

        # Add activation and dropout for all but last layer
        if i < num_layers - 1:
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))

    return GraphEncoderWrapper(Sequential('x, edge_index', layers), expects_edge_attr=False)


class GraphEncoderWrapper(nn.Module):
    """Wrapper to provide consistent interface for graph encoders.

    Expects PyG Data objects with:
        - data.graph_node: [num_nodes, hidden_size] node features
        - data.edge_index: [2, num_edges] edge connectivity
        - data.edge_attr: [num_edges, edge_dim] edge features (optional)
        - data.edge_type: [num_edges] relation types (optional, for KGs)

    Returns:
        [num_nodes, hidden_size] node embeddings
    """

    def __init__(self, sequential_model, expects_edge_attr=False):
        super().__init__()
        self.model = sequential_model
        self.expects_edge_attr = expects_edge_attr

    def forward(self, data):
        """Forward pass on PyG Data object.

        Args:
            data: PyG Data with graph_node, edge_index, and optionally edge_attr/edge_type

        Returns:
            [num_nodes, hidden_size] node embeddings
        """
        x = data.graph_node
        edge_index = data.edge_index

        # Align graph feature dtype/device with encoder weights to avoid
        # matmul dtype mismatch (e.g., Float vs BFloat16).
        ref_param = next(self.model.parameters(), None)
        if ref_param is not None:
            x = x.to(device=ref_param.device, dtype=ref_param.dtype)
            edge_index = edge_index.to(device=ref_param.device)

        # Prepare edge features (combine edge_attr and/or edge_type if present)
        edge_attr = None
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_attr = data.edge_attr
            if ref_param is not None:
                edge_attr = edge_attr.to(device=ref_param.device, dtype=ref_param.dtype)
        # Note: If edge_type is present, you may want to embed it and combine with edge_attr
        # For now, we just use edge_attr directly

        # Call model with appropriate signature
        if self.expects_edge_attr:
            return self.model(x, edge_index, edge_attr)
        else:
            return self.model(x, edge_index)
