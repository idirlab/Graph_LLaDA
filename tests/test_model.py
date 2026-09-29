"""
Comprehensive tests for Graph-LLaDA model components.

Tests cover:
1. Graph encoder builder - PyG-based encoders
2. Graph projector - all types, shape mapping
3. Token insertion (prepare_inputs_with_graphs) - placeholder replacement, label masking, padding
4. Combined model (GraphLLaDAForMLM) - end-to-end forward with mock LLaDA
5. Tokenizer integration - graph token addition, input building

Run with: pytest tests/test_model.py -v
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock
from transformers.modeling_outputs import CausalLMOutputWithPast

from graph_llada.constants import (
    IGNORE_INDEX,
    GRAPH_TOKEN_INDEX,
    DEFAULT_GRAPH_TOKEN,
    DEFAULT_GRAPH_PATCH_TOKEN,
    DEFAULT_G_START_TOKEN,
    DEFAULT_G_END_TOKEN,
    MASK_TOKEN_ID,
)
from graph_llada.model.graph_encoder.builder import build_graph_encoder
from graph_llada.model.graph_projector.builder import build_graph_projector
from graph_llada.model.graph_llada_arch import (
    _find_contiguous_segments,
    _pad_batch,
    GraphLLaDAMetaForCausalLM,
)
from graph_llada.model.language_model.configuration import GraphLLaDAConfig
from graph_llada.model.language_model.graph_llada_model import (
    GraphLLaDAForMLM,
    build_graph_text_input,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_graph(num_nodes=5, feat_dim=128, num_edges=8, seed=42):
    """Create a synthetic PyG Data object for testing."""
    torch.manual_seed(seed)
    graph_node = torch.randn(num_nodes, feat_dim)
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    edge_index = torch.stack([src, dst])
    return Data(graph_node=graph_node, edge_index=edge_index)


def make_graph_with_relations(num_nodes=5, feat_dim=128, num_edges=8,
                               num_relation_types=4, edge_attr_dim=None, seed=42):
    """Create a PyG Data with discrete edge_type and optional continuous edge_attr."""
    torch.manual_seed(seed)
    graph_node = torch.randn(num_nodes, feat_dim)
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    edge_index = torch.stack([src, dst])
    edge_type = torch.randint(0, num_relation_types, (num_edges,))
    data = Data(graph_node=graph_node, edge_index=edge_index, edge_type=edge_type)
    if edge_attr_dim is not None:
        data.edge_attr = torch.randn(num_edges, edge_attr_dim)
    return data


def make_link_prediction_graph(num_nodes=5, feat_dim=128, num_edges=8, seed=42):
    """Create a dict with two graphs for link prediction testing."""
    return {
        "graph_1": make_graph(num_nodes, feat_dim, num_edges, seed),
        "graph_2": make_graph(num_nodes, feat_dim, num_edges, seed + 1),
    }


@dataclass
class SimpleConfig:
    """Minimal config for testing encoder/projector builders."""
    graph_tower: str = "graph_transformer"
    graph_hidden_size: int = 128
    hidden_size: int = 256  # Small for testing
    graph_projector_type: str = "linear"
    graph_num_layers: int = 2
    graph_num_heads: int = 4
    graph_dropout: float = 0.1
    graph_edge_dim: Optional[int] = None
    graph_concat_heads: bool = True
    freeze_graph_tower: bool = True


# ---------------------------------------------------------------------------
# Test: Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_ignore_index(self):
        assert IGNORE_INDEX == -100

    def test_graph_token_index(self):
        assert GRAPH_TOKEN_INDEX == -200

    def test_graph_tokens(self):
        assert DEFAULT_GRAPH_TOKEN == "<graph>"
        assert DEFAULT_GRAPH_PATCH_TOKEN == "<g_patch>"
        assert DEFAULT_G_START_TOKEN == "<g_start>"
        assert DEFAULT_G_END_TOKEN == "<g_end>"

    def test_mask_token_id(self):
        assert MASK_TOKEN_ID == 126336


# ---------------------------------------------------------------------------
# Test: Graph Projector
# ---------------------------------------------------------------------------


class TestGraphProjector:
    def test_linear(self):
        config = SimpleConfig(graph_projector_type="linear")
        proj = build_graph_projector(config)
        x = torch.randn(10, 128)
        out = proj(x)
        assert out.shape == (10, 256)

    def test_mlp(self):
        config = SimpleConfig(graph_projector_type="mlp")
        proj = build_graph_projector(config)
        x = torch.randn(10, 128)
        out = proj(x)
        assert out.shape == (10, 256)

    def test_mlp2x_gelu(self):
        config = SimpleConfig(graph_projector_type="mlp2x_gelu")
        proj = build_graph_projector(config)
        x = torch.randn(10, 128)
        out = proj(x)
        assert out.shape == (10, 256)

    def test_invalid_type(self):
        config = SimpleConfig(graph_projector_type="invalid")
        with pytest.raises(ValueError):
            build_graph_projector(config)


# ---------------------------------------------------------------------------
# Test: Encoder Builder
# ---------------------------------------------------------------------------


class TestEncoderBuilder:
    def test_build_graph_transformer(self):
        config = SimpleConfig(graph_tower="graph_transformer", graph_hidden_size=64)
        encoder = build_graph_encoder(config)
        g = make_graph(num_nodes=5, feat_dim=64, num_edges=10)
        out = encoder(g)
        assert out.shape == (5, 64)

    def test_build_gcn(self):
        config = SimpleConfig(graph_tower="gcn", graph_hidden_size=64)
        encoder = build_graph_encoder(config)
        g = make_graph(num_nodes=5, feat_dim=64, num_edges=10)
        out = encoder(g)
        assert out.shape == (5, 64)

    def test_invalid_type(self):
        config = SimpleConfig(graph_tower="nonexistent")
        with pytest.raises(ValueError):
            build_graph_encoder(config)

    def test_no_tower(self):
        config = SimpleConfig(graph_tower=None)
        with pytest.raises(ValueError):
            build_graph_encoder(config)

    def test_with_edge_features(self):
        """Test encoder with edge features."""
        config = SimpleConfig(
            graph_tower="graph_transformer",
            graph_hidden_size=64,
            graph_edge_dim=32,
        )
        encoder = build_graph_encoder(config)
        g = make_graph_with_relations(
            num_nodes=5, feat_dim=64, num_edges=10,
            edge_attr_dim=32
        )
        out = encoder(g)
        assert out.shape == (5, 64)


# ---------------------------------------------------------------------------
# Test: Token Insertion Utilities
# ---------------------------------------------------------------------------


class TestContiguousSegments:
    def test_single_segment(self):
        assert _find_contiguous_segments([3, 4, 5]) == [(3, 6)]

    def test_multiple_segments(self):
        assert _find_contiguous_segments([3, 4, 5, 10, 11]) == [(3, 6), (10, 12)]

    def test_single_element(self):
        assert _find_contiguous_segments([7]) == [(7, 8)]

    def test_non_contiguous(self):
        assert _find_contiguous_segments([1, 3, 5]) == [(1, 2), (3, 4), (5, 6)]

    def test_empty(self):
        assert _find_contiguous_segments([]) == []


class TestPadBatch:
    def test_uniform_length(self):
        """No padding needed when all sequences have same length."""
        embeds = [torch.randn(5, 32), torch.randn(5, 32)]
        labels = [torch.arange(5), torch.arange(5)]
        pe, pl, pm = _pad_batch(embeds, labels, None, torch.float32, "cpu")
        assert pe.shape == (2, 5, 32)
        assert pl.shape == (2, 5)
        assert pm.shape == (2, 5)
        assert (pm == 1).all()

    def test_variable_length(self):
        """Should pad shorter sequences with zeros/IGNORE_INDEX."""
        embeds = [torch.randn(3, 32), torch.randn(7, 32)]
        labels = [torch.arange(3), torch.arange(7)]
        pe, pl, pm = _pad_batch(embeds, labels, None, torch.float32, "cpu")
        assert pe.shape == (2, 7, 32)
        assert pl.shape == (2, 7)
        # First sequence padded at positions 3-6
        assert (pm[0, :3] == 1).all()
        assert (pm[0, 3:] == 0).all()
        assert (pl[0, 3:] == IGNORE_INDEX).all()
        # Second sequence fully valid
        assert (pm[1] == 1).all()

    def test_no_labels(self):
        embeds = [torch.randn(5, 32)]
        pe, pl, pm = _pad_batch(embeds, None, None, torch.float32, "cpu")
        assert pe.shape == (1, 5, 32)
        assert pl is None


# ---------------------------------------------------------------------------
# Test: Configuration
# ---------------------------------------------------------------------------


class TestGraphLLaDAConfig:
    def test_defaults(self):
        config = GraphLLaDAConfig()
        assert config.graph_tower == "graph_transformer"
        assert config.graph_hidden_size == 128
        assert config.hidden_size == 4096
        assert config.graph_projector_type == "linear"
        assert config.use_graph_start_end is True
        assert config.freeze_graph_tower is True

    def test_custom_values(self):
        config = GraphLLaDAConfig(
            graph_tower="gcn",
            graph_hidden_size=256,
            graph_projector_type="mlp",
        )
        assert config.graph_tower == "gcn"
        assert config.graph_hidden_size == 256
        assert config.graph_projector_type == "mlp"

    def test_update_from_llada_config(self):
        config = GraphLLaDAConfig()
        mock_llada_config = MagicMock()
        mock_llada_config.hidden_size = 2048
        config.update_from_llada_config(mock_llada_config)
        assert config.hidden_size == 2048


# ---------------------------------------------------------------------------
# Test: Meta-architecture (Token Insertion)
# ---------------------------------------------------------------------------


class MockModel(nn.Module):
    """Minimal mock model with embed_tokens and graph modules for testing."""

    def __init__(self, vocab_size=100, hidden_size=32, graph_hidden_size=16):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        # Use actual PyG encoder
        from graph_llada.model.graph_encoder.builder import build_graph_encoder
        config = SimpleConfig(
            graph_tower="gcn",
            graph_hidden_size=graph_hidden_size,
            graph_num_layers=1,
        )
        self.graph_tower = build_graph_encoder(config)
        self.graph_projector = nn.Linear(graph_hidden_size, hidden_size)

    def get_graph_tower(self):
        return self.graph_tower


class MockGraphLLaDA(nn.Module):
    """Minimal model implementing GraphLLaDAMetaForCausalLM interface for testing."""

    def __init__(self, vocab_size=100, hidden_size=32, graph_hidden_size=16):
        super().__init__()
        self._model = MockModel(vocab_size, hidden_size, graph_hidden_size)

    def get_model(self):
        return self._model

    def get_graph_tower(self):
        return self._model.get_graph_tower()

    @property
    def embed_tokens(self):
        return self._model.embed_tokens

    @property
    def graph_projector(self):
        return self._model.graph_projector


class HelperGraphLLaDA(MockGraphLLaDA, GraphLLaDAMetaForCausalLM):
    """Mix in meta class methods for testing."""
    pass


class TestTokenInsertion:
    def setup_method(self):
        self.model = HelperGraphLLaDA(
            vocab_size=100, hidden_size=32, graph_hidden_size=16
        )

    def test_no_graph_tokens(self):
        """When no GRAPH_TOKEN_INDEX in input, should just embed text."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        labels = torch.tensor([[1, 2, 3, 4, 5]])
        embeds, out_labels, attn_mask = self.model.prepare_inputs_with_graphs(
            input_ids, labels, graph_data=None
        )
        assert embeds.shape == (1, 5, 32)

    def test_with_graph_tokens(self):
        """Graph token positions should be replaced and labels set to IGNORE_INDEX."""
        input_ids = torch.tensor([[1, 2, GRAPH_TOKEN_INDEX, GRAPH_TOKEN_INDEX, GRAPH_TOKEN_INDEX, 5]])
        labels = torch.tensor([[1, 2, 3, 4, 5, 6]])

        g = make_graph(num_nodes=3, feat_dim=16, num_edges=4)
        embeds, out_labels, attn_mask = self.model.prepare_inputs_with_graphs(
            input_ids, labels, graph_data=[g]
        )

        assert embeds.shape == (1, 6, 32)
        assert out_labels.shape == (1, 6)

        # Graph token positions should have IGNORE_INDEX labels
        assert out_labels[0, 2].item() == IGNORE_INDEX
        assert out_labels[0, 3].item() == IGNORE_INDEX
        assert out_labels[0, 4].item() == IGNORE_INDEX

        # Non-graph positions should retain original labels
        assert out_labels[0, 0].item() == 1
        assert out_labels[0, 1].item() == 2
        assert out_labels[0, 5].item() == 6

    def test_link_prediction_dual_graph(self):
        """Link prediction samples have two graphs per sample."""
        input_ids = torch.tensor([[
            1, 2,
            GRAPH_TOKEN_INDEX, GRAPH_TOKEN_INDEX,  # First graph (2 nodes)
            3,
            GRAPH_TOKEN_INDEX, GRAPH_TOKEN_INDEX,  # Second graph (2 nodes)
            4
        ]])
        labels = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])

        g_dict = make_link_prediction_graph(num_nodes=2, feat_dim=16, num_edges=2)

        embeds, out_labels, attn_mask = self.model.prepare_inputs_with_graphs(
            input_ids, labels, graph_data=[g_dict]
        )

        assert embeds.shape == (1, 8, 32)

        # Graph positions should have IGNORE_INDEX
        assert out_labels[0, 2].item() == IGNORE_INDEX
        assert out_labels[0, 3].item() == IGNORE_INDEX
        assert out_labels[0, 5].item() == IGNORE_INDEX
        assert out_labels[0, 6].item() == IGNORE_INDEX


# ---------------------------------------------------------------------------
# Test: GraphLLaDAForMLM (with mock LLaDA)
# ---------------------------------------------------------------------------


class MockLLaDAModel(nn.Module):
    """Mock LLaDA model for testing the full GraphLLaDAForMLM without loading real weights."""

    def __init__(self, vocab_size=1000, hidden_size=256, num_layers=2):
        super().__init__()
        self.config = MagicMock()
        self.config.hidden_size = hidden_size
        self.config.vocab_size = vocab_size

        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids=None, inputs_embeds=None, labels=None,
                attention_mask=None, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        batch_size, seq_len, hidden_size = inputs_embeds.shape
        logits = self.lm_head(inputs_embeds)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def resize_token_embeddings(self, new_size):
        old_embed = self.model.embed_tokens
        new_embed = nn.Embedding(new_size, old_embed.embedding_dim)
        min_size = min(old_embed.num_embeddings, new_size)
        new_embed.weight.data[:min_size] = old_embed.weight.data[:min_size]
        self.model.embed_tokens = new_embed

        old_head = self.lm_head
        new_head = nn.Linear(old_head.in_features, new_size, bias=False)
        min_size = min(old_head.out_features, new_size)
        new_head.weight.data[:min_size] = old_head.weight.data[:min_size]
        self.lm_head = new_head


class TestGraphLLaDAForMLM:
    def setup_method(self):
        """Create a GraphLLaDAForMLM with mock LLaDA."""
        self.mock_llada = MockLLaDAModel(vocab_size=1000, hidden_size=256)
        self.config = GraphLLaDAConfig(
            graph_tower="gcn",
            graph_hidden_size=128,
            hidden_size=256,
            graph_projector_type="linear",
            freeze_graph_tower=True,
            graph_num_layers=2,
        )
        self.model = GraphLLaDAForMLM(self.config, llada_model=self.mock_llada)

    def test_initialization(self):
        """Model should initialize with graph tower, projector, and LLaDA."""
        assert self.model.graph_tower is not None
        assert self.model.graph_projector is not None
        assert self.model.llada is not None

    def test_graph_tower_frozen(self):
        """Graph tower should be frozen when freeze_graph_tower=True."""
        for param in self.model.graph_tower.parameters():
            assert not param.requires_grad

    def test_projector_trainable(self):
        """Projector should be trainable by default."""
        for param in self.model.graph_projector.parameters():
            assert param.requires_grad

    def test_forward_without_graph(self):
        """Forward without graph data should work (text-only)."""
        input_ids = torch.randint(0, 100, (2, 10))
        labels = torch.randint(0, 100, (2, 10))
        output = self.model(input_ids=input_ids, labels=labels)
        assert output.loss is not None
        assert output.logits.shape[0] == 2

    def test_forward_with_graph(self):
        """Forward with graph data should compute loss."""
        input_ids = torch.tensor([
            [1, 2, 3, GRAPH_TOKEN_INDEX, GRAPH_TOKEN_INDEX, GRAPH_TOKEN_INDEX, 7, 8, 9, 10]
        ])
        labels = torch.tensor([
            [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 4, 5, 6, 7, 8, 9, 10]
        ])

        g = make_graph(num_nodes=3, feat_dim=128, num_edges=6)
        output = self.model(input_ids=input_ids, labels=labels, graph_data=[g])

        assert output.loss is not None
        assert output.logits is not None
        assert output.logits.shape[1] == 10

    def test_mlp_projector(self):
        """Should work with MLP projector too."""
        config = GraphLLaDAConfig(
            graph_tower="gcn",
            graph_hidden_size=128,
            hidden_size=256,
            graph_projector_type="mlp",
            graph_num_layers=1,
        )
        mock_llada = MockLLaDAModel(vocab_size=1000, hidden_size=256)
        model = GraphLLaDAForMLM(config, llada_model=mock_llada)

        input_ids = torch.tensor([[1, GRAPH_TOKEN_INDEX, GRAPH_TOKEN_INDEX, 4]])
        labels = torch.tensor([[1, 2, 3, 4]])
        g = make_graph(num_nodes=2, feat_dim=128, num_edges=2)

        output = model(input_ids=input_ids, labels=labels, graph_data=[g])
        assert output.loss is not None

    def test_graph_transformer_encoder(self):
        """Should work with Graph Transformer encoder too."""
        config = GraphLLaDAConfig(
            graph_tower="graph_transformer",
            graph_hidden_size=128,
            hidden_size=256,
            graph_num_layers=2,
            graph_num_heads=4,
        )
        mock_llada = MockLLaDAModel(vocab_size=1000, hidden_size=256)
        model = GraphLLaDAForMLM(config, llada_model=mock_llada)

        input_ids = torch.tensor([[1, GRAPH_TOKEN_INDEX, GRAPH_TOKEN_INDEX, 4]])
        labels = torch.tensor([[1, 2, 3, 4]])
        g = make_graph(num_nodes=2, feat_dim=128, num_edges=2)

        output = model(input_ids=input_ids, labels=labels, graph_data=[g])
        assert output.loss is not None


# ---------------------------------------------------------------------------
# Test: Tokenizer Integration
# ---------------------------------------------------------------------------


class TestTokenizerIntegration:
    """Tests using a simple mock tokenizer."""

    def setup_method(self):
        """Set up a mock tokenizer that can handle graph tokens."""
        from transformers import PreTrainedTokenizerFast
        from tokenizers import Tokenizer, models, pre_tokenizers

        tokenizer_model = models.WordLevel(
            vocab={
                "<pad>": 0, "<s>": 1, "</s>": 2, "Given": 3, "a": 4,
                "graph": 5, ":": 6, "What": 7, "is": 8, "the": 9,
                "category": 10, "?": 11, "\n": 12, " ": 13,
            },
            unk_token="<pad>",
        )
        base_tokenizer = Tokenizer(tokenizer_model)
        base_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=base_tokenizer,
            pad_token="<pad>",
            bos_token="<s>",
            eos_token="</s>",
        )

    def test_add_graph_tokens(self):
        """Should be able to add graph tokens to the tokenizer."""
        orig_size = len(self.tokenizer)
        self.tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
        self.tokenizer.add_tokens(
            [DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True
        )
        assert len(self.tokenizer) == orig_size + 3

    def test_graph_token_ids(self):
        """Graph tokens should have valid IDs after being added."""
        self.tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
        self.tokenizer.add_tokens(
            [DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True
        )
        patch_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_GRAPH_PATCH_TOKEN)
        start_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_G_START_TOKEN)
        end_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_G_END_TOKEN)

        assert patch_id != self.tokenizer.unk_token_id
        assert start_id != self.tokenizer.unk_token_id
        assert end_id != self.tokenizer.unk_token_id


# ---------------------------------------------------------------------------
# Test: build_graph_text_input helper
# ---------------------------------------------------------------------------


class TestBuildGraphTextInput:
    """Test the helper function that builds input_ids with graph placeholders."""

    def setup_method(self):
        from transformers import PreTrainedTokenizerFast
        from tokenizers import Tokenizer, models, pre_tokenizers

        vocab = {
            "<pad>": 0, "<s>": 1, "</s>": 2,
            "Given": 3, "a": 4, "citation": 5, "graph": 6,
            ":": 7, "What": 8, "is": 9, "it": 10, "?": 11,
        }
        tokenizer_model = models.WordLevel(vocab=vocab, unk_token="<pad>")
        base_tokenizer = Tokenizer(tokenizer_model)
        base_tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=base_tokenizer,
            pad_token="<pad>",
        )
        # Add graph tokens
        self.tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
        self.tokenizer.add_tokens(
            [DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True
        )

    def test_with_start_end(self):
        """Should place <g_start>, N x <g_patch>, <g_end> where <graph> was."""
        text = f"Given a {DEFAULT_GRAPH_TOKEN} What is it ?"
        input_ids = build_graph_text_input(
            text, self.tokenizer, num_graph_nodes=3, use_graph_start_end=True
        )
        # Check that GRAPH_TOKEN_INDEX appears exactly 3 times
        assert (input_ids == GRAPH_TOKEN_INDEX).sum().item() == 3

        # Check that g_start and g_end tokens are present
        g_start_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_G_START_TOKEN)
        g_end_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_G_END_TOKEN)
        assert (input_ids == g_start_id).sum().item() == 1
        assert (input_ids == g_end_id).sum().item() == 1

    def test_without_start_end(self):
        """Without start/end, should have only <g_patch> replacements."""
        text = f"Given a {DEFAULT_GRAPH_TOKEN} What"
        input_ids = build_graph_text_input(
            text, self.tokenizer, num_graph_nodes=5, use_graph_start_end=False
        )
        assert (input_ids == GRAPH_TOKEN_INDEX).sum().item() == 5

    def test_graph_tokens_are_contiguous(self):
        """The GRAPH_TOKEN_INDEX positions should be contiguous."""
        text = f"Given {DEFAULT_GRAPH_TOKEN} What"
        input_ids = build_graph_text_input(
            text, self.tokenizer, num_graph_nodes=4, use_graph_start_end=True
        )
        graph_positions = torch.where(input_ids == GRAPH_TOKEN_INDEX)[0]
        if len(graph_positions) > 1:
            diffs = graph_positions[1:] - graph_positions[:-1]
            assert (diffs == 1).all()


# ---------------------------------------------------------------------------
# Test: End-to-End Flow
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """Integration tests that verify the full pipeline works on CPU."""

    def test_full_pipeline(self):
        """Test: config -> model -> forward with graph data."""
        mock_llada = MockLLaDAModel(vocab_size=500, hidden_size=64)
        config = GraphLLaDAConfig(
            graph_tower="gcn",
            graph_hidden_size=32,
            hidden_size=64,
            graph_projector_type="linear",
            graph_num_layers=2,
        )
        model = GraphLLaDAForMLM(config, llada_model=mock_llada)

        # Build input
        num_nodes = 5
        input_ids = torch.tensor([[
            1, 2, 3,
            GRAPH_TOKEN_INDEX, GRAPH_TOKEN_INDEX, GRAPH_TOKEN_INDEX,
            GRAPH_TOKEN_INDEX, GRAPH_TOKEN_INDEX,
            10, 11, 12
        ]])
        labels = torch.tensor([[
            IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX,
            IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX,
            IGNORE_INDEX, IGNORE_INDEX,
            10, 11, 12
        ]])

        g = make_graph(num_nodes=num_nodes, feat_dim=32, num_edges=10)
        output = model(input_ids=input_ids, labels=labels, graph_data=[g])

        assert output.loss is not None
        assert not torch.isnan(output.loss)
        assert output.logits.shape == (1, 11, 500)

    def test_gradient_flow_through_projector(self):
        """Gradients should flow through the projector but not the frozen encoder."""
        mock_llada = MockLLaDAModel(vocab_size=100, hidden_size=64)
        config = GraphLLaDAConfig(
            graph_tower="gcn",
            graph_hidden_size=32,
            hidden_size=64,
            graph_num_layers=1,
            freeze_graph_tower=True,
        )
        model = GraphLLaDAForMLM(config, llada_model=mock_llada)

        input_ids = torch.tensor([[1, GRAPH_TOKEN_INDEX, GRAPH_TOKEN_INDEX, 4]])
        labels = torch.tensor([[1, 2, 3, 4]])
        g = make_graph(num_nodes=2, feat_dim=32, num_edges=2)

        output = model(input_ids=input_ids, labels=labels, graph_data=[g])
        output.loss.backward()

        # Projector should have gradients
        proj_grads = [p.grad for p in model.graph_projector.parameters() if p.grad is not None]
        assert len(proj_grads) > 0

        # Graph tower should NOT have gradients (frozen)
        tower_grads = [p.grad for p in model.graph_tower.parameters() if p.grad is not None]
        assert len(tower_grads) == 0

    def test_multiple_graph_types(self):
        """Test with both Graph Transformer and GCN encoders."""
        for tower_type in ["graph_transformer", "gcn"]:
            mock_llada = MockLLaDAModel(vocab_size=100, hidden_size=64)
            config = GraphLLaDAConfig(
                graph_tower=tower_type,
                graph_hidden_size=32,
                hidden_size=64,
                graph_num_layers=2,
                graph_num_heads=4,
            )
            model = GraphLLaDAForMLM(config, llada_model=mock_llada)

            input_ids = torch.tensor([[1, GRAPH_TOKEN_INDEX, GRAPH_TOKEN_INDEX, 4]])
            labels = torch.tensor([[1, 2, 3, 4]])
            g = make_graph(num_nodes=2, feat_dim=32, num_edges=2)

            output = model(input_ids=input_ids, labels=labels, graph_data=[g])
            assert output.loss is not None, f"Failed for {tower_type}"
