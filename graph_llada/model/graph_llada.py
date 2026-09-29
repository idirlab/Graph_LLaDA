"""
Graph-LLaDA: Clean wrapper that combines GNN encoder + LLaDA masked diffusion model.

Simple architecture:
    Graph (PyG Data) -> GNN Encoder -> Projector -> Token Insertion -> LLaDA (MDM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from torch_geometric.data import Data
from transformers import AutoModel

from .graph_encoder.builder import build_graph_encoder
from .graph_projector.builder import build_graph_projector
from .config import GraphLLaDAConfig
from ..constants import (
    GRAPH_TOKEN_INDEX,
    IGNORE_INDEX,
    MASK_TOKEN_ID,
    DEFAULT_GRAPH_TOKEN,
    DEFAULT_GRAPH_PATCH_TOKEN,
    DEFAULT_G_START_TOKEN,
    DEFAULT_G_END_TOKEN,
)


class GraphLLaDA(nn.Module):
    """Graph-LLaDA model: GNN encoder + projector + LLaDA.

    Args:
        config: GraphLLaDAConfig with model parameters
        llada_model: Pre-loaded LLaDA model (optional, will load from config if not provided)
    """

    def __init__(self, config: GraphLLaDAConfig, llada_model=None):
        super().__init__()
        self.config = config

        # Load LLaDA model
        if llada_model is None:
            llada_model = AutoModel.from_pretrained(
                config.llada_model_name_or_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )

        # Update config with LLaDA's hidden size
        if hasattr(llada_model.config, 'hidden_size'):
            config.update_from_llada_config(llada_model.config)

        self.llada = llada_model

        if config.freeze_llada:
            self.llada.requires_grad_(False)

        # Build graph modules
        self.graph_encoder = build_graph_encoder(config)
        self.graph_projector = build_graph_projector(config)

        # Input projections: LLaDA embed dim -> graph encoder dim
        # Used when node/edge features are initialized from LLaDA token embeddings
        self.node_input_proj = nn.Linear(config.hidden_size, config.graph_hidden_size)
        self.edge_input_proj = (
            nn.Linear(config.hidden_size, config.graph_edge_dim)
            if config.graph_edge_dim is not None else None
        )

        # Freeze graph encoder if specified
        if config.freeze_graph_tower:
            self.graph_encoder.requires_grad_(False)

    @property
    def embed_tokens(self):
        """Access LLaDA's token embedding layer."""
        return self.llada.get_input_embeddings()

    def encode_graphs(self, graph_data: List[Data]) -> List[torch.Tensor]:
        """Encode graphs through GNN encoder and projector.

        If graph_data contains node_token_ids (tokenized entity names), computes
        node features by embedding + mean-pooling through LLaDA's token embeddings,
        then projecting down to graph_hidden_size. Same for edge features.

        Args:
            graph_data: List of PyG Data objects

        Returns:
            List of projected node features [num_nodes, hidden_size]
        """
        results = []
        for g in graph_data:
            if hasattr(g, 'node_token_ids') and g.node_token_ids is not None:
                device = self.node_input_proj.weight.device
                dtype = self.node_input_proj.weight.dtype

                # Embed entity tokens and mean-pool per node
                tok_ids = g.node_token_ids.to(device)
                tok_mask = g.node_token_mask.to(device).unsqueeze(-1).to(dtype)
                embeds = self.embed_tokens(tok_ids).to(dtype)
                g.graph_node = self.node_input_proj(
                    (embeds * tok_mask).sum(1) / tok_mask.sum(1).clamp(min=1)
                )

                # Embed relation tokens for edge features
                if (self.edge_input_proj is not None
                        and hasattr(g, 'edge_token_ids')
                        and g.edge_token_ids.numel() > 0):
                    e_ids = g.edge_token_ids.to(device)
                    e_mask = g.edge_token_mask.to(device).unsqueeze(-1).to(dtype)
                    e_embeds = self.embed_tokens(e_ids).to(dtype)
                    g.edge_attr = self.edge_input_proj(
                        (e_embeds * e_mask).sum(1) / e_mask.sum(1).clamp(min=1)
                    )

            results.append(self.graph_projector(self.graph_encoder(g)))
        return results

    def prepare_inputs_with_graphs(
        self,
        input_ids: torch.LongTensor,
        graph_data: Optional[List[Data]] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """Replace GRAPH_TOKEN_INDEX placeholders with encoded graph features.

        Graph tokens are inserted 1:1 with graph nodes (one <g_patch> per node),
        so the sequence length does not change — graph positions are simply
        overwritten with projected node embeddings.

        Args:
            input_ids: [batch, seq_len] with GRAPH_TOKEN_INDEX at graph positions
            graph_data: List of PyG Data objects (one per batch element with graph tokens)
            labels: [batch, seq_len] training labels (optional)
            attention_mask: [batch, seq_len] attention mask (optional)

        Returns:
            Tuple of (inputs_embeds, labels, attention_mask) with graph features inserted
        """
        # Embed all tokens (replace GRAPH_TOKEN_INDEX sentinel with 0 before lookup;
        # those positions will be overwritten with graph features below)
        clean_ids = input_ids.clone()
        clean_ids[clean_ids == GRAPH_TOKEN_INDEX] = 0
        inputs_embeds = self.embed_tokens(clean_ids).clone()

        if not graph_data:
            return inputs_embeds, labels, attention_mask

        graph_features = self.encode_graphs(graph_data)
        device, dtype = inputs_embeds.device, inputs_embeds.dtype
        new_labels = labels.clone() if labels is not None else None

        graph_idx = 0
        for b in range(input_ids.shape[0]):
            mask = (input_ids[b] == GRAPH_TOKEN_INDEX)
            if not mask.any():
                continue

            cur_graph = graph_features[graph_idx].to(device=device, dtype=dtype)
            graph_idx += 1

            num_tokens = mask.sum().item()
            if num_tokens != cur_graph.shape[0]:
                raise ValueError(
                    f"Sample {b}: {num_tokens} graph tokens but {cur_graph.shape[0]} graph nodes. "
                    f"Ensure build_graph_text_input uses the correct num_graph_nodes."
                )

            # Validate contiguity
            positions = mask.nonzero(as_tuple=True)[0]
            if positions[-1] - positions[0] + 1 != num_tokens:
                raise ValueError(f"Sample {b}: graph tokens must be contiguous in the sequence.")

            # Replace graph token positions with projected graph features
            inputs_embeds[b, mask] = cur_graph
            if new_labels is not None:
                new_labels[b, mask] = IGNORE_INDEX

        if graph_idx != len(graph_features):
            raise ValueError(
                f"Expected {len(graph_features)} graphs but only {graph_idx} samples had graph tokens."
            )

        return inputs_embeds, new_labels, attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor,
        graph_data: Optional[List[Data]] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Forward pass with graph-conditioned masked diffusion.

        When `labels` is provided (training), applies the MDM forward process
        (embedding-space masking) and computes the ELBO loss manually.
        This follows the base LLaDA usage pattern where masking is applied
        externally before the model call (see references/LLaDA/get_log_likelihood.py
        and references/LLaDA-V/train/llava/model/language_model/modeling_llada.py:1544).

        When `labels` is None (inference), passes clean embeddings through for logits.

        Args:
            input_ids: [batch, seq_len] token IDs with GRAPH_TOKEN_INDEX placeholders
            graph_data: List of PyG Data objects
            labels: [batch, seq_len] target labels (IGNORE_INDEX=-100 for graph/prompt tokens)
            attention_mask: [batch, seq_len] attention mask
            **kwargs: Additional arguments for LLaDA

        Returns:
            LLaDA model output. If labels provided, output.loss is the MDM ELBO loss.
        """
        inputs_embeds, labels, attention_mask = self.prepare_inputs_with_graphs(
            input_ids, graph_data, labels, attention_mask
        )

        if labels is None:
            # Inference: pass clean embeddings directly
            return self.llada(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **kwargs,
            )

        # Training: apply MDM forward process in embedding space.
        # Only mask positions with valid labels (not IGNORE_INDEX).
        # Graph tokens and prompt tokens are labeled IGNORE_INDEX and never masked.
        b, l, _ = inputs_embeds.shape
        device = inputs_embeds.device

        # Sample a noise level per batch element (linear noise schedule, LLaDA eq. 8)
        t = torch.rand(b, device=device)
        p_mask = (1 - 1e-3) * t + 1e-3          # [b], masking probability in [eps, 1]
        p_mask_per_token = p_mask[:, None].expand(b, l)  # [b, l]

        # Randomly mask valid (answer) positions
        valid = (labels != IGNORE_INDEX)                              # [b, l]
        masked_indices = (torch.rand(b, l, device=device) < p_mask_per_token) & valid

        # Replace masked positions with the MASK token embedding
        mask_embed = self.embed_tokens(
            torch.tensor([MASK_TOKEN_ID], device=device)
        )  # [1, hidden_size]
        noisy_embeds = torch.where(
            masked_indices.unsqueeze(-1),
            mask_embed.view(1, 1, -1).expand(b, l, -1),
            inputs_embeds,
        )

        # Forward through LLaDA on noisy embeddings (labels not passed — loss computed below)
        outputs = self.llada(
            inputs_embeds=noisy_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )
        logits = outputs.logits

        # ELBO loss: cross-entropy on masked positions, weighted by 1/p_mask
        # (LLaDA-V normalization: divide by #valid tokens per sample, then average over batch)
        if masked_indices.any():
            token_loss = F.cross_entropy(
                logits[masked_indices],
                labels[masked_indices],
                reduction='none',
            ) / p_mask_per_token[masked_indices]

            valid_count = valid.sum(dim=-1, keepdim=True).float().clamp(min=1)  # [b, 1]
            loss = (token_loss / valid_count.expand(b, l)[masked_indices]).sum() / b
        else:
            # Zero loss that still connects to all parameters (including LoRA)
            # for DDP gradient reduction compatibility
            loss = logits.sum() * 0.

        outputs.loss = loss
        return loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        graph_data: Optional[List[Data]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        steps: int = 128,
        gen_length: int = 128,
        block_length: int = 128,
        temperature: float = 0.,
        cfg_scale: float = 0.,
        remasking: str = 'low_confidence',
        alpha: float = 1.0,
        structural_token_ids: Optional[set] = None,
        return_trajectory: bool = False,
        **kwargs,
    ):
        """Generate text conditioned on graph using masked diffusion.

        Args:
            input_ids: [batch, seq_len] prompt with GRAPH_TOKEN_INDEX placeholders
            graph_data: List of PyG Data objects
            attention_mask: [batch, seq_len] attention mask
            steps: Number of diffusion steps per block
            gen_length: Total length to generate
            block_length: Block size for semi-autoregressive generation
            temperature: Gumbel noise temperature
            cfg_scale: Classifier-free guidance scale
            remasking: Remasking strategy ('low_confidence' or 'random')
            alpha: Confidence scaling for structural tokens (1.0 = no scaling)
            structural_token_ids: Set of token IDs for ".", EOS, EOT (used with alpha < 1)
            return_trajectory: If True, return (x, trajectory) with unmasking order

        Returns:
            Generated token IDs [batch, seq_len + gen_length], or
            (token_ids, trajectory) if return_trajectory=True
        """
        prompt_embeds, _, prompt_mask = self.prepare_inputs_with_graphs(
            input_ids, graph_data, None, attention_mask
        )

        batch_size = prompt_embeds.shape[0]
        prompt_len = prompt_embeds.shape[1]
        device = prompt_embeds.device

        x = torch.full((batch_size, prompt_len + gen_length), MASK_TOKEN_ID,
                       dtype=torch.long, device=device)
        x[:, :prompt_len] = 0  # sentinel — prompt_embeds used instead

        if prompt_mask is not None:
            full_mask = torch.cat([
                prompt_mask,
                torch.ones((batch_size, gen_length), dtype=prompt_mask.dtype, device=device)
            ], dim=1)
        else:
            full_mask = None

        # Alpha scaling setup
        structural_set = (
            torch.tensor(sorted(structural_token_ids), device=device)
            if alpha < 1.0 and structural_token_ids else None
        )

        trajectory = [[] for _ in range(batch_size)] if return_trajectory else None

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks

        for block_idx in range(num_blocks):
            block_start = prompt_len + block_idx * block_length
            block_end = prompt_len + (block_idx + 1) * block_length

            block_mask_index = (x[:, block_start:block_end] == MASK_TOKEN_ID)
            num_transfer_tokens = _get_num_transfer_tokens(block_mask_index, steps_per_block)

            for step in range(steps_per_block):
                mask_index = (x == MASK_TOKEN_ID)
                mask_index[:, :prompt_len] = False

                x_embeds = torch.cat([prompt_embeds, self.embed_tokens(x[:, prompt_len:])], dim=1)

                if cfg_scale > 0.:
                    un_embeds = x_embeds.clone()
                    un_embeds[:, :prompt_len] = self.embed_tokens(
                        torch.full((batch_size, prompt_len), MASK_TOKEN_ID,
                                   dtype=torch.long, device=device)
                    )
                    combined_embeds = torch.cat([x_embeds, un_embeds], dim=0)
                    combined_mask = torch.cat([full_mask, full_mask], dim=0) if full_mask is not None else None
                    logits = self.llada(inputs_embeds=combined_embeds, attention_mask=combined_mask).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = self.llada(inputs_embeds=x_embeds, attention_mask=full_mask).logits

                logits_with_noise = _add_gumbel_noise(logits, temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if remasking == 'low_confidence':
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
                elif remasking == 'random':
                    x0_p = torch.rand_like(x0, dtype=torch.float)
                else:
                    raise ValueError(f"Unknown remasking strategy: {remasking}")

                # Alpha scaling: reduce confidence of structural tokens
                if structural_set is not None:
                    is_structural = (x0.unsqueeze(-1) == structural_set).any(-1)
                    x0_p = torch.where(is_structural, x0_p * alpha, x0_p)

                x0_p[:, block_end:] = -float('inf')
                x0_p[:, :prompt_len] = -float('inf')

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -float('inf'))

                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for b in range(batch_size):
                    _, select_index = torch.topk(confidence[b], k=num_transfer_tokens[b, step])
                    transfer_index[b, select_index] = True
                    if trajectory is not None:
                        positions = (select_index - prompt_len).tolist()
                        tokens = x0[b, select_index].tolist()
                        trajectory[b].extend(zip(positions, tokens))

                x[transfer_index] = x0[transfer_index]

        return (x, trajectory) if return_trajectory else x

    def graph_state_dict(self):
        """Extract graph-specific parameters (encoder, projector, input projections)."""
        prefixes = ("graph_encoder.", "graph_projector.", "node_input_proj.", "edge_input_proj.")
        return {k: v for k, v in self.state_dict().items() if any(k.startswith(p) for p in prefixes)}

    def load_graph_state_dict(self, state_dict, strict=True):
        """Load graph-specific parameters from a checkpoint."""
        missing, unexpected = [], []
        own_state = self.state_dict()
        for k, v in state_dict.items():
            if k in own_state:
                own_state[k].copy_(v)
            else:
                unexpected.append(k)
        if strict:
            prefixes = ("graph_encoder.", "graph_projector.", "node_input_proj.", "edge_input_proj.")
            for k in own_state:
                if any(k.startswith(p) for p in prefixes) and k not in state_dict:
                    missing.append(k)
            if missing or unexpected:
                raise RuntimeError(f"Missing keys: {missing}, Unexpected keys: {unexpected}")
        return missing, unexpected

    def initialize_tokenizer(self, tokenizer, use_graph_start_end=True):
        """Add graph tokens to tokenizer and resize embeddings.

        Args:
            tokenizer: HuggingFace tokenizer to extend
            use_graph_start_end: Whether to add <g_start>/<g_end> tokens

        Returns:
            Updated tokenizer
        """
        tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
        self.llada.resize_token_embeddings(len(tokenizer))

        if use_graph_start_end:
            num_new = tokenizer.add_tokens(
                [DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN],
                special_tokens=True
            )
            self.llada.resize_token_embeddings(len(tokenizer))

            # Initialize new token embeddings with mean of existing embeddings
            if num_new > 0:
                input_embeddings = self.llada.get_input_embeddings().weight.data
                output_embeddings = self.llada.get_output_embeddings().weight.data

                input_avg = input_embeddings[:-num_new].mean(dim=0, keepdim=True)
                output_avg = output_embeddings[:-num_new].mean(dim=0, keepdim=True)

                input_embeddings[-num_new:] = input_avg
                output_embeddings[-num_new:] = output_avg

        return tokenizer


def build_graph_text_input(
    text: str,
    tokenizer,
    num_graph_nodes: int,
    use_graph_start_end: bool = True,
):
    """Build input_ids with graph token placeholders.

    Replaces <graph> in text with graph token pattern and returns input_ids
    with GRAPH_TOKEN_INDEX at graph positions.

    Args:
        text: Input text containing "<graph>" marker
        tokenizer: Tokenizer with graph tokens added
        num_graph_nodes: Number of nodes in the graph
        use_graph_start_end: Whether to use <g_start>/<g_end> delimiters

    Returns:
        input_ids tensor with GRAPH_TOKEN_INDEX at graph positions
    """
    patch_tokens = DEFAULT_GRAPH_PATCH_TOKEN * num_graph_nodes
    if use_graph_start_end:
        replace_token = DEFAULT_G_START_TOKEN + patch_tokens + DEFAULT_G_END_TOKEN
    else:
        replace_token = patch_tokens

    text = text.replace(DEFAULT_GRAPH_TOKEN, replace_token)
    input_ids = tokenizer(text, return_tensors='pt')['input_ids'][0]

    g_patch_id = tokenizer.convert_tokens_to_ids(DEFAULT_GRAPH_PATCH_TOKEN)
    input_ids[input_ids == g_patch_id] = GRAPH_TOKEN_INDEX

    return input_ids


@torch.no_grad()
def generate_llada(
    model,
    prompt: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.,
    cfg_scale: float = 0.,
    remasking: str = 'low_confidence',
    return_trajectory: bool = False,
):
    """Generate text using LLaDA masked diffusion (reference implementation).

    Standalone function for base LLaDA generation (no graph conditioning),
    adapted from references/LLaDA/generate.py.

    Args:
        model: LLaDA model (not GraphLLaDA)
        prompt: [batch, prompt_len] input token IDs
        attention_mask: [batch, prompt_len] attention mask
        steps: Number of diffusion steps per block
        gen_length: Total length to generate
        block_length: Block size for semi-autoregressive generation
        temperature: Gumbel noise temperature
        cfg_scale: Classifier-free guidance scale
        remasking: Remasking strategy ('low_confidence' or 'random')

    Returns:
        Generated token IDs [batch, prompt_len + gen_length]
    """
    device = prompt.device
    batch_size = prompt.shape[0]
    prompt_len = prompt.shape[1]

    x = torch.full(
        (batch_size, prompt_len + gen_length),
        MASK_TOKEN_ID,
        dtype=torch.long,
        device=device,
    )
    x[:, :prompt_len] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((batch_size, gen_length), dtype=attention_mask.dtype, device=device)
        ], dim=-1)

    prompt_index = (x != MASK_TOKEN_ID)
    trajectory = [[] for _ in range(batch_size)] if return_trajectory else None

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end = prompt_len + (block_idx + 1) * block_length

        block_mask_index = (x[:, block_start:block_end] == MASK_TOKEN_ID)
        num_transfer_tokens = _get_num_transfer_tokens(block_mask_index, steps_per_block)

        for step in range(steps_per_block):
            mask_index = (x == MASK_TOKEN_ID)

            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = MASK_TOKEN_ID
                x_ = torch.cat([x, un_x], dim=0)
                attn_mask_ = torch.cat([attention_mask, attention_mask], dim=0) if attention_mask is not None else None

                logits = model(x_, attention_mask=attn_mask_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, attention_mask=attention_mask).logits

            logits_with_noise = _add_gumbel_noise(logits, temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
            elif remasking == 'random':
                x0_p = torch.rand_like(x0, dtype=torch.float)
            else:
                raise ValueError(f"Unknown remasking: {remasking}")

            x0_p[:, block_end:] = -float('inf')

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -float('inf'))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for b in range(batch_size):
                _, select_index = torch.topk(confidence[b], k=num_transfer_tokens[b, step])
                transfer_index[b, select_index] = True
                if trajectory is not None:
                    positions = (select_index - prompt_len).tolist()
                    tokens = x0[b, select_index].tolist()
                    trajectory[b].extend(zip(positions, tokens))

            x[transfer_index] = x0[transfer_index]

    return (x, trajectory) if return_trajectory else x


def _add_gumbel_noise(logits, temperature):
    """Add Gumbel noise to logits for sampling (from references/LLaDA/generate.py)."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _get_num_transfer_tokens(mask_index, steps):
    """Compute number of tokens to transfer at each step (from references/LLaDA/generate.py)."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer = torch.zeros(
        mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
    ) + base
    for i in range(mask_num.size(0)):
        num_transfer[i, :remainder[i]] += 1
    return num_transfer
