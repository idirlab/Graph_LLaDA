"""
WebNLG graph-text dataset for Graph-LLaDA training.

Converts RDF triples into PyG Data objects with tokenized entity/relation names
(for embedding initialization from LLaDA token embeddings) and builds input_ids
with GRAPH_TOKEN_INDEX placeholders for graph token insertion.

Usage:
    from graph_llada.data.graph_dataset import WebNLGGraphDataset, collate_graph_batch

    ds = WebNLGGraphDataset(tokenizer, split="train")
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_graph_batch)
"""

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from datasets import load_dataset

from graph_llada.constants import IGNORE_INDEX
from graph_llada.model.graph_llada import build_graph_text_input

GRAPH_PROMPT_TEMPLATE = (
    "You are given a knowledge graph. "
    "Generate a single natural-language sentence that describes the "
    "information in the graph.\n\n"
    "Requirements:\n"
    "- The sentence must be fully consistent with the graph.\n"
    "- The sentence must cover all facts in the graph.\n"
    "- The sentence must not add any facts not present in the graph.\n"
    "- Reuse the entity and relation names from the triples when possible.\n"
    "- Only output the sentence and nothing else.\n\n"
    "Triples:\n"
    "{flat_graph_text}\n\n"
    "Graph tokens: <graph>\n"
)


def parse_triples(triples):
    """Parse WebNLG RDF triples into (entity_dict, edge_list).

    Returns:
        entities: dict mapping entity text -> node index
        edges: list of (src_idx, dst_idx, relation_text)
    """
    entities = {}
    edges = []
    for triple in triples:
        parts = [p.strip() for p in triple.split("|")]
        subj = parts[0] if parts else triple.strip()
        rel = parts[1] if len(parts) > 1 else "related_to"
        obj = "|".join(parts[2:]).strip() if len(parts) > 2 else ""
        for ent in (subj, obj):
            if ent and ent not in entities:
                entities[ent] = len(entities)
        if subj and obj:
            edges.append((entities[subj], entities[obj], rel))
    return entities, edges


def tokenize_and_pad(texts, tokenizer, max_tokens=32):
    """Tokenize texts, truncate, pad to uniform length. Returns (ids, mask) tensors."""
    all_ids = [
        tokenizer(t.replace("_", " ").strip('"'), add_special_tokens=False)["input_ids"][:max_tokens]
        for t in texts
    ]
    max_len = max((len(ids) for ids in all_ids), default=1)
    ids = torch.zeros(len(all_ids), max_len, dtype=torch.long)
    mask = torch.zeros(len(all_ids), max_len, dtype=torch.bool)
    for i, tok_ids in enumerate(all_ids):
        ids[i, :len(tok_ids)] = torch.tensor(tok_ids)
        mask[i, :len(tok_ids)] = True
    return ids, mask


def flatten_graph_text(triples):
    """Render triples as a bracketed tuple list for stronger lexical grounding."""
    rendered = []
    for idx, triple in enumerate(triples):
        parts = [p.strip() for p in triple.split("|")]
        subj = parts[0] if parts else triple.strip()
        rel = parts[1] if len(parts) > 1 else "related_to"
        obj = "|".join(parts[2:]).strip() if len(parts) > 2 else ""
        suffix = "," if idx < len(triples) - 1 else ""
        rendered.append(f"  ({subj}, {rel}, {obj}){suffix}")
    if not rendered:
        rendered = ["  [EMPTY]"]
    return "[\n" + "\n".join(rendered) + "\n]"


class WebNLGGraphDataset(Dataset):
    """WebNLG dataset producing PyG graph objects + tokenized text for Graph-LLaDA.

    Each example contains:
        - input_ids: prompt tokens with GRAPH_TOKEN_INDEX at graph positions
        - target_ids: tokenized reference sentence
        - graph_data: PyG Data with node_token_ids, edge_index, edge_token_ids
    """

    def __init__(self, tokenizer, split="train", lang="en",
                 max_seq_len=512, max_samples=-1, max_node_tokens=32):
        self.examples = []

        data = load_dataset("GEM/web_nlg", lang, split=split)
        if max_samples > 0:
            data = data.select(range(min(max_samples, len(data))))

        for row in data:
            triples = list(row["input"])
            entities, edges = parse_triples(triples)
            if not entities:
                continue

            num_nodes = len(entities)
            entity_texts = [None] * num_nodes
            for ent, idx in entities.items():
                entity_texts[idx] = ent

            node_token_ids, node_token_mask = tokenize_and_pad(
                entity_texts, tokenizer, max_node_tokens
            )

            if edges:
                src, dst, rel_texts = zip(*[(s, d, r) for s, d, r in edges])
                edge_index = torch.tensor([list(src), list(dst)], dtype=torch.long)
                edge_token_ids, edge_token_mask = tokenize_and_pad(
                    list(rel_texts), tokenizer, max_node_tokens
                )
            else:
                edge_index = torch.zeros(2, 0, dtype=torch.long)
                edge_token_ids = torch.zeros(0, 1, dtype=torch.long)
                edge_token_mask = torch.zeros(0, 1, dtype=torch.bool)

            prompt_text = GRAPH_PROMPT_TEMPLATE.format(
                flat_graph_text=flatten_graph_text(triples)
            )
            rendered_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                add_generation_prompt=True, tokenize=False,
            )

            # Build input_ids with GRAPH_TOKEN_INDEX placeholders
            input_ids = build_graph_text_input(rendered_prompt, tokenizer, num_nodes)

            # Tokenize target
            target_ids = tokenizer(row["target"], add_special_tokens=False)["input_ids"]

            total = len(input_ids) + len(target_ids)
            if total > max_seq_len:
                overflow = total - max_seq_len
                if overflow >= len(target_ids):
                    continue
                target_ids = target_ids[:len(target_ids) - overflow]

            self.examples.append({
                "input_ids": input_ids,
                "target_ids": torch.tensor(target_ids, dtype=torch.long),
                "graph_data": Data(
                    node_token_ids=node_token_ids,
                    node_token_mask=node_token_mask,
                    edge_index=edge_index,
                    edge_token_ids=edge_token_ids,
                    edge_token_mask=edge_token_mask,
                    num_nodes=num_nodes,
                ),
                "triples": triples,
                "target": row["target"],
                "gem_id": row.get("gem_id", ""),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_graph_batch(batch, pad_id=0):
    """Collate graph-text examples into a padded batch.

    Returns dict with:
        input_ids: [B, max_len] with GRAPH_TOKEN_INDEX at graph positions
        labels: [B, max_len] with IGNORE_INDEX for prompt/graph, target ids for answer
        attention_mask: [B, max_len]
        graph_data: list of PyG Data objects
    """
    max_len = max(len(ex["input_ids"]) + len(ex["target_ids"]) for ex in batch)

    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    graph_data = []

    for i, ex in enumerate(batch):
        p_ids = ex["input_ids"]
        t_ids = ex["target_ids"]
        p_len = len(p_ids)
        seq_len = p_len + len(t_ids)

        input_ids[i, :p_len] = p_ids
        input_ids[i, p_len:seq_len] = t_ids
        labels[i, p_len:seq_len] = t_ids
        attention_mask[i, :seq_len] = 1
        graph_data.append(ex["graph_data"])

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "graph_data": graph_data,
    }
