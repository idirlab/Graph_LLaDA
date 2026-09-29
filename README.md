# Graph-LLaDA

GNN-augmented LLaDA 8B for knowledge graph understanding. Uses a GNN encoder to embed knowledge graph structure and projects it into LLaDA's embedding space for graph-conditioned masked diffusion generation. Both the graph encoder and projector are trained from scratch on WebNLG and KGQA datasets.

## Architecture

```
Knowledge Graph (PyG Data) -> GNN Encoder -> Projector -> Token Insertion -> LLaDA 8B (Masked Diffusion LM)
```

- **GNN Encoder**: Graph Transformer (TransformerConv) or GCN via PyG standard layers, trained from scratch
- **Projector**: Linear or MLP (`mlp2x_gelu`) mapping from `graph_hidden_size` -> `llm_hidden_size`
- **LLaDA 8B**: Masked diffusion language model with bidirectional (non-causal) attention

## Project Structure

```
graph_llada/
├── __init__.py
├── constants.py                     # Token indices and special tokens
├── model/
│   ├── __init__.py                  # Clean exports
│   ├── graph_llada.py               # Main model (wrapper around LLaDA)
│   ├── config.py                    # GraphLLaDAConfig dataclass
│   ├── graph_encoder/
│   │   └── builder.py               # Encoder factory (graph_transformer, gcn)
│   └── graph_projector/
│       └── builder.py               # Projector factory (linear, mlp)
├── train/
│   └── train.py                     # Training pipeline
├── eval/                            # Evaluation scripts
└── data/
    └── graph_dataset.py             # Graph-text dataset loading
references/                          # Cloned reference repos (design patterns only)
├── LLaDA-V/                         # Encoder integration into LLaDA
├── LLaDA/                           # Base masked diffusion model
└── GraphGPT/                        # GNN encoder integration into LLM
```

## Usage

```python
from graph_llada.model import GraphLLaDA, GraphLLaDAConfig

# Create config
config = GraphLLaDAConfig(
    graph_tower="graph_transformer",
    graph_hidden_size=128,
)

# Build model (automatically loads LLaDA)
model = GraphLLaDA(config)

# Generate
output = model.generate(
    input_ids=input_ids,
    graph_data=[graph],
    steps=128,
    gen_length=128,
)
```

## Training Stages

1. **Stage 1 - Graph-text alignment**: Freeze LLM, train graph encoder + projector on graph-text pairs (WebNLG/KGQA)
2. **Stage 2 - Joint fine-tuning**: Unfreeze LLM + projector, optionally unfreeze encoder, fine-tune on instruction data

## Roadmap

### Phase 1: Text-only baselines on WebNLG

Evaluate LLaDA and autoregressive models on WebNLG graph-to-text using textualized triples as input (no GNN encoder yet).

- Load WebNLG data, design prompts with textualized graph input (e.g., `["Darlington | areaCode | 01325"]` -> natural language description)
- Run LLaDA inference on test set with two output length strategies:
  - Fixed length (e.g., 64 tokens)
  - Oracle length (match the number of tokens in the reference sentence)
- Compare with autoregressive baselines (Llama 3, Qwen3) using the same prompts

### Phase 2: Graph-LLaDA model

Train the full graph encoder -> projector -> LLaDA pipeline.

- Train graph encoder + projector on WebNLG train set, evaluate on test set
- Compare two input masking strategies:
  - Random token masking
  - Random entity/relation masking (avoids leakage from partial entity tokens)
- Compare with existing graph-LLM baselines (GraphGPT, G-Retriever) if possible

### Phase 3: Workshop paper submission

### Phase 4: KGQA and EMNLP

Extend to knowledge graph question answering for a full EMNLP paper.

## Scripts

### Run Dream-7B WebNLG generation with trajectories

```bash
python src/phase1_dream.py --split test --gen-length 64 --trajectory
```

Then analyze with Dream tokenizer:

```bash
python experiments/trajectory.py results/Dream-v0-Instruct-7B_en_test_gl64_<timestamp>_fixed.jsonl \
  --tokenizer Dream-org/Dream-v0-Instruct-7B
```

### Preview WebNLG data

```bash
python -m graph_llada.data.webnlg --split test -n 5
python -m graph_llada.data.webnlg --split train --category Airport -n 3
```

Options: `--split` (train/validation/test), `--category` (filter by DBpedia category), `-n` (number of examples to show).

## Reference Repos

These repos provide design patterns only -- we don't load their pretrained weights.

| Repo | Purpose |
|------|---------|
| [LLaDA-V](https://github.com/ML-GSAI/LLaDA-V) | How to integrate an encoder into LLaDA's embedding space |
| [LLaDA](https://github.com/ML-GSAI/LLaDA) | Base masked diffusion language model |
| [GraphGPT](https://github.com/HKUDS/GraphGPT) | How to integrate a GNN encoder into an LLM |
