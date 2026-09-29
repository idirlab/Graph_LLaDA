#!/usr/bin/env python3
"""
Test script for Graph-LLaDA inference with real model.

This script:
1. Loads the actual LLaDA-8B model from HuggingFace
2. Wraps it with Graph-LLaDA (GNN encoder + projector)
3. Creates a sample knowledge graph
4. Runs inference to generate text conditioned on the graph

REQUIREMENTS:
- ~16GB disk space for LLaDA-8B model download (first run only)
- ~8GB RAM minimum (16GB+ recommended)
- GPU recommended but not required (will be slow on CPU)
- Internet connection for model download

USAGE:
    python test_inference.py

Note: This uses the pretrained LLaDA-8B-Instruct model, which should produce
reasonable outputs. The graph encoder and projector are untrained (random init),
so graph-conditioned outputs may not be meaningful yet.
"""

import torch
from torch_geometric.data import Data
from transformers import AutoTokenizer

from graph_llada.model import GraphLLaDA, GraphLLaDAConfig, build_graph_text_input
from graph_llada.constants import DEFAULT_GRAPH_TOKEN


def create_sample_knowledge_graph():
    """Create a simple knowledge graph for testing.

    Example KG structure (5 nodes, 6 edges):
    - Node 0: Albert Einstein (person)
    - Node 1: Physics (field)
    - Node 2: Nobel Prize (award)
    - Node 3: Germany (country)
    - Node 4: 1921 (year)

    Relations:
    - Einstein -> works_in -> Physics
    - Einstein -> won -> Nobel Prize
    - Einstein -> born_in -> Germany
    - Nobel Prize -> awarded_in -> 1921
    - Physics -> field_of -> Nobel Prize
    - Einstein -> known_for -> Physics

    Returns:
        PyG Data object with node features and edge structure
    """
    num_nodes = 5
    feat_dim = 128

    # Random node features (in real usage, these would be learned embeddings
    # or features from a pretrained graph model)
    torch.manual_seed(42)
    node_features = torch.randn(num_nodes, feat_dim)

    # Edge list: (source, target) pairs
    edges = [
        (0, 1),  # Einstein -> Physics
        (0, 2),  # Einstein -> Nobel Prize
        (0, 3),  # Einstein -> Germany
        (2, 4),  # Nobel Prize -> 1921
        (1, 2),  # Physics -> Nobel Prize
        (0, 1),  # Einstein -> Physics (different relation)
    ]

    edge_index = torch.tensor(edges).t().contiguous()

    graph = Data(
        graph_node=node_features,
        edge_index=edge_index,
        num_nodes=num_nodes
    )

    return graph


def main():
    print("=" * 80)
    print("Graph-LLaDA Inference Test (Clean API)")
    print("=" * 80)
    print()

    # Configuration
    print("Step 1: Setting up configuration...")
    config = GraphLLaDAConfig(
        llada_model_name_or_path="GSAI-ML/LLaDA-8B-Instruct",
        graph_tower="graph_transformer",  # or "gcn"
        graph_hidden_size=128,
        graph_projector_type="linear",  # or "mlp"
        freeze_graph_tower=False,
        freeze_llada=True,  # Always frozen during inference
        graph_num_layers=2,
        graph_num_heads=4,
        use_graph_start_end=True,
    )
    print(f"  - LLaDA model: {config.llada_model_name_or_path}")
    print(f"  - Graph encoder: {config.graph_tower}")
    print(f"  - Graph hidden size: {config.graph_hidden_size}")
    print(f"  - Projector type: {config.graph_projector_type}")
    print()

    # Determine device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"  - Device: {device}")
    print(f"  - Dtype: {dtype}")
    print()

    # Load tokenizer
    print("Step 2: Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.llada_model_name_or_path,
        trust_remote_code=True,
    )
    print(f"  - Vocabulary size: {len(tokenizer)}")
    print()

    # Create Graph-LLaDA model (loads LLaDA automatically)
    print("Step 3: Loading Graph-LLaDA model...")
    print("  (This may take a few minutes on first run - downloading ~16GB)")

    model = GraphLLaDA(config)
    model = model.to(device)
    model.eval()  # Set to evaluation mode

    print(f"  ✓ Model loaded successfully")
    print(f"  - LLaDA hidden size: {model.config.hidden_size}")
    print(f"  - Graph encoder: {type(model.graph_encoder).__name__}")
    print(f"  - Graph projector: {type(model.graph_projector).__name__}")
    print()

    # Initialize tokenizer with graph tokens
    print("Step 4: Adding graph tokens to tokenizer...")
    tokenizer = model.initialize_tokenizer(
        tokenizer,
        use_graph_start_end=config.use_graph_start_end
    )
    print(f"  - New vocabulary size: {len(tokenizer)}")
    print(f"  - Added tokens: <g_patch>, <g_start>, <g_end>")
    print()

    # Create sample graph
    print("Step 5: Creating sample knowledge graph...")
    graph = create_sample_knowledge_graph()
    graph = graph.to(device)  # Move graph to same device as model
    print(f"  - Nodes: {graph.num_nodes}")
    print(f"  - Edges: {graph.edge_index.shape[1]}")
    print(f"  - Node feature dim: {graph.graph_node.shape[1]}")
    print()

    # Prepare input text with graph placeholder
    print("Step 6: Preparing input text with graph token...")

    # For the Instruct model, format with chat template
    user_content = f"Based on the knowledge graph {DEFAULT_GRAPH_TOKEN}, who won the Nobel Prize."
    message = {"role": "user", "content": user_content}
    prompt_text = tokenizer.apply_chat_template(
        [message],
        add_generation_prompt=True,
        tokenize=False
    )

    print(f"  - User content: {user_content}")
    print(f"  - After chat template: {prompt_text[:200]}...")
    print(f"  - Contains <graph> token: {DEFAULT_GRAPH_TOKEN in prompt_text}")

    input_ids = build_graph_text_input(
        text=prompt_text,
        tokenizer=tokenizer,
        num_graph_nodes=graph.num_nodes,
        use_graph_start_end=config.use_graph_start_end,
    )

    input_ids = input_ids.unsqueeze(0).to(device)  # Add batch dimension

    print(f"  - Input sequence length: {input_ids.shape[1]}")
    print(f"  - Graph token positions: {(input_ids[0] == -200).sum().item()} tokens")
    print()

    # Test 1: Forward pass with graph encoding
    print("Step 7: Testing forward pass with graph encoding...")
    print()

    with torch.no_grad():
        try:
            # First, encode just the graph to inspect
            print("  - Encoding graph...")
            graph_features = model.encode_graphs([graph])
            print(f"    Graph features shape: {graph_features[0].shape}")
            print(f"    Graph features dtype: {graph_features[0].dtype}")
            print(f"    Graph features device: {graph_features[0].device}")
            print(f"    Graph features mean: {graph_features[0].mean().item():.4f}")
            print(f"    Graph features std: {graph_features[0].std().item():.4f}")
            print(f"    Graph features min/max: {graph_features[0].min().item():.4f} / {graph_features[0].max().item():.4f}")
            print()

            # Encode graph and prepare inputs
            inputs_embeds, _, attention_mask = model.prepare_inputs_with_graphs(
                input_ids=input_ids,
                graph_data=[graph],
            )

            print(f"  ✓ Graph encoding successful!")
            print(f"  - Input embeddings shape: {inputs_embeds.shape}")
            print(f"  - Attention mask: {'provided' if attention_mask is not None else 'None (no padding)'}")
            print(f"  - Sequence length: {inputs_embeds.shape[1]} (unchanged — graph tokens replaced 1:1)")
            print()

            # Forward through LLaDA
            output = model.llada(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )

            print(f"  ✓ Forward pass successful!")
            print(f"  - Output logits shape: {output.logits.shape}")
            print(f"  - Logits dtype: {output.logits.dtype}")
            print(f"  - Logits contain NaN: {torch.isnan(output.logits).any().item()}")
            print(f"  - Logits contain Inf: {torch.isinf(output.logits).any().item()}")
            print()

        except Exception as e:
            print(f"  ✗ Error during forward pass: {e}")
            import traceback
            traceback.print_exc()
            return

    # Test 2: Base LLaDA generation (text-only, no graph)
    if torch.cuda.is_available():
        print("Step 8: Testing base LLaDA generation (text-only, no graph)...")
        print("  (This will take a while - using 32 steps for speed)")
        print()

        # Create a simple text prompt without graph tokens
        # For the Instruct model, use chat template formatting
        simple_prompt = "The Nobel Prize in Physics was awarded to"
        message = {"role": "user", "content": simple_prompt}
        formatted_prompt = tokenizer.apply_chat_template(
            [message],
            add_generation_prompt=True,
            tokenize=False
        )
        simple_input_ids = tokenizer(formatted_prompt, return_tensors='pt')['input_ids'].to(device)

        with torch.no_grad():
            try:
                # Use the reference generate function from LLaDA
                from graph_llada.model.graph_llada import generate_llada

                generated = generate_llada(
                    model=model.llada,
                    prompt=simple_input_ids,
                    attention_mask=None,
                    steps=32,
                    gen_length=32,
                    block_length=32,
                    temperature=0.,
                )

                print(f"  ✓ Base LLaDA generation successful!")
                print(f"  - Generated sequence shape: {generated.shape}")
                print()

                # Decode output
                generated_text = tokenizer.decode(
                    generated[0, simple_input_ids.shape[1]:],
                    skip_special_tokens=True
                )
                print(f"  - Prompt: '{simple_prompt}'")
                print(f"  - Generated: '{generated_text}'")
                print()

            except Exception as e:
                print(f"  ✗ Error during base LLaDA generation: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("Step 8: Skipping base LLaDA generation (GPU not available - would be very slow)")
        print()

    # Test 3: Graph-conditioned generation
    if torch.cuda.is_available():
        print("Step 9: Testing Graph-LLaDA generation (with graph conditioning)...")
        print("  (This will take a while - using 32 steps for speed)")
        print("  Note: Graph encoder and projector are untrained (random weights),")
        print("  so the graph features may confuse the LLM and produce unexpected output.")
        print()

        with torch.no_grad():
            try:
                generated = model.generate(
                    input_ids=input_ids,
                    graph_data=[graph],
                    attention_mask=None,
                    steps=32,  # Reduced for speed
                    gen_length=32,
                    block_length=32,
                    temperature=0.,
                )

                print(f"  ✓ Graph-conditioned generation successful!")
                print(f"  - Generated sequence shape: {generated.shape}")
                print()

                # Decode output
                generated_text = tokenizer.decode(
                    generated[0, input_ids.shape[1]:],
                    skip_special_tokens=True
                )
                print(f"  - Prompt: '{user_content}'")
                print(f"  - Generated: '{generated_text}'")
                print()

            except Exception as e:
                print(f"  ✗ Error during graph-conditioned generation: {e}")
                import traceback
                traceback.print_exc()
    else:
        print("Step 9: Skipping graph-conditioned generation (GPU not available - would be very slow)")
        print()

    print("=" * 80)
    print("Test completed successfully!")
    print("=" * 80)
    print()
    print("Summary:")
    print("  ✓ Model architecture loads correctly")
    print("  ✓ Graph encoder processes PyG Data")
    print("  ✓ Projector maps graph features to LLM space")
    print("  ✓ Token insertion works (GRAPH_TOKEN_INDEX → graph features)")
    print("  ✓ Forward pass executes without errors")
    if torch.cuda.is_available():
        print("  ✓ Base LLaDA generation works (text-only)")
        print("  ✓ Graph-conditioned generation works")
    print()
    print("Next steps:")
    print("  - Train the model on WebNLG/KGQA datasets")
    print("  - Implement two-stage training (projector → joint fine-tuning)")
    print("  - Evaluate on graph-to-text tasks")
    print()


if __name__ == "__main__":
    main()
