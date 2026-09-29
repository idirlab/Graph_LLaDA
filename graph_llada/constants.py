"""
Constants for Graph-LLaDA.

Graph token conventions follow GraphGPT, index conventions follow LLaDA-V.
"""

# Label index for positions excluded from masked diffusion loss (graph tokens, padding, prompts)
IGNORE_INDEX = -100

# Placeholder value in input_ids that marks where graph features should be inserted
GRAPH_TOKEN_INDEX = -200

# Special tokens added to the tokenizer vocabulary
DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"

# LLaDA-specific token IDs
MASK_TOKEN_ID = 126336  # [MASK] token used in masked diffusion
