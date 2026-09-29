"""
Training script for Graph-LLaDA.

Training stages (following GraphGPT's approach):
  1. Graph-text alignment pre-training: freeze LLaDA + GNN, train projector only
  2. Graph-aware instruction tuning: fine-tune LLaDA with LoRA, train projector + GNN

See references/GraphGPT/graphgpt/train/train_graph.py for the full training pipeline.
See references/LLaDA-V/train/llava/train/ for LLaDA-V's training approach.
"""
