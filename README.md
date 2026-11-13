# What Gets Unmasked First? Trajectory Analysis of Diffusion Models for Graph-to-Text Generation

**Accepted to Findings of EMNLP 2026**



[Qing Wang](https://scholar.google.com/citations?user=TYo5kK4AAAAJ&hl=en), [Jacob Devasier](https://scholar.google.com/citations?user=hQvkzsQAAAAJ&hl=en&oi=sra), [Chengkai Li](https://scholar.google.com/citations?hl=en&user=v8ZQDf8AAAAJ)

[[Paper]](https://arxiv.org/abs/2605.31564)

abstract

> We present the first systematic study of masked diffusion language models (MDLMs) for graph-to-text generation. We analyze MDLM generation trajectories -- the order in which tokens are unmasked during iterative decoding -- and find that, unlike autoregressive LLMs which generate text linearly, MDLMs naturally prioritize entities first, followed by relational and function words, with structural tokens resolved last. We further identify a previously undocumented failure mode of supervised fine-tuning: SFT disrupts this strategy by prematurely anchoring structural sentence-ending tokens early in the decoding trajectory, effectively fixing the output length which can lead to omitted or hallucinated information. To address this, we propose lambda-scaled structural decoding, a training-free inference-time modification that downweights structural token confidence and recovers +9.4 BLEU-4. Finally, we introduce Graph-LLaDA, which integrates a Graph Transformer encoder into LLaDA's decoding process to explicitly incorporate relational graph structure. Cross-dataset evaluation on LAGRANGE reveals that previous baselines overfit to dataset-specific patterns, while LLM- and MDLM-based approaches generalize significantly better.

## Dataset

We use **WebNLG** and **LAGRANGE** for graph-to-text generation.

- **WebNLG**: Our primary dataset for graph-to-text experiments and generation trajectory analysis.
- **LAGRANGE**: Used for cross-dataset evaluation to assess how well models generalize beyond WebNLG.
