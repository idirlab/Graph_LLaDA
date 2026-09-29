#!/usr/bin/env python3
"""
Stage 1: Train graph encoder + projector with frozen LLaDA.

Trains the GNN encoder, projector, and node/edge input projections on WebNLG
graph-to-text data. LLaDA is frozen — only graph modules receive gradients.

Single-GPU:
    python graph_llada/train/train_stage1.py --max-train-samples 50 --epochs 1

Multi-GPU:
    accelerate launch --num_processes=N graph_llada/train/train_stage1.py --epochs 3

With edge features (recommended for graph transformer):
    accelerate launch --num_processes=N graph_llada/train/train_stage1.py \
        --edge-dim 256 --epochs 3

Skip inference after training:
    accelerate launch --num_processes=N graph_llada/train/train_stage1.py \
        --epochs 3 --skip-inference
"""

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime

import pandas as pd
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from graph_llada.constants import MASK_TOKEN_ID
from graph_llada.data.graph_dataset import WebNLGGraphDataset, collate_graph_batch
from graph_llada.model.config import GraphLLaDAConfig
from graph_llada.model.graph_llada import GraphLLaDA

DEFAULT_MODEL = "GSAI-ML/LLaDA-8B-Instruct"
WRITE_EVERY = 10

# Structural token IDs for alpha scaling (Llama-3 tokenizer)
STRUCTURAL_IDS = {13, 126081, 126348}  # ".", <|endoftext|>, <|eot_id|>


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1: Train graph encoder + projector (LLaDA frozen)")

    # Model
    p.add_argument("--model", default=DEFAULT_MODEL, help="Base LLaDA model")
    p.add_argument("--graph-tower", default="graph_transformer", choices=["graph_transformer", "gcn"])
    p.add_argument("--graph-hidden-size", type=int, default=256)
    p.add_argument("--graph-num-layers", type=int, default=2)
    p.add_argument("--graph-num-heads", type=int, default=4)
    p.add_argument("--graph-dropout", type=float, default=0.1)
    p.add_argument("--edge-dim", type=int, default=128, help="Edge feature dim (None = no edge features)")
    p.add_argument("--projector-type", default="mlp", choices=["linear", "mlp"])

    # Training
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--warmup-steps", type=int, default=10)
    p.add_argument("--max-seq-len", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lang", default="en", choices=["en", "ru"])
    p.add_argument("--max-train-samples", type=int, default=-1)
    p.add_argument("--max-test-samples", type=int, default=-1)
    p.add_argument("--output-dir", default="experiments/graph_llada/stage1")

    # Inference
    p.add_argument("--skip-inference", action="store_true")
    p.add_argument("--gen-length", type=int, default=128)
    p.add_argument("--alpha", type=float, default=0.5, help="Alpha scaling for structural tokens")
    p.add_argument("--trajectory", action="store_true", help="Record unmasking trajectory")

    return p.parse_args()


@torch.no_grad()
def run_inference(model, test_ds, tokenizer, gen_length, alpha, trajectory, device, is_main=True):
    """Run inference on test examples, one at a time."""
    model.eval()
    structural_ids = STRUCTURAL_IDS if alpha < 1.0 else None
    records = []

    for ex in tqdm(test_ds.examples, desc="Inference", disable=not is_main):
        input_ids = ex["input_ids"].unsqueeze(0).to(device)
        graph_data = [ex["graph_data"]]

        result = model.generate(
            input_ids=input_ids,
            graph_data=graph_data,
            steps=gen_length,
            gen_length=gen_length,
            block_length=gen_length,
            temperature=0.0,
            alpha=alpha,
            structural_token_ids=structural_ids,
            return_trajectory=trajectory,
        )

        if trajectory:
            x, traj = result
            traj = traj[0]
        else:
            x, traj = result, None

        prompt_len = input_ids.shape[1]
        pred = tokenizer.decode(x[0, prompt_len:], skip_special_tokens=True).strip()
        rec = {
            "id": ex["gem_id"],
            "triples": ex["triples"],
            "reference": ex["target"],
            "prediction": pred,
            "gen_length": gen_length,
            "alpha": alpha,
        }
        if traj is not None:
            rec["trajectory"] = traj
        records.append(rec)

    return records


def load_jsonl_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision="bf16",
    )
    device = accelerator.device

    # Load tokenizer
    accelerator.print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"

    # Build config
    config = GraphLLaDAConfig(
        llada_model_name_or_path=args.model,
        graph_tower=args.graph_tower,
        graph_hidden_size=args.graph_hidden_size,
        graph_num_layers=args.graph_num_layers,
        graph_num_heads=args.graph_num_heads,
        graph_dropout=args.graph_dropout,
        graph_edge_dim=args.edge_dim,
        graph_projector_type=args.projector_type,
        freeze_llada=True,
        freeze_graph_tower=False,  # encoder trains in stage 1
    )

    # Build model
    model = GraphLLaDA(config)
    model.initialize_tokenizer(tokenizer)

    # Log trainable params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Total params: {total:,} | Trainable: {trainable:,} ({100*trainable/total:.2f}%)")

    # Load data
    accelerator.print("Loading WebNLG data...")
    train_ds = WebNLGGraphDataset(
        tokenizer, split="train", lang=args.lang,
        max_seq_len=args.max_seq_len, max_samples=args.max_train_samples,
    )
    accelerator.print(f"Train: {len(train_ds)} examples")

    loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_graph_batch(b, pad_id=tokenizer.pad_token_id or 0),
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    total_steps = args.epochs * math.ceil(len(loader) / args.grad_accum)
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)

    # Training
    accelerator.print(f"\nStage 1: {args.epochs} epochs, graph_tower={args.graph_tower}")
    accelerator.print(f"Effective batch size: {args.batch_size * args.grad_accum * accelerator.num_processes}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_steps = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}", disable=not accelerator.is_main_process)
        for batch in pbar:
            with accelerator.accumulate(model):
                loss = model(
                    input_ids=batch["input_ids"],
                    graph_data=batch["graph_data"],
                    labels=batch["labels"],
                    attention_mask=batch["attention_mask"],
                )
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.detach().item()
            n_steps += 1
            if n_steps % 50 == 0:
                pbar.set_postfix(loss=f"{total_loss / n_steps:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_loss = total_loss / max(n_steps, 1)
        accelerator.print(f"Epoch {epoch+1}/{args.epochs} - avg_loss: {avg_loss:.4f}")

        # Save graph module checkpoint
        if accelerator.is_main_process:
            ckpt_dir = os.path.join(args.output_dir, f"stage1_{timestamp}", f"epoch_{epoch+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            unwrapped = accelerator.unwrap_model(model)
            torch.save(unwrapped.graph_state_dict(), os.path.join(ckpt_dir, "graph_modules.pt"))
            with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
                json.dump(vars(config), f, indent=2)
            accelerator.print(f"Saved checkpoint to {ckpt_dir}")

    # Inference
    if not args.skip_inference:
        accelerator.print("\nRunning inference on test set...")
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.eval()

        test_ds = WebNLGGraphDataset(
            tokenizer, split="test", lang=args.lang,
            max_seq_len=2048, max_samples=args.max_test_samples,
        )

        # Shard test set across processes
        my_indices = list(range(accelerator.process_index, len(test_ds), accelerator.num_processes))
        my_ds_examples = [test_ds.examples[i] for i in my_indices]

        # Build a thin wrapper so run_inference can iterate
        class _Subset:
            def __init__(self, exs): self.examples = exs
        my_ds = _Subset(my_ds_examples)

        my_records = run_inference(
            unwrapped, my_ds, tokenizer,
            gen_length=args.gen_length, alpha=args.alpha,
            trajectory=args.trajectory, device=device,
            is_main=accelerator.is_main_process,
        )

        # Write shard
        rank_tag = f"_r{accelerator.process_index}" if accelerator.num_processes > 1 else ""
        alpha_tag = f"_a{args.alpha:.2f}".replace(".", "") if args.alpha < 1.0 else ""
        shard_path = os.path.join(
            args.output_dir,
            f"stage1_{args.lang}_test_gl{args.gen_length}{alpha_tag}_{timestamp}{rank_tag}.jsonl",
        )
        os.makedirs(os.path.dirname(shard_path), exist_ok=True)
        pd.DataFrame(my_records).to_json(shard_path, orient="records", lines=True, force_ascii=False)
        print(f"[r{accelerator.process_index}] Wrote {len(my_records)} predictions to {shard_path}")

        # Merge shards
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and accelerator.num_processes > 1:
            merged_path = os.path.join(
                args.output_dir,
                f"stage1_{args.lang}_test_gl{args.gen_length}{alpha_tag}_{timestamp}.jsonl",
            )
            all_records = []
            for rank in range(accelerator.num_processes):
                rtag = f"_r{rank}"
                rpath = os.path.join(
                    args.output_dir,
                    f"stage1_{args.lang}_test_gl{args.gen_length}{alpha_tag}_{timestamp}{rtag}.jsonl",
                )
                all_records.extend(load_jsonl_records(rpath))
                os.remove(rpath)
            pd.DataFrame(all_records).sort_values(by="id").to_json(
                merged_path, orient="records", lines=True, force_ascii=False,
            )
            accelerator.print(f"Merged {len(all_records)} records -> {merged_path}")
            accelerator.print(f"\nEvaluate with:\n  python src/eval_webnlg_coco.py {merged_path}")


if __name__ == "__main__":
    main()
