#!/usr/bin/env python3
"""
Stage 2: Fine-tune LLaDA (LoRA) + projector with graph-conditioned generation.

Loads a Stage 1 checkpoint (encoder + projector weights) and continues training
with LLaDA unfrozen via LoRA. The encoder can be optionally frozen or unfrozen.

Single-GPU (quick test):
    python graph_llada/train/train_stage2.py \
        --stage1-checkpoint experiments/graph_llada/stage1/stage1_*/epoch_3 \
        --max-train-samples 50 --epochs 1

Multi-GPU:
    accelerate launch --num_processes=N graph_llada/train/train_stage2.py \
        --stage1-checkpoint experiments/graph_llada/stage1/stage1_*/epoch_3 \
        --epochs 2

With alpha scaling at inference:
    accelerate launch --num_processes=N graph_llada/train/train_stage2.py \
        --stage1-checkpoint ... --alpha 0.5 --gen-length 128

Inference-only from a stage 2 checkpoint:
    accelerate launch --num_processes=N graph_llada/train/train_stage2.py \
        --stage2-checkpoint experiments/graph_llada/stage2/stage2_*/epoch_3 \
        --alpha 0.5 --gen-length 128
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
from accelerate import Accelerator, DistributedDataParallelKwargs
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from graph_llada.data.graph_dataset import WebNLGGraphDataset, collate_graph_batch
from graph_llada.model.config import GraphLLaDAConfig
from graph_llada.model.graph_llada import GraphLLaDA

DEFAULT_MODEL = "GSAI-ML/LLaDA-8B-Instruct"
STRUCTURAL_IDS = {13, 126081, 126348}  # ".", <|endoftext|>, <|eot_id|>


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2: Fine-tune LLaDA (LoRA) + projector")

    # Model
    p.add_argument("--model", default=DEFAULT_MODEL, help="Base LLaDA model")
    p.add_argument("--stage1-checkpoint", default=None, help="Path to stage 1 checkpoint directory")
    p.add_argument("--stage2-checkpoint", default=None,
                   help="Path to stage 2 checkpoint directory (graph_modules.pt + lora_adapter/). "
                        "Implies --skip-training. Loads LoRA adapter and graph weights for inference.")
    p.add_argument("--freeze-graph-tower", action="store_true", help="Freeze encoder in stage 2")
    p.add_argument("--freeze-llada", action="store_true",
                   help="Freeze LLaDA (no LoRA); train only graph encoder + projector")
    p.add_argument("--skip-training", action="store_true",
                   help="Skip LoRA injection and all stage 2 training; load stage 1 graph weights and run inference only")

    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-targets", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])

    # Training
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lang", default="en", choices=["en", "ru"])
    p.add_argument("--max-train-samples", type=int, default=-1)
    p.add_argument("--max-test-samples", type=int, default=-1)
    p.add_argument("--output-dir", default="experiments/graph_llada/stage2")

    # Inference
    p.add_argument("--skip-inference", action="store_true")
    p.add_argument("--gen-length", type=int, default=128)
    p.add_argument("--alpha", type=float, default=1.0, help="Alpha scaling for structural tokens")
    p.add_argument("--trajectory", action="store_true", help="Record unmasking trajectory")

    return p.parse_args()


@torch.no_grad()
def run_inference(model, test_ds, tokenizer, gen_length, alpha, trajectory, device, is_main=True):
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

    # Resolve stage2-checkpoint: implies skip-training, derives stage1-checkpoint
    if args.stage2_checkpoint:
        args.skip_training = True
        # Stage 2 config contains the original stage 1 graph architecture params
        stage2_config_path = os.path.join(args.stage2_checkpoint, "config.json")
        if not os.path.isfile(stage2_config_path):
            raise FileNotFoundError(f"Stage 2 config not found: {stage2_config_path}")

    if args.freeze_llada and args.skip_training:
        raise ValueError("--freeze-llada and --skip-training are mutually exclusive")

    if not args.stage1_checkpoint and not args.stage2_checkpoint:
        raise ValueError("Either --stage1-checkpoint or --stage2-checkpoint is required")

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision="bf16",
        kwargs_handlers=[ddp_kwargs],
    )
    device = accelerator.device

    # Load config: from stage 2 checkpoint if available, else stage 1
    if args.stage2_checkpoint:
        config_path = os.path.join(args.stage2_checkpoint, "config.json")
        accelerator.print(f"Loading stage 2 checkpoint from {args.stage2_checkpoint}")
    else:
        config_path = os.path.join(args.stage1_checkpoint, "config.json")
        accelerator.print(f"Loading stage 1 checkpoint from {args.stage1_checkpoint}")

    with open(config_path) as f:
        ckpt_cfg = json.load(f)

    # Load tokenizer
    # For stage 2 checkpoint: load from adapter dir (has graph tokens already)
    if args.stage2_checkpoint:
        lora_dir = os.path.join(args.stage2_checkpoint, "lora_adapter")
        tokenizer = AutoTokenizer.from_pretrained(lora_dir, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"

    # Load base LLaDA
    if args.stage2_checkpoint:
        # Load base model, resize to match adapter (includes graph tokens), then merge LoRA
        accelerator.print(f"Loading {args.model} + LoRA adapter from {lora_dir}...")
        llada_model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
        llada_model.resize_token_embeddings(len(tokenizer))
        llada_model = PeftModel.from_pretrained(llada_model, lora_dir)
        llada_model = llada_model.merge_and_unload()
        accelerator.print("Merged LoRA adapter into base model")
    elif args.freeze_llada:
        accelerator.print(f"Loading {args.model} (frozen LLaDA, no LoRA)...")
        llada_model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    elif not args.skip_training:
        accelerator.print(f"Loading {args.model} + LoRA...")
        llada_model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_targets,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        llada_model = get_peft_model(llada_model, lora_config)
        llada_model.print_trainable_parameters()
    else:
        accelerator.print(f"Loading {args.model} (inference-only, no LoRA)...")
        llada_model = AutoModel.from_pretrained(args.model, trust_remote_code=True, torch_dtype=torch.bfloat16)

    # Build GraphLLaDA with graph config from checkpoint
    config = GraphLLaDAConfig(
        llada_model_name_or_path=args.model,
        graph_tower=ckpt_cfg["graph_tower"],
        graph_hidden_size=ckpt_cfg["graph_hidden_size"],
        graph_num_layers=ckpt_cfg["graph_num_layers"],
        graph_num_heads=ckpt_cfg["graph_num_heads"],
        graph_dropout=ckpt_cfg["graph_dropout"],
        graph_edge_dim=ckpt_cfg.get("graph_edge_dim"),
        graph_projector_type=ckpt_cfg["graph_projector_type"],
        freeze_llada=args.skip_training or args.freeze_llada,
        freeze_graph_tower=args.freeze_graph_tower,
    )

    model = GraphLLaDA(config, llada_model=llada_model)
    model.initialize_tokenizer(tokenizer)

    # Load graph weights from stage 2 or stage 1 checkpoint
    graph_weights_path = os.path.join(
        args.stage2_checkpoint or args.stage1_checkpoint, "graph_modules.pt"
    )
    graph_weights = torch.load(graph_weights_path, map_location="cpu", weights_only=True)
    model.load_graph_state_dict(graph_weights, strict=False)
    accelerator.print(f"Loaded graph module weights from {graph_weights_path}")

    # Log trainable params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print(f"Total params: {total:,} | Trainable: {trainable:,} ({100*trainable/total:.2f}%)")

    # Broadcast timestamp from rank 0 so all processes use the same value
    # (avoids filename mismatch when processes cross a second boundary)
    import torch.distributed as dist
    if dist.is_initialized():
        ts_tensor = torch.zeros(15, dtype=torch.long, device=device)
        if accelerator.is_main_process:
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            ts_tensor[:] = torch.tensor([ord(c) for c in ts_str], dtype=torch.long)
        dist.broadcast(ts_tensor, src=0)
        timestamp = "".join(chr(c) for c in ts_tensor.tolist())
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.skip_training:
        loader = optimizer = scheduler = None
        model = accelerator.prepare(model)
        if args.stage2_checkpoint:
            accelerator.print("\nLoaded Stage 2 checkpoint (LoRA merged + graph modules); running inference only.")
        else:
            accelerator.print("\nSkipping Stage 2 training and LoRA injection; running inference with base LLaDA + Stage 1 graph modules.")
    else:
        # Load data (rank 0 first to cache HF dataset, then other ranks)
        accelerator.print("Loading WebNLG data...")
        if accelerator.is_main_process:
            train_ds = WebNLGGraphDataset(
                tokenizer, split="train", lang=args.lang,
                max_seq_len=args.max_seq_len, max_samples=args.max_train_samples,
            )
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
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
        if args.freeze_llada:
            accelerator.print(f"\nStage 2 (frozen LLaDA): {args.epochs} epochs, training graph encoder + projector only")
        else:
            accelerator.print(f"\nStage 2: {args.epochs} epochs, LoRA r={args.lora_r}")
        accelerator.print(f"Effective batch size: {args.batch_size * args.grad_accum * accelerator.num_processes}")

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

            # Save checkpoint: graph modules + LoRA adapter
            if accelerator.is_main_process:
                ckpt_dir = os.path.join(args.output_dir, f"stage2_{timestamp}", f"epoch_{epoch+1}")
                os.makedirs(ckpt_dir, exist_ok=True)

                unwrapped = accelerator.unwrap_model(model)

                # Save graph modules
                torch.save(unwrapped.graph_state_dict(), os.path.join(ckpt_dir, "graph_modules.pt"))

                # Save LoRA adapter (only when LLaDA is not frozen)
                if not args.freeze_llada:
                    lora_dir = os.path.join(ckpt_dir, "lora_adapter")
                    unwrapped.llada.save_pretrained(lora_dir)
                    tokenizer.save_pretrained(lora_dir)
                else:
                    tokenizer.save_pretrained(ckpt_dir)

                # Save config
                cfg_dict = {**vars(config), "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
                             "freeze_llada_training": args.freeze_llada}
                with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
                    json.dump(cfg_dict, f, indent=2)

                accelerator.print(f"Saved checkpoint to {ckpt_dir}")

    # Inference
    if not args.skip_inference:
        accelerator.print("\nRunning inference on test set...")
        unwrapped = accelerator.unwrap_model(model)

        # Merge LoRA for faster inference when stage 2 actually used adapters.
        # (stage2-checkpoint already merged during loading; freeze-llada has no LoRA)
        if not args.skip_training and not args.freeze_llada:
            unwrapped.llada = unwrapped.llada.merge_and_unload()
        unwrapped.eval()

        # Rank 0 loads first to cache HF dataset; other ranks read from cache
        if accelerator.is_main_process:
            test_ds = WebNLGGraphDataset(
                tokenizer, split="test", lang=args.lang,
                max_seq_len=2048, max_samples=args.max_test_samples,
            )
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            test_ds = WebNLGGraphDataset(
                tokenizer, split="test", lang=args.lang,
                max_seq_len=2048, max_samples=args.max_test_samples,
            )

        # Shard test set across processes
        my_indices = list(range(accelerator.process_index, len(test_ds), accelerator.num_processes))

        class _Subset:
            def __init__(self, exs): self.examples = exs
        my_ds = _Subset([test_ds.examples[i] for i in my_indices])

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
            f"stage2_{args.lang}_test_gl{args.gen_length}{alpha_tag}_{timestamp}{rank_tag}.jsonl",
        )
        os.makedirs(os.path.dirname(shard_path), exist_ok=True)
        pd.DataFrame(my_records).to_json(shard_path, orient="records", lines=True, force_ascii=False)
        print(f"[r{accelerator.process_index}] Wrote {len(my_records)} predictions to {shard_path}")

        # Merge shards
        accelerator.wait_for_everyone()
        if accelerator.is_main_process and accelerator.num_processes > 1:
            merged_path = os.path.join(
                args.output_dir,
                f"stage2_{args.lang}_test_gl{args.gen_length}{alpha_tag}_{timestamp}.jsonl",
            )
            all_records = []
            shard_paths = []
            for rank in range(accelerator.num_processes):
                rtag = f"_r{rank}"
                rpath = os.path.join(
                    args.output_dir,
                    f"stage2_{args.lang}_test_gl{args.gen_length}{alpha_tag}_{timestamp}{rtag}.jsonl",
                )
                all_records.extend(load_jsonl_records(rpath))
                shard_paths.append(rpath)
            pd.DataFrame(all_records).sort_values(by="id").to_json(
                merged_path, orient="records", lines=True, force_ascii=False,
            )
            # Clean up shards only after successful merge
            for rpath in shard_paths:
                os.remove(rpath)
            accelerator.print(f"Merged {len(all_records)} records -> {merged_path}")
            accelerator.print(f"\nEvaluate with:\n  python src/eval_webnlg_coco.py {merged_path}")


if __name__ == "__main__":
    main()
