"""Step 3: supervised fine-tuning (SFT) of gemma-4-E2B-it with QLoRA.

QLoRA in one paragraph: the base model is loaded in 4-bit (NF4) and frozen. Small
trainable matrices (LoRA adapters, rank r) are added next to the attention and MLP
layers. Only the adapters learn, so the whole thing fits in 12 GB of VRAM and the
result is a ~100 MB adapter instead of a 10 GB model. Step 4 merges it back in.

Data: the three Step 2 datasets (conjugation, corrections, explanations). 5% of each
kind is held out as a validation set, so we can see overfitting happen: validation
loss going up while training loss keeps going down.

Loss is computed on the tutor's answers only, not on the system prompt or the
question. We want the model to learn to *answer*, not to write learner questions.

    uv run python -m train.sft --smoke        # 10 steps on 40 examples: checks everything runs
    uv run python -m train.sft                # the real run (~hours), writes outputs/sft/
"""

import argparse
import collections
import json
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForMultimodalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "out"
FILES = ["sft_conjugation.jsonl", "sft_corrections.jsonl", "sft_explanations.jsonl"]
BASE = "google/gemma-4-E2B-it"  # same model as Ollama's gemma4:e2b, the Step 1 winner
VAL_FRACTION = 0.05

# LoRA goes on the text model's linear layers only. The vision and audio encoders stay untouched.
TARGET = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"


def load_split(seed, limit=None):
    """Prompt/completion pairs, split per kind so every kind shows up in validation."""
    by_kind = collections.defaultdict(list)
    for name in FILES:
        for line in (DATA / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                sys_msg, user, assistant = r["messages"]
                by_kind[r["meta"]["kind"]].append({"prompt": [sys_msg, user], "completion": [assistant],
                                                   "kind": r["meta"]["kind"]})
    rng = random.Random(seed)
    train, val = [], []
    for rows in by_kind.values():
        rng.shuffle(rows)
        n_val = max(1, round(len(rows) * VAL_FRACTION))
        val += rows[:n_val]
        train += rows[n_val:]
    rng.shuffle(train)
    if limit:
        train, val = train[:limit], val[:max(4, limit // 10)]
    print(f"train {len(train)}, val {len(val)}  " + ", ".join(f"{k}={len(v)}" for k, v in sorted(by_kind.items())))
    return Dataset.from_list(train), Dataset.from_list(val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the pipeline")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "sft"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_ds, val_ds = load_split(args.seed, limit=40 if args.smoke else None)

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForMultimodalLM.from_pretrained(
        BASE,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_storage=torch.bfloat16,
        ),
    )

    peft_config = LoraConfig(r=args.rank, lora_alpha=args.rank, lora_dropout=0.05, bias="none",
                             task_type="CAUSAL_LM", target_modules=TARGET)

    config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=10 if args.smoke else -1,
        # Effective batch 16 = 2 examples per step x 8 accumulated steps. Bigger per-step batches don't fit.
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=2,
        gradient_checkpointing=True,  # recompute activations in the backward pass: slower, much less VRAM
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,  # a float < 1 means "this fraction of all steps"
        max_length=args.max_length,
        completion_only_loss=True,  # loss on the tutor's answer only
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=1 if args.smoke else 10,
        eval_strategy="steps",
        eval_steps=5 if args.smoke else 50,
        save_strategy="steps",
        save_steps=10 if args.smoke else 50,
        save_total_limit=3,
        load_best_model_at_end=not args.smoke,
        metric_for_best_model="eval_loss",
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(model=model, args=config, train_dataset=train_ds, eval_dataset=val_ds,
                         processing_class=tokenizer, peft_config=peft_config)
    trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.save_model(Path(args.out) / "final")
    print(f"adapter saved to {Path(args.out) / 'final'}")
    print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
