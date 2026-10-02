"""Step 5: reinforcement learning with GRPO, starting from the v2 SFT adapter.

GRPO in one paragraph: for each question, sample a group of answers (num_generations) from the
current model. Grade each with train/rewards.py. An answer's *advantage* is its reward minus the
group's mean (divided by the group's std): better than its siblings → make those tokens more
likely, worse → less likely. No value network, no human labels, just the verifiers. A question
where all answers score the same teaches nothing, which is why the prompts are a mix the model
sometimes gets right and sometimes doesn't.

We keep training a LoRA adapter (v2's, or round 1's for round 2) on the same 4-bit base as Step 3.
--beta adds a KL penalty toward the adapter we *start* from: TRL copies it as a frozen "ref"
adapter, so the penalty holds back drift on skills no reward covers (explanations, pronunciation).
Round 1 ran with beta=0.

    uv run python -m train.grpo --smoke          # 2 steps: checks generation, rewards and the update run
    uv run python -m train.grpo                  # the real run, writes outputs/grpo/
    uv run python -m train.grpo --resume         # continue an interrupted run from its last checkpoint
    uv run python -m train.grpo --adapter outputs/grpo/final --prompts data/out/rl_prompts_hard.jsonl         --beta 0.03 --steps 150 --save-steps 25 --out outputs/grpo2      # round 2 (scripts/grpo_round2.py)
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig, GRPOTrainer

from train.rewards import make_rewards
from train.sft import BASE

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "data" / "out" / "rl_prompts.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 steps on a few prompts")
    ap.add_argument("--adapter", default=str(ROOT / "outputs" / "sft-v2" / "final"))
    ap.add_argument("--prompts", default=str(PROMPTS), type=Path)
    ap.add_argument("--beta", type=float, default=0.0, help="KL penalty toward the starting adapter")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--generations", type=int, default=8, help="answers sampled per question")
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--max-completion", type=int, default=512)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "grpo"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--save-limit", type=int, default=4, help="checkpoints kept; 0 keeps all")
    ap.add_argument("--resume", action="store_true", help="continue from the newest checkpoint in --out")
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.prompts.read_text(encoding="utf-8").splitlines() if line.strip()]
    ds = Dataset.from_list(rows[:16] if args.smoke else rows)
    print(f"{len(ds)} prompts from {args.prompts.name}")

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
    model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)

    config = GRPOConfig(
        output_dir=args.out,
        max_steps=2 if args.smoke else args.steps,
        # All groups of a step (prompts-per-step x generations answers) are generated together, then the
        # update runs 2 answers at a time: Gemma's 262k-token vocabulary makes the logits of 6 long answers
        # alone ~4 GB, more than a 12 GB card has left.
        per_device_train_batch_size=2,
        gradient_accumulation_steps=args.prompts_per_step * args.generations // 2,
        num_generations=args.generations,
        max_completion_length=args.max_completion,
        temperature=1.0,  # sampling must vary, or every answer in a group is the same and nothing is learned
        beta=args.beta,
        loss_type="dapo",  # token-level normalization over the batch: no bias toward short or long answers
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=0.05,
        reward_weights=[1.0, 1.0],
        gradient_checkpointing=True,
        bf16=True,
        # Not the paged variant: paging moves optimizer state between GPU and RAM through the driver, and a
        # checkpoint save with the card nearly full froze Windows (nvlddmkm error). LoRA's state is ~50 MB anyway.
        optim="adamw_bnb_8bit",
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_limit or None,
        report_to="none",
        seed=args.seed,
    )

    trainer = GRPOTrainer(model=model, reward_funcs=make_rewards(tokenizer, args.max_completion),
                          args=config, train_dataset=ds, processing_class=tokenizer)
    # TRL casts a 4-bit model's trainable weights to bf16 (the QLoRA recipe). At lr 1e-5 most Adam steps are
    # smaller than bf16's spacing near the adapter's weights and round to nothing, so train them in fp32 again.
    # The optimizer is built in train(), after this.
    for p in trainer.model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    trainer.model.print_trainable_parameters()
    print("trainable dtypes:", sorted({str(p.dtype) for p in trainer.model.parameters() if p.requires_grad}))
    trainer.train(resume_from_checkpoint=args.resume or None)  # True = newest checkpoint in output_dir
    trainer.save_model(Path(args.out) / "final")
    print(f"adapter saved to {Path(args.out) / 'final'}")
    print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
