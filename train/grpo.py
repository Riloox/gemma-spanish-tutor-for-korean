"""Step 5: reinforcement learning with GRPO, starting from the v2 SFT adapter.

GRPO in one paragraph: for each question, sample a group of answers (num_generations) from the
current model. Grade each with train/rewards.py. An answer's *advantage* is its reward minus the
group's mean (divided by the group's std): better than its siblings → make those tokens more
likely, worse → less likely. No value network, no human labels, just the verifiers. A question
where all answers score the same teaches nothing, which is why the prompts are a mix the model
sometimes gets right and sometimes doesn't.

We keep training the v2 LoRA adapter on the same 4-bit base as Step 3. beta=0 (no KL penalty):
with a KL term, TRL uses the adapter-disabled model as the reference, and that's the stock base,
not v2. Drift is watched with the eval instead.

    uv run python -m train.grpo --smoke          # 2 steps: checks generation, rewards and the update run
    uv run python -m train.grpo                  # the real run, writes outputs/grpo/
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
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--generations", type=int, default=8, help="answers sampled per question")
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--max-completion", type=int, default=512)
    ap.add_argument("--out", default=str(ROOT / "outputs" / "grpo"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = [json.loads(line) for line in PROMPTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    ds = Dataset.from_list(rows[:16] if args.smoke else rows)
    print(f"{len(ds)} prompts from {PROMPTS.name}")

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
        beta=0.0,
        loss_type="dapo",  # token-level normalization over the batch: no bias toward short or long answers
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=0.05,
        reward_weights=[1.0, 1.0],
        gradient_checkpointing=True,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_steps=1,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=4,
        report_to="none",
        seed=args.seed,
    )

    trainer = GRPOTrainer(model=model, reward_funcs=make_rewards(tokenizer, args.max_completion),
                          args=config, train_dataset=ds, processing_class=tokenizer)
    trainer.model.print_trainable_parameters()
    trainer.train()
    trainer.save_model(Path(args.out) / "final")
    print(f"adapter saved to {Path(args.out) / 'final'}")
    print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
