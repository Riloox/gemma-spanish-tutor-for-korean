# my-own-lightweight-llm

A small (2–4B) Spanish tutor for Korean speakers that runs locally in Ollama and
aims to beat much larger general models at this one job.

Target behavior comes from real learner pain points: verb conjugation,
no feedback on your own writing, grammar Korean doesn't have, pronunciation,
English habits interfering. See [docs/02-model-spec.md](docs/02-model-spec.md).

## Roadmap

Each step has a doc that explains the concepts before the code.

- [x] **Step 1: Baseline.** Measured 5 models; picked `gemma4:e2b` (48/57, 106 tok/s). → [docs/03](docs/03-step1-baseline.md)
- [x] **Step 2: Training data.** 2,299 conjugations + 924 corrections (correct by construction) + 634 explanations distilled from local `gemma4:26b` (56/57 on the eval). Hand review: ~10–20% of explanations have a wrong claim. → [docs/04](docs/04-step2-training-data.md)
- [x] **Step 3: SFT with QLoRA.** First adapter trained (78 min, 8 GB VRAM). It learned the format and "답만"; it also invents errors in correct sentences, so it needs the real eval and probably more `corr_ok` data. → [docs/05](docs/05-step3-sft.md)
- [x] **Step 4: Ship to Ollama.** Merge → GGUF → Q4_K_M → `ollama create gemma4-tutor`. Eval: **50/57** (base 48): conjugation 24/24 (was 19), but corrections 15/18 (was 17) and two new wrong explanations. v2 (targeted correction data): 60/67 vs base 57/67, still 15/18 corrections, now failing by praising wrong sentences ("네, 맞아요! 👍"). On the 100-item eval v2 ties the base (83/100) but catches fewer errors (29/41 vs 35/41). → [docs/06](docs/06-step4-ollama.md)
- [ ] **Step 5: RL (GRPO).** Reward correct conjugations and corrections with automatic verifiers.
- [ ] **Step 6: Re-evaluate.** Beat the baseline, compare against a bigger model.

Start with [docs/01-how-llms-are-made.md](docs/01-how-llms-are-made.md) if the terms above are new.

## Layout

```
docs/                  concept explainers + step guides
prompts/               the tutor system prompt (used in eval and later in the Modelfile)
eval/eval_set.jsonl    100 test cases (41 corrections, 20 already-correct sentences, 24 conjugations, ...). Never train on these.
eval/run_eval.py       runs the eval against Ollama models, writes results/<timestamp>/report.md
data/                  Step 2 data builders; output goes to data/out/ (not committed)
data/manual_rejects.json  teacher answers rejected by hand review
scripts/               overnight job, hand-review sampler
train/sft.py           Step 3 QLoRA fine-tune, writes outputs/sft/
train/merge.py         Step 4 adapter + base -> outputs/merged (full HF checkpoint)
train/ship.py          Step 4 in one command: merge -> GGUF -> Q4_K_M -> ollama create
data/build_rl.py       Step 5 RL prompts with gold facts for the verifiers -> data/out/rl_prompts.jsonl
train/rewards.py       Step 5 verifiers (reuse the eval graders)
train/grpo.py          Step 5 GRPO on top of the v2 adapter, writes outputs/grpo/
Modelfile              Step 4 Ollama import of the quantized GGUF
```

## Hardware this was planned for

RTX 4000 Ada Laptop (12 GB VRAM), 64 GB RAM, Windows 11.
