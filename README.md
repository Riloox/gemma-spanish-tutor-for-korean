# gemma-spanish-tutor-for-korean

A 2B Spanish tutor for Korean speakers, fine-tuned from Gemma 4 E2B on one laptop GPU, running locally in Ollama.

**Result:** on a 208-item held-out test set, built and frozen before the final round of training, the tuned
model scores **177/208 vs 123/208** for the stock `gemma4:e2b` (McNemar p < 0.001, 95% interval +18% to +34%).
With the app system prompt it scores 182/208.

This is a learning project and a prototype, not a product. Every step has a doc that explains the concepts
before the code, including the rounds that didn't work.

## What it does

The learner writes a Spanish sentence and asks (in Korean) whether it's right. The tutor rewrites it, names
what changed, explains the rule in Korean, and gives one practice exercise:

> **Learner:** 버스 놓칠까 봐 걱정되는 마음을 담아봤어요. "Si el autobús llegaría a tiempo, no perderíamos el tour." 괜찮나요?
>
> **Tutor (v4):**
> **다시 써 보면:** Si el autobús llegara a tiempo, no perderíamos el tour.
> **바뀐 곳:** 'llegaría' → 'llegara'
>
> 아주 잘 쓰셨어요! … 동사의 형태를 조금 바꿔야 더 자연스러워요.
>
> **고친 문장:** Si el autobús llegara a tiempo, no perderíamos el tour.
> (만약 버스가 제시간에 도착한다면, 우리는 투어를 놓치지 않을 것이다.)
>
> **왜 그런가요?** … 조건절에는 보통 직설법 과거(pretérito imperfecto)를 사용해요. 'llegaría'는 가정법(subjuntivo)으로 …

The fix is right, but the explanation gets the terms wrong: `llegara` is the imperfect *subjunctive*, and
`llegaría` is the *conditional*. That's typical. The verifiers grade the fix, not the explanation (see Limitations).

## Results

The headline numbers are from `eval/test_set_v4.jsonl`. It was written by `gemma4:26b`, checked by blind judges,
kept away from every training sentence, and scored on the stock model before Round 4 training started.

| model | all (208) | real error caught (104) | correct sentence left alone (104) |
|---|---|---|---|
| stock `gemma4:e2b` | 123 | **90** | 33 |
| grpo3 (Round 3) | 167 | 71 | 96 |
| **v4 (shipped)** | **177** | 81 | **96** |
| grpo4 (v4 + GRPO) | 179 | 82 | 97 |

- The win comes from no longer inventing errors. The stock model "corrects" two out of three already-correct sentences.
- On sentences with a real error, v4 is **slightly worse** than the stock model (81 vs 90, p=0.064, not significant).
- GRPO on top of v4 didn't add anything measurable (179 vs 177, p=0.75), so the shipped model is SFT only.
- Conjugation stays at 24/24 on the 100-item dev set.

## How it was built

| Step | What | Doc |
|---|---|---|
| 1 | Baseline: measured 5 local models, picked `gemma4:e2b` (fast, best per size) | [03](docs/03-step1-baseline.md) |
| 2 | Training data: conjugations and corrections that are correct by construction, explanations distilled from `gemma4:26b` and hand-reviewed | [04](docs/04-step2-training-data.md) |
| 3 | SFT with QLoRA on 12 GB VRAM | [05](docs/05-step3-sft.md) |
| 4 | Merge → GGUF → Q4_K_M → Ollama; first evals (SFT taught the format but started praising wrong sentences) | [06](docs/06-step4-ollama.md) |
| 5 | GRPO with automatic verifiers as rewards; hard-prompt filtering; significance tests | [07](docs/07-step5-grpo.md) |
| 6 | Rounds 3–4: check-first vs rewrite-first, teacher-written items, a new frozen test set, self-distilled SFT | [08](docs/08-rounds-3-4.md) |

New to the terms? Start with [docs/01-how-llms-are-made.md](docs/01-how-llms-are-made.md). The target
behavior is in [docs/02-model-spec.md](docs/02-model-spec.md).

## Limitations

- **Explanations aren't verified.** Rewards and evals check the verdict and the corrected form. The rule text
  can be wrong even when the fix is right (example above). Hand review of teacher explanations found a wrong
  claim in ~10–20%.
- **Recall.** It misses some errors the stock model catches (90 → 81 of 104).
- **Synthetic tests.** Test items are written by a bigger model and checked by blind judges, not by native
  speakers. They cover 8 error types; pronunciation and free conversation are rubric items, read by hand.
- **Text only.** llama.cpp's converter drops Gemma 4's vision and audio encoders.
- **Windows-first.** Developed on Windows 11; the `.bat` launchers are Windows-only, but each one just runs a Python script.

## Running it

The weights aren't published, so you train them yourself. You need:
- an NVIDIA GPU with 12 GB of VRAM
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com) with `gemma4:e2b` and `gemma4:26b` pulled
- a [llama.cpp](https://github.com/ggml-org/llama.cpp) checkout next to this repo, for GGUF conversion

The repo is a research log, not a one-command build. Each round's pipeline reads the data, models and saved
answers of the rounds before it, so a from-scratch run goes in order. Each pipeline is resumable, and most
take an overnight run:

1. Step 2 data: `data/build_conjugation.py`, `data/build_corrections.py`, `scripts/overnight.py` ([04](docs/04-step2-training-data.md))
2. SFT and ship: `train/sft.py`, `train/ship.py` ([05](docs/05-step3-sft.md), [06](docs/06-step4-ollama.md))
3. GRPO round 1, then `scripts/grpo_round2.py` ([07](docs/07-step5-grpo.md))
4. `scripts/round3.py`, then `scripts/round4.py` ([08](docs/08-rounds-3-4.md))

Then build the app model and talk to it:

```sh
uv run python scripts/ship_app.py         # v4 + prompts/app_system.txt -> Ollama "spanish-tutor"
ollama run spanish-tutor
```

Evaluate any Ollama model (standard library only):

```sh
python eval/run_eval.py --models gemma4:e2b spanish-tutor --eval eval/test_set_v4.jsonl
```

## Layout

```
docs/                     concept explainers + one doc per step
prompts/tutor_system.txt  system prompt used in training and eval
prompts/app_system.txt    the same prompt + rules for the planned voice/3D-character app
eval/
  test_set_v4.jsonl       208 held-out items: the verdict from Round 4 on
  test_set.jsonl          70 held-out items, used through Round 3 (Round 4 was designed after reading its failures)
  eval_set.jsonl          100-item dev set; picks checkpoints, so its winners' scores are flattered
  run_eval.py, compare.py eval runner and paired significance tests (McNemar + bootstrap)
data/                     data builders: generators, teacher items, self-distillation, RL prompts; output -> data/out/
train/                    sft.py, grpo.py, rewards.py (the verifiers), merge.py, ship.py
scripts/                  one resumable pipeline per round; scripts/windows/ has the overnight launchers
Modelfile                 Ollama import of a quantized GGUF
```

Generated data, checkpoints, GGUFs, logs and eval results are git-ignored.

## Hardware

RTX 4000 Ada Laptop (12 GB VRAM), 64 GB RAM, Windows 11.

## License

Code: [MIT](LICENSE). The base model and anything derived from it, including fine-tuned weights, are
subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). Part of the training data was
generated by `gemma4:26b`.
