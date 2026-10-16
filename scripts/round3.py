"""Round 3, start to finish: teacher items -> SFT v3 (check-first) -> GRPO round 3 -> held-out test vs the base.
Started by scripts/windows/round3.bat; safe to stop and re-run (every stage resumes).

Why: grpo2 (59/70) vs the base (56/70) is noise. Every correction it misses opens with "네, 맞아요! 👍": the verdict
comes in the first tokens, before the model has looked at the word. And in round 1, 41-44% of personal_a and
si_clause prompts scored 0 on every sample, so RL had nothing to push toward. Two fixes:
  - check before verdict: the staged build_corrections redesign ("확인할 점 / 쓴 형태 ✓✗" first), never trained until now
  - new sentence frames from the teacher (data/build_teacher_items.py), checked by blind judging, for the weak types

  1. teacher  gemma4:26b writes and judges correction items -> data/out/teacher_items.jsonl, sft_teacher_corrections.jsonl
  2. data     RL prompts incl. teacher items -> data/out/rl_prompts_v3.jsonl
  3. sft      SFT v3 from the stock model on all four SFT files -> outputs/sft-v3/
  4. ship_v3  -> Ollama gemma4-tutor-v3, scored on the dev set
  5. filter   v3 answers every RL prompt 4 times; keep the ones it doesn't always ace -> rl_prompts_v3_hard.jsonl
  6. train    GRPO from v3: praise penalty, check-first reward, KL toward v3 -> outputs/grpo3/
  7. select   ship checkpoints, score on the dev set, keep the best as gemma4-tutor-grpo3
  8. test     the frozen test set, once: grpo3 and v3 (base and grpo2 answers are reused from their runs)
  9. compare  paired significance tests -> outputs/grpo3/summary.md

A stage is done when outputs/grpo3/stage-<name>.done exists; delete one to redo it.
Everything is logged to logs/round3.log.

    uv run python -u scripts/round3.py [--shutdown]
"""

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import grpo_round2 as r2  # Ollama and process helpers; they log through r2.LOG, pointed at this run's log below
from grpo_round2 import api, ensure_ollama, has_model, log, ollama, run, score, unload_all, vram_used_mib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from compare import compare  # noqa: E402

LOG = ROOT / "logs" / "round3.log"
OUT = ROOT / "outputs" / "grpo3"
SFT_OUT = ROOT / "outputs" / "sft-v3"
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "out"
PROMPTS, SCORES, HARD = DATA / "rl_prompts_v3.jsonl", DATA / "rl_difficulty_v3.jsonl", DATA / "rl_prompts_v3_hard.jsonl"

TEACHER = "gemma4:26b"
V3, NEW = "gemma4-tutor-v3", "gemma4-tutor-grpo3"
BASE_TEST = RESULTS / "base-test" / "gemma4_e2b.jsonl"
GRPO2_TEST = RESULTS / "grpo2-test" / "gemma4-tutor-grpo2.jsonl"
V3_DEV = RESULTS / "v3-dev" / f"{V3}.jsonl"
TEST_DIR = RESULTS / "grpo3-test"
DEV, TEST = ROOT / "eval" / "eval_set.jsonl", ROOT / "eval" / "test_set.jsonl"

BETA = 0.03            # KL toward v3: holds explanations and pronunciation, which no reward covers
PRAISE_PENALTY = 0.5   # "맞아요" on a sentence with an error is the failure that kept grpo2 level with the base
CHECK_WEIGHT = 0.25    # the "쓴 형태 ✓/✗" line quotes the learner's form with the right mark
STEPS, SAVE_STEPS, MAX_COMPLETION = 200, 25, 384
CANDIDATES = [50, 100, 150, 200]

r2.LOG = LOG


def stage(name, fn):
    marker = OUT / f"stage-{name}.done"
    if marker.exists():
        log(f"stage {name}: already done, skipping")
        return
    log(f"stage {name}: start")
    fn()
    marker.write_text(datetime.now().isoformat(), encoding="utf-8")
    log(f"stage {name}: done")


def free_gpu_for_training():
    unload_all()
    time.sleep(5)
    used = vram_used_mib()
    log(f"VRAM in use before training: {used} MiB (training needs ~10.3 GiB of the 12)")
    if used is not None and used > 1200:
        log("WARNING: other apps hold a lot of VRAM (Docker Desktop, Steam, browsers...). Close them if you can.")


# ---------- stages ----------

def do_teacher():
    ensure_ollama()
    if not has_model(TEACHER):
        raise RuntimeError(f"Ollama model {TEACHER} is missing: ollama pull {TEACHER}")
    run(sys.executable, "-m", "data.build_teacher_items", "--stage", "all")


def do_data():
    run(sys.executable, "-m", "data.build_corrections")  # deterministic: the check-first redesign, regenerated
    run(sys.executable, "-m", "data.build_rl", "--out", PROMPTS)


def do_sft():
    free_gpu_for_training()
    run(sys.executable, "-m", "train.sft", "--out", SFT_OUT)


def do_ship_v3():
    ensure_ollama()
    run(sys.executable, "-m", "train.ship", "--adapter", SFT_OUT / "final", "--name", V3)
    run(sys.executable, "eval/run_eval.py", "--models", V3, "--out", V3_DEV.parent)
    log(f"v3 dev score {score(V3_DEV)}/100")


def do_filter():
    ensure_ollama()
    run(sys.executable, "-m", "data.filter_rl", "--model", V3, "--max-tokens", MAX_COMPLETION,
        "--prompts", PROMPTS, "--scores", SCORES, "--hard", HARD)


def do_train():
    free_gpu_for_training()
    resume = ["--resume"] if any(OUT.glob("checkpoint-*")) else []
    run(sys.executable, "-m", "train.grpo", "--adapter", SFT_OUT / "final", "--prompts", HARD,
        "--beta", BETA, "--praise-penalty", PRAISE_PENALTY, "--check-weight", CHECK_WEIGHT,
        "--steps", STEPS, "--max-completion", MAX_COMPLETION, "--save-steps", SAVE_STEPS, "--save-limit", 0,
        "--out", OUT, *resume)


def dev_result(step):
    return RESULTS / f"grpo3-dev-s{step}" / f"{NEW}-s{step}.jsonl"


def do_select():
    ensure_ollama()
    for step in CANDIDATES:
        if dev_result(step).exists():
            log(f"checkpoint {step}: dev score {score(dev_result(step))}/100 (already evaluated)")
            continue
        name = f"{NEW}-s{step}"
        run(sys.executable, "-m", "train.ship", "--adapter", OUT / f"checkpoint-{step}", "--name", name)
        run(sys.executable, "eval/run_eval.py", "--models", name, "--out", dev_result(step).parent)
        log(f"checkpoint {step}: dev score {score(dev_result(step))}/100")

    scores = {step: score(dev_result(step)) for step in CANDIDATES}
    best = max(CANDIDATES, key=lambda s: (scores[s], -s))  # ties -> earlier, less drifted
    log(f"dev scores {scores}, v3 {score(V3_DEV)}/100 -> best is step {best}")
    (OUT / "best.json").write_text(json.dumps({"step": best, "dev_scores": scores}), encoding="utf-8")
    if has_model(f"{NEW}-s{best}"):
        run(ollama(), "cp", f"{NEW}-s{best}", NEW)
    for step in CANDIDATES:  # 3.4 GB each; the best one lives on as gemma4-tutor-grpo3
        name = f"{NEW}-s{step}"
        if has_model(name):
            run(ollama(), "rm", name)
        if step != best:
            for f in (ROOT / "outputs" / f"{name}-q4_k_m.gguf", ROOT / "outputs" / f"Modelfile.{name}"):
                f.unlink(missing_ok=True)


def do_test():
    ensure_ollama()
    run(sys.executable, "eval/run_eval.py", "--eval", TEST, "--models", NEW, V3, "--out", TEST_DIR)


def do_compare():
    best = json.loads((OUT / "best.json").read_text(encoding="utf-8"))
    new, v3 = TEST_DIR / f"{NEW}.jsonl", TEST_DIR / f"{V3}.jsonl"
    parts = [
        "# Round 3 summary", "",
        "SFT v3: check-first corrections + checked teacher items, from the stock model. "
        f"GRPO: {STEPS} steps from v3, beta {BETA}, praise penalty {PRAISE_PENALTY}, check weight {CHECK_WEIGHT}.",
        f"Dev scores by checkpoint: {best['dev_scores']} (v3: {score(V3_DEV)}/100). Chosen: step {best['step']}.", "",
        "The **test set** is the verdict: the dev set picked the checkpoint, so its dev score is flattered.",
        "A difference counts when its McNemar p is below 0.05 and the interval excludes 0.", "",
        "# Test set (70 held-out items)", "",
        compare(BASE_TEST, new, TEST),
        compare(GRPO2_TEST, new, TEST),
        compare(BASE_TEST, v3, TEST),
        compare(v3, new, TEST),
        "# Dev set (100 items, used for checkpoint choice)", "",
        compare(V3_DEV, dev_result(best["step"]), DEV),
    ]
    summary = OUT / "summary.md"
    summary.write_text("\n".join(parts), encoding="utf-8")
    log(f"summary written to {summary}")
    for line in summary.read_text(encoding="utf-8").splitlines():
        log("  " + line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shutdown", action="store_true", help="turn the PC off when finished (or failed)")
    args = ap.parse_args()
    LOG.parent.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no sleep while this runs; the screen may still turn off
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    start = time.time()
    log("=== round 3 started")
    try:
        for name, fn in [("teacher", do_teacher), ("data", do_data), ("sft", do_sft), ("ship_v3", do_ship_v3),
                         ("filter", do_filter), ("train", do_train), ("select", do_select), ("test", do_test),
                         ("compare", do_compare)]:
            stage(name, fn)
        log(f"=== ALL DONE in {(time.time() - start) / 3600:.1f} h. Read outputs/grpo3/summary.md")
    except KeyboardInterrupt:
        log("stopped by you (Ctrl+C). Run scripts/windows/round3.bat again to resume.")
    except Exception as e:
        log(f"FAILED: {e}. Run scripts/windows/round3.bat again to resume from where it stopped.")
        raise SystemExit(1)
    finally:
        if args.shutdown:
            log("shutting down in 2 minutes (cancel with: shutdown /a)")
            subprocess.run(["shutdown", "/s", "/t", "120", "/c", "Round 3 finished"])


if __name__ == "__main__":
    main()
