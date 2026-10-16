"""Round 4, start to finish: new test set -> baselines -> self-distilled SFT v4 (rewrite-first) -> GRPO round 4 -> test.
Started by round4.bat; safe to stop and re-run (every stage resumes).

Why: grpo3 (59/70) vs the base (56/70) is noise again, and it trades recall for precision: it stopped inventing
errors in correct sentences (20/30 -> 28/30, p=0.039) but praises errors the base catches (36/40 -> 31/40). All 9
of its misses look the same: check-first makes it name *one* spot to check before reading ("확인할 점: foto는 여성
→ preciosa"), and when the error is elsewhere ("El foto") it marks its spot ✓ and says "네, 맞아요". The base
catches errors by rewriting the whole sentence. SFT v3 is where recall was lost (36 -> 27/40, p=0.035).

  0. testset   gemma4:26b writes and blind-judges a new held-out test set (data/build_test_v4.py), frozen first:
               208 items, 104 with an error and 104 correct, away from every training sentence -> eval/test_set_v4.jsonl
  1. baseline  the base and grpo3 answer it before anything is trained
  2. distill   the base writes rewrite-first answers ("다시 써 보면: … / 바뀐 곳: …"), kept only if right
               (data/build_selfdistill.py); template answers where it never gets one right
  3. sft       SFT v4 from the stock model: self-distilled corrections + conjugation + explanations, lr 1e-4
  4. ship_v4   -> Ollama gemma4-tutor-v4, scored on the dev set
  5. filter    v4 answers every RL prompt 4 times under the new reward; keep the ones it doesn't always ace
  6. train     GRPO from v4 with the rewrite reward (train/rewards.grade_rewrite), KL toward v4 -> outputs/grpo4/
  7. select    ship checkpoints, score on the dev set, keep the best as gemma4-tutor-grpo4
  8. test      grpo4 and v4 on the new test set, once (+ grpo4 on the old 70-item set, for continuity only)
  9. compare   paired significance tests -> outputs/grpo4/summary.md

A stage is done when outputs/grpo4/stage-<name>.done exists; delete one to redo it.
Everything is logged to logs/round4.log.

    uv run python -u scripts/round4.py [--shutdown]
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
from grpo_round2 import ensure_ollama, has_model, log, ollama, run, score, unload_all, vram_used_mib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from compare import compare  # noqa: E402

LOG = ROOT / "logs" / "round4.log"
OUT = ROOT / "outputs" / "grpo4"
SFT_OUT = ROOT / "outputs" / "sft-v4"
RESULTS = ROOT / "results"
DATA = ROOT / "data" / "out"
PROMPTS = DATA / "rl_prompts_v3.jsonl"  # same RL prompts as round 3; filter_rl drops the new test set's sentences
SCORES, HARD = DATA / "rl_difficulty_v4.jsonl", DATA / "rl_prompts_v4_hard.jsonl"
SFT_FILES = ["sft_conjugation.jsonl", "sft_explanations.jsonl", "sft_corrections_v4.jsonl"]

TEACHER, BASE = "gemma4:26b", "gemma4:e2b"
GRPO3, V4, NEW = "gemma4-tutor-grpo3", "gemma4-tutor-v4", "gemma4-tutor-grpo4"
DEV, OLD_TEST, TEST = ROOT / "eval" / "eval_set.jsonl", ROOT / "eval" / "test_set.jsonl", ROOT / "eval" / "test_set_v4.jsonl"
TEST_DIR = RESULTS / "round4-test"
OLD_TEST_DIR = RESULTS / "grpo4-oldtest"
BASE_OLD_TEST = RESULTS / "base-test" / "gemma4_e2b.jsonl"
V4_DEV = RESULTS / "v4-dev" / f"{V4}.jsonl"

SFT_LR = 1e-4   # half of v3's: the self-distilled answers are close to what the model already writes
BETA = 0.03     # KL toward v4: holds explanations and pronunciation, which no reward covers
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


def need(*models):
    for m in models:
        if not has_model(m):
            raise RuntimeError(f"Ollama model {m} is missing: ollama pull {m} (or ship it)")


def result(model, directory):
    return directory / f"{model.replace(':', '_')}.jsonl"


# ---------- stages ----------

def do_testset():
    ensure_ollama()
    need(TEACHER)
    if TEST.exists():
        log(f"{TEST.name} already exists: frozen, not rebuilt")
        return
    run(sys.executable, "-m", "data.build_test_v4", "--stage", "all")


def do_baseline():
    ensure_ollama()
    need(BASE, GRPO3)
    run(sys.executable, "eval/run_eval.py", "--eval", TEST, "--models", BASE, GRPO3, "--out", TEST_DIR)
    for m in (BASE, GRPO3):
        log(f"{m} on the new test set: {score(result(m, TEST_DIR))}")


def do_distill():
    ensure_ollama()
    need(BASE)
    run(sys.executable, "-m", "data.build_selfdistill", "--model", BASE)


def do_sft():
    free_gpu_for_training()
    run(sys.executable, "-m", "train.sft", "--files", *SFT_FILES, "--lr", SFT_LR, "--out", SFT_OUT)


def do_ship_v4():
    ensure_ollama()
    run(sys.executable, "-m", "train.ship", "--adapter", SFT_OUT / "final", "--name", V4)
    run(sys.executable, "eval/run_eval.py", "--models", V4, "--out", V4_DEV.parent)
    log(f"v4 dev score {score(V4_DEV)}/100")


def do_filter():
    ensure_ollama()
    run(sys.executable, "-m", "data.filter_rl", "--model", V4, "--rewrite", "--max-tokens", MAX_COMPLETION,
        "--prompts", PROMPTS, "--scores", SCORES, "--hard", HARD)


def do_train():
    free_gpu_for_training()
    resume = ["--resume"] if any(OUT.glob("checkpoint-*")) else []
    run(sys.executable, "-m", "train.grpo", "--adapter", SFT_OUT / "final", "--prompts", HARD, "--rewrite",
        "--beta", BETA, "--steps", STEPS, "--max-completion", MAX_COMPLETION, "--save-steps", SAVE_STEPS,
        "--save-limit", 0, "--out", OUT, *resume)


def dev_result(step):
    return RESULTS / f"grpo4-dev-s{step}" / f"{NEW}-s{step}.jsonl"


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
    log(f"dev scores {scores}, v4 {score(V4_DEV)}/100 -> best is step {best}")
    (OUT / "best.json").write_text(json.dumps({"step": best, "dev_scores": scores}), encoding="utf-8")
    if has_model(f"{NEW}-s{best}"):
        run(ollama(), "cp", f"{NEW}-s{best}", NEW)
    for step in CANDIDATES:  # 3.4 GB each; the best one lives on as gemma4-tutor-grpo4
        name = f"{NEW}-s{step}"
        if has_model(name):
            run(ollama(), "rm", name)
        if step != best:
            for f in (ROOT / "outputs" / f"{name}-q4_k_m.gguf", ROOT / "outputs" / f"Modelfile.{name}"):
                f.unlink(missing_ok=True)


def do_test():
    ensure_ollama()
    run(sys.executable, "eval/run_eval.py", "--eval", TEST, "--models", NEW, V4, "--out", TEST_DIR)
    run(sys.executable, "eval/run_eval.py", "--eval", OLD_TEST, "--models", NEW, "--out", OLD_TEST_DIR)


def do_compare():
    best = json.loads((OUT / "best.json").read_text(encoding="utf-8"))
    base, grpo3, v4, new = (result(m, TEST_DIR) for m in (BASE, GRPO3, V4, NEW))
    n = sum(1 for line in TEST.read_text(encoding="utf-8").splitlines() if line.strip())
    parts = [
        "# Round 4 summary", "",
        f"SFT v4: rewrite-first corrections written by the base itself (kept if right), lr {SFT_LR}, from the stock "
        f"model. GRPO: {STEPS} steps from v4, rewrite reward, beta {BETA}.",
        f"Dev scores by checkpoint: {best['dev_scores']} (v4: {score(V4_DEV)}/100). Chosen: step {best['step']}.", "",
        f"The **new test set** ({n} items, built and scored for the base before training) is the verdict. "
        "A difference counts when its McNemar p is below 0.05 and the interval excludes 0.", "",
        f"# New test set ({n} held-out items)", "",
        compare(base, new, TEST),
        compare(grpo3, new, TEST),
        compare(base, v4, TEST),
        compare(v4, new, TEST),
        "# Old test set (70 items): continuity only",
        "",
        "Round 4 was designed after reading grpo3's failures on this set, so it can't grade round 4.", "",
        compare(BASE_OLD_TEST, result(NEW, OLD_TEST_DIR), OLD_TEST),
        "# Dev set (100 items, used for checkpoint choice)", "",
        compare(V4_DEV, dev_result(best["step"]), DEV),
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
    log("=== round 4 started")
    try:
        for name, fn in [("testset", do_testset), ("baseline", do_baseline), ("distill", do_distill),
                         ("sft", do_sft), ("ship_v4", do_ship_v4), ("filter", do_filter), ("train", do_train),
                         ("select", do_select), ("test", do_test), ("compare", do_compare)]:
            stage(name, fn)
        log(f"=== ALL DONE in {(time.time() - start) / 3600:.1f} h. Read outputs/grpo4/summary.md")
    except KeyboardInterrupt:
        log("stopped by you (Ctrl+C). Run round4.bat again to resume.")
    except Exception as e:
        log(f"FAILED: {e}. Run round4.bat again to resume from where it stopped.")
        raise SystemExit(1)
    finally:
        if args.shutdown:
            log("shutting down in 2 minutes (cancel with: shutdown /a)")
            subprocess.run(["shutdown", "/s", "/t", "120", "/c", "Round 4 finished"])


if __name__ == "__main__":
    main()
