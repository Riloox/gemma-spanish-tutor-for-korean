"""GRPO round 2, start to finish. Started by scripts/windows/grpo-round2.bat; safe to stop and re-run (every stage resumes).

  1. filter   ask the round-1 model every RL prompt 4 times, keep the ones it doesn't always ace
              (data.filter_rl -> data/out/rl_prompts_hard.jsonl)
  2. train    continue from round 1's adapter on those prompts, KL penalty toward it (beta 0.03),
              a checkpoint every 25 steps (train.grpo -> outputs/grpo2/)
  3. select   ship checkpoints 50/100/150 to Ollama, score each on the dev set (eval/eval_set.jsonl),
              keep the best as gemma4-tutor-grpo2 (ties go to the earlier, less-drifted one)
  4. test     the frozen test set (eval/test_set.jsonl), once: grpo2 vs round 1 vs v2
  5. compare  paired significance tests (eval/compare.py) -> outputs/grpo2/summary.md

A stage is done when its marker file outputs/grpo2/stage-<name>.done exists; delete one to redo it.
Everything is logged to logs/grpo-round2.log.

    uv run python -u scripts/grpo_round2.py [--shutdown]
"""

import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from compare import compare  # noqa: E402

HOST = "http://127.0.0.1:11434"
LOG = ROOT / "logs" / "grpo-round2.log"
OUT = ROOT / "outputs" / "grpo2"
RESULTS = ROOT / "results"

START_ADAPTER = ROOT / "outputs" / "grpo" / "final"
ROUND1, V2, NEW = "gemma4-tutor-grpo", "gemma4-tutor-v2", "gemma4-tutor-grpo2"
ROUND1_DEV = RESULTS / "20260929-030807" / f"{ROUND1}.jsonl"  # last night's dev run; redone if missing
BETA, STEPS, SAVE_STEPS = 0.03, 150, 25
# Round 1's answers averaged ~95 tokens and real ones stayed under ~350; only runaways reached 512. Two 512-token
# answers need 1 GiB of fp32 logits in the backward pass, and one such batch ran the 12 GB card out of memory.
MAX_COMPLETION = 384
CANDIDATES = [50, 100, 150]
DEV, TEST = ROOT / "eval" / "eval_set.jsonl", ROOT / "eval" / "test_set.jsonl"


def log(msg):
    line = f"[{datetime.now():%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(*cmd):
    """Run a command, streaming its output into the log. Raises if it fails."""
    cmd = [str(c) for c in cmd]
    log("$ " + " ".join(cmd))
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        log("  " + line.rstrip()[-400:])  # progress bars redraw with \r; keep the latest part
    if proc.wait() != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}")


# ---------- Ollama ----------

def api(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{HOST}{path}", data, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def ollama():
    return shutil.which("ollama") or str(Path(os.environ["LOCALAPPDATA"]) / "Programs" / "Ollama" / "ollama.exe")


def ensure_ollama():
    try:
        api("/api/version")
        return
    except OSError:
        log("Ollama not running; starting it")
    subprocess.Popen([ollama(), "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    for _ in range(60):
        time.sleep(2)
        try:
            api("/api/version")
            return
        except OSError:
            pass
    raise RuntimeError("Ollama did not start")


def has_model(name):
    return any(m["name"] in (name, f"{name}:latest") for m in api("/api/tags")["models"])


def unload_all():
    """Training needs the whole 12 GB card: evict whatever Ollama has in VRAM."""
    try:
        for m in api("/api/ps")["models"]:
            log(f"unloading {m['name']} from the GPU")
            api("/api/generate", {"model": m["name"], "keep_alive": 0})
    except OSError:
        pass  # Ollama not running: nothing loaded


# ---------- stages ----------

def stage(name, fn):
    marker = OUT / f"stage-{name}.done"
    if marker.exists():
        log(f"stage {name}: already done, skipping")
        return
    log(f"stage {name}: start")
    fn()
    marker.write_text(datetime.now().isoformat(), encoding="utf-8")
    log(f"stage {name}: done")


def do_filter():
    ensure_ollama()
    for m in (ROUND1, V2):
        if not has_model(m):
            raise RuntimeError(f"Ollama model {m} is missing; ship it first (python -m train.ship ...)")
    run(sys.executable, "-m", "data.filter_rl", "--model", ROUND1)


def vram_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
        return int(out.split()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def do_train():
    unload_all()
    time.sleep(5)
    used = vram_used_mib()
    log(f"VRAM in use before training: {used} MiB (training needs ~10.3 GiB of the 12)")
    if used is not None and used > 1200:
        log("WARNING: other apps hold a lot of VRAM (Docker Desktop, Steam, browsers...). "
            "Close them if you can: with <1 GB free the driver crashed once.")
    resume = ["--resume"] if any(OUT.glob("checkpoint-*")) else []
    run(sys.executable, "-m", "train.grpo", "--adapter", START_ADAPTER,
        "--prompts", ROOT / "data" / "out" / "rl_prompts_hard.jsonl",
        "--beta", BETA, "--steps", STEPS, "--max-completion", MAX_COMPLETION, "--save-steps", SAVE_STEPS, "--save-limit", 0, "--out", OUT, *resume)


def dev_result(step):
    return RESULTS / f"grpo2-dev-s{step}" / f"{NEW}-s{step}.jsonl"


def score(path):
    return sum(json.loads(line)["check_pass"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def do_select():
    ensure_ollama()
    if not ROUND1_DEV.exists():
        run(sys.executable, "eval/run_eval.py", "--models", ROUND1, "--out", ROUND1_DEV.parent)
    for step in CANDIDATES:
        if dev_result(step).exists():
            log(f"checkpoint {step}: dev score {score(dev_result(step))}/100 (already evaluated)")
            continue
        name = f"{NEW}-s{step}"
        run(sys.executable, "-m", "train.ship", "--adapter", OUT / f"checkpoint-{step}", "--name", name)
        run(sys.executable, "eval/run_eval.py", "--models", name, "--out", dev_result(step).parent)
        log(f"checkpoint {step}: dev score {score(dev_result(step))}/100")

    scores = {step: score(dev_result(step)) for step in CANDIDATES}
    best = max(CANDIDATES, key=lambda s: (scores[s], -s))  # ties -> earlier step
    log(f"dev scores {scores}, round 1 {score(ROUND1_DEV)}/100 -> best is step {best}")
    (OUT / "best.json").write_text(json.dumps({"step": best, "dev_scores": scores}), encoding="utf-8")
    if has_model(f"{NEW}-s{best}"):  # gone only if a previous try already copied it and cleaned up
        run(ollama(), "cp", f"{NEW}-s{best}", NEW)
    for step in CANDIDATES:  # 3.4 GB each; the best one lives on as gemma4-tutor-grpo2
        name = f"{NEW}-s{step}"
        if has_model(name):
            run(ollama(), "rm", name)
        if step != best:
            for f in (ROOT / "outputs" / f"{name}-q4_k_m.gguf", ROOT / "outputs" / f"Modelfile.{name}"):
                f.unlink(missing_ok=True)


def do_test():
    ensure_ollama()
    run(sys.executable, "eval/run_eval.py", "--eval", TEST, "--models", NEW, ROUND1, V2, "--out", RESULTS / "grpo2-test")


def do_compare():
    best = json.loads((OUT / "best.json").read_text(encoding="utf-8"))
    test = RESULTS / "grpo2-test"
    parts = [
        "# GRPO round 2 summary", "",
        f"Round 2: round 1's adapter + {STEPS} steps on the hard prompts, beta {BETA}.",
        f"Dev scores by checkpoint: {best['dev_scores']} (round 1: {score(ROUND1_DEV)}/100). Chosen: step {best['step']}.", "",
        "The **test set** is the verdict: the dev set picked the checkpoint, so its score for the winner is",
        "flattered (the best of 3 tries). A difference counts when its McNemar p is below 0.05 and the interval excludes 0.", "",
        "# Test set (70 held-out items)", "",
        compare(test / f"{ROUND1}.jsonl", test / f"{NEW}.jsonl", TEST),
        compare(test / f"{V2}.jsonl", test / f"{NEW}.jsonl", TEST),
        compare(test / f"{V2}.jsonl", test / f"{ROUND1}.jsonl", TEST),
        "# Dev set (100 items, used for checkpoint choice)", "",
        compare(ROUND1_DEV, dev_result(best["step"]), DEV),
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
    log("=== GRPO round 2 started")
    try:
        stage("filter", do_filter)
        stage("train", do_train)
        stage("select", do_select)
        stage("test", do_test)
        stage("compare", do_compare)
        log(f"=== ALL DONE in {(time.time() - start) / 3600:.1f} h. Read outputs/grpo2/summary.md")
    except KeyboardInterrupt:
        log("stopped by you (Ctrl+C). Run scripts/windows/grpo-round2.bat again to resume.")
    except Exception as e:
        log(f"FAILED: {e}. Run scripts/windows/grpo-round2.bat again to resume from where it stopped.")
        raise SystemExit(1)
    finally:
        if args.shutdown:
            log("shutting down in 2 minutes (cancel with: shutdown /a)")
            subprocess.run(["shutdown", "/s", "/t", "120", "/c", "GRPO round 2 finished"])


if __name__ == "__main__":
    main()
