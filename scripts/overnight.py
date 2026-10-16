"""Overnight job for step 2c. Started by scripts/windows/overnight.bat; safe to re-run (every stage resumes).

  1. keep Windows awake while this runs
  2. make sure Ollama is running and the teacher model is downloaded
  3. teacher exam: run the 57-question eval on the teacher (once)
  4. generate questions -> answers -> filter (data.build_explanations)

Everything is logged to logs/overnight-<date>.log.
"""

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
sys.path.insert(0, str(ROOT))
from data.build_explanations import HOST, TEACHER  # noqa: E402

LOG = ROOT / "logs" / f"overnight-{datetime.now():%Y%m%d-%H%M}.log"
TEACHER_EVAL_MARKER = ROOT / "data" / "out" / "teacher_eval_done.txt"


def log(msg):
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def keep_awake():
    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no sleep while this process lives. The screen may still turn off.
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def ollama_exe():
    return shutil.which("ollama") or str(Path(os.environ["LOCALAPPDATA"]) / "Programs" / "Ollama" / "ollama.exe")


def get(path):
    with urllib.request.urlopen(f"{HOST}{path}", timeout=10) as r:
        return json.loads(r.read())


def ensure_ollama():
    try:
        get("/api/version")
        return
    except OSError:
        log("Ollama not running; starting it")
    subprocess.Popen([ollama_exe(), "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    for _ in range(60):
        time.sleep(2)
        try:
            get("/api/version")
            return
        except OSError:
            pass
    raise RuntimeError("Ollama did not start")


def ensure_model():
    names = {m["name"] for m in get("/api/tags")["models"]}
    if TEACHER in names or f"{TEACHER}:latest" in names:
        return
    log(f"downloading {TEACHER} (~17 GB)")
    run([ollama_exe(), "pull", TEACHER])


def run(cmd):
    """Run a command, streaming its output into the log. Raises if it fails."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        if "%" in line and "pulling" in line:
            continue  # skip download progress spam
        log("  " + line.rstrip())
    if proc.wait() != 0:
        raise RuntimeError(f"command failed: {' '.join(map(str, cmd))}")


def main():
    LOG.parent.mkdir(exist_ok=True)
    start = time.time()
    keep_awake()
    log(f"overnight job started; log: {LOG}")
    try:
        ensure_ollama()
        ensure_model()
        if TEACHER_EVAL_MARKER.exists():
            log("teacher exam already done, skipping")
        else:
            log(f"teacher exam: 57 questions on {TEACHER}")
            run([sys.executable, "eval/run_eval.py", "--models", TEACHER])
            TEACHER_EVAL_MARKER.parent.mkdir(parents=True, exist_ok=True)
            TEACHER_EVAL_MARKER.write_text(datetime.now().isoformat(), encoding="utf-8")
        log("generating explanations (questions -> answers -> review -> filter)")
        run([sys.executable, "-m", "data.build_explanations", "--stage", "all"])
        log(f"ALL DONE in {(time.time() - start) / 3600:.1f} h")
    except KeyboardInterrupt:
        log("stopped by you (Ctrl+C). Run scripts/windows/overnight.bat again to resume.")
    except Exception as e:
        log(f"FAILED: {e}. Run scripts/windows/overnight.bat again to resume from where it stopped.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
