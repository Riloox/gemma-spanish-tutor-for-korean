"""Step 4 in one command: adapter -> merged checkpoint -> bf16 GGUF -> Q4_K_M GGUF -> Ollama model.

Each stage runs as a subprocess so a failure stops the chain with the tool's own error message.

    uv run python -m train.ship                                    # outputs/sft/final -> gemma4-tutor
    uv run python -m train.ship --adapter outputs/sft-v2/final --name gemma4-tutor-v2

Needs llama.cpp next to this repo (../llama.cpp: the git checkout for the converter, and a
release zip's binaries in ../llama.cpp/bin for llama-quantize), or --llama-cpp <dir>.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from train.sft import ROOT


def run(*cmd):
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=ROOT / "outputs" / "sft" / "final", type=Path)
    ap.add_argument("--name", default="gemma4-tutor", help="Ollama model name")
    ap.add_argument("--llama-cpp", default=ROOT.parent / "llama.cpp", type=Path)
    ap.add_argument("--keep", action="store_true", help="keep the 10 GB merged checkpoint and 9 GB bf16 GGUF")
    args = ap.parse_args()

    out = ROOT / "outputs"
    merged, bf16, q4 = out / f"merged-{args.name}", out / f"{args.name}-bf16.gguf", out / f"{args.name}-q4_k_m.gguf"

    run(sys.executable, "-m", "train.merge", "--adapter", args.adapter, "--out", merged)
    run(sys.executable, args.llama_cpp / "convert_hf_to_gguf.py", merged, "--outtype", "bf16", "--outfile", bf16)
    run(args.llama_cpp / "bin" / "llama-quantize.exe", bf16, q4, "Q4_K_M")

    # Same Modelfile as the repo's, pointed at this build's GGUF.
    modelfile = out / f"Modelfile.{args.name}"
    lines = (ROOT / "Modelfile").read_text(encoding="utf-8").splitlines()
    modelfile.write_text("\n".join(f"FROM {q4}" if ln.startswith("FROM ") else ln for ln in lines) + "\n",
                         encoding="utf-8")
    run("ollama", "create", args.name, "-f", modelfile)

    if not args.keep:
        bf16.unlink()
        for f in merged.iterdir():
            f.unlink()
        merged.rmdir()
    print(f"\ndone: `ollama run {args.name}`, then `uv run python eval/run_eval.py --models {args.name}`")


if __name__ == "__main__":
    main()
