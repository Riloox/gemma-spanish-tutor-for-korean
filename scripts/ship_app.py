"""Ship the app model: a trained tutor with prompts/app_system.txt built in as its system prompt.

Training and eval always send prompts/tutor_system.txt, but the shipped gemma4-tutor-* models have no SYSTEM line,
so a bare `ollama run` talks to them without the prompt they were trained under. app_system.txt is that prompt
word for word plus the app's rules (voice in and out, the 3D character, the [칠판] whiteboard blocks).

    uv run python scripts/ship_app.py [--base gemma4-tutor-v4] [--name spanish-tutor]
"""

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="gemma4-tutor-v4", help="Ollama model to build on")
    ap.add_argument("--name", default="spanish-tutor", help="name of the new Ollama model")
    ap.add_argument("--system", default=ROOT / "prompts" / "app_system.txt", type=Path)
    args = ap.parse_args()

    system = args.system.read_text(encoding="utf-8").strip()
    modelfile = ROOT / "outputs" / f"Modelfile.{args.name}"
    # Same prompt handling and sampling as the base (see Modelfile); only the SYSTEM line is new
    modelfile.write_text(
        f"FROM {args.base}\n\n"
        "RENDERER gemma4\nPARSER gemma4\n"
        "PARAMETER temperature 1\nPARAMETER top_k 64\nPARAMETER top_p 0.95\n\n"
        f'SYSTEM """{system}"""\n',
        encoding="utf-8",
    )
    subprocess.run(["ollama", "create", args.name, "-f", str(modelfile)], check=True)
    print(f"Ready: ollama run {args.name}")


if __name__ == "__main__":
    main()
