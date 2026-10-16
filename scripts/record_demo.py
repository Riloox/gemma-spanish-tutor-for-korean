"""Record a terminal demo of the app model as an asciicast (.cast), for the README GIF.

Looks like an `ollama run spanish-tutor` session. The learner's typing is simulated at a steady pace; the
tutor's answers are real, streamed from Ollama's /api/chat with the times the tokens arrived. Sampling uses the
model's own settings (temperature 1, as `ollama run` does) with a fixed seed, so a take can be reproduced.

    python scripts/record_demo.py [--model spanish-tutor] [--seed 7] [--out outputs/demo.cast]
    agg --font-dir C:/Windows/Fonts --text-font-family "Cascadia Mono,Malgun Gothic" --theme github-dark \
        --font-size 18 outputs/demo.cast assets/demo.gif

Standard library only.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# One error to fix, then a correct sentence that should be left alone (the stock model "fixes" most of those)
TURNS = [
    '친구가 오늘 아프다고 말하고 싶어요. "Mi amiga es enferma hoy." 맞아요?',
    '"Me gustan mucho los perros de mi vecino." 이 문장도 괜찮아요?',
]

TYPE_DELAY = 0.045  # seconds per typed character
DIM, RESET = "\x1b[2m", "\x1b[0m"
PLACEHOLDER = f">>> {DIM}Send a message (/? for help){RESET}"


class Cast:
    """asciicast v2 writer: a header line, then [time, "o", text] events."""

    def __init__(self, cols, rows):
        self.cols, self.rows = cols, rows
        self.t = 0.0
        self.events = []

    def out(self, text, wait=0.0):
        self.t += wait
        self.events.append([round(self.t, 3), "o", text])

    def type(self, text):
        for ch in text:
            self.out(ch, TYPE_DELAY)

    def save(self, path):
        header = {"version": 2, "width": self.cols, "height": self.rows, "env": {"TERM": "xterm-256color"}}
        lines = [json.dumps(header)] + [json.dumps(e, ensure_ascii=False) for e in self.events]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def stream_chat(host, model, messages, seed):
    """Yield (seconds since request, text chunk) as the answer streams in."""
    body = {"model": model, "messages": messages, "stream": True, "think": False, "options": {"seed": seed}}
    req = urllib.request.Request(f"{host}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            chunk = json.loads(line)
            text = chunk.get("message", {}).get("content", "")
            if text:
                yield time.monotonic() - start, text
            if chunk.get("done"):
                return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="spanish-tutor")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cols", type=int, default=96)
    ap.add_argument("--rows", type=int, default=31)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "demo.cast")
    args = ap.parse_args()

    # Load the model first so the recording doesn't include Ollama's cold start
    list(stream_chat(args.host, args.model, [{"role": "user", "content": "hola"}], args.seed))

    cast = Cast(args.cols, args.rows)
    cast.out("$ ", 0.3)
    cast.type(f"ollama run {args.model}")
    cast.out("\r\n", 0.4)

    messages = []
    for i, prompt in enumerate(TURNS):
        cast.out(PLACEHOLDER + "\r>>> \x1b[K", 0.6 if i == 0 else 1.0)  # placeholder vanishes on the first key
        cast.type(prompt)
        cast.out("\r\n", 0.5)

        messages.append({"role": "user", "content": prompt})
        answer, last = "", 0.0
        for t, text in stream_chat(args.host, args.model, messages, args.seed):
            answer += text
            cast.out(text.replace("\n", "\r\n"), t - last)
            last = t
        messages.append({"role": "assistant", "content": answer})
        cast.out("\r\n\r\n")
        print(f"--- turn {i + 1} ({last:.1f}s) ---\n{answer}\n")

    cast.out(PLACEHOLDER + "\r\x1b[4C", 0.3)  # cursor back at the start of the input
    cast.out("", 4.0)  # hold the last frame
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cast.save(args.out)
    print(f"Wrote {args.out} ({cast.t:.0f}s)")


if __name__ == "__main__":
    main()
