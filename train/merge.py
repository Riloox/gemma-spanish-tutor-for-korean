"""Step 4a: fold the LoRA adapter back into the full-precision base weights.

Training used a 4-bit base, but we merge into the bf16 original: W' = W + B·A (scaled by alpha/r).
The result is a normal Hugging Face checkpoint with exactly the same tensors as google/gemma-4-E2B-it,
so Ollama can import and quantize it like the stock model.

Why not just `save_pretrained`: layers 15-34 reuse the key/value cache of earlier layers, so
transformers never builds their k_proj/v_proj/k_norm and drops those 60 tensors when saving.
They're unused, but converters expect the stock tensor list, so we copy them from the base file.

Runs on the CPU (~20 GB of RAM, a couple of minutes), so the GPU can stay busy with Ollama.

    uv run python -m train.merge                                   # outputs/sft/final -> outputs/merged
    uv run python -m train.merge --adapter outputs/sft/checkpoint-400 --out outputs/merged-400
"""

import argparse
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForMultimodalLM

from train.sft import BASE, ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=str(ROOT / "outputs" / "sft" / "final"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "merged"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    base_dir = Path(snapshot_download(BASE))
    model = AutoModelForMultimodalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": "cpu"})
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    merged = model.state_dict()

    tensors, changed = {}, 0
    with safe_open(base_dir / "model.safetensors", "pt") as base:
        for key in base.keys():
            if key in merged:
                tensors[key] = merged[key].contiguous()
                changed += not torch.equal(tensors[key], base.get_tensor(key))
            else:
                tensors[key] = base.get_tensor(key)  # the unused KV-shared tensors
    save_file(tensors, out / "model.safetensors", metadata={"format": "pt"})
    print(f"{len(tensors)} tensors, {changed} changed by the adapter")

    # Config, tokenizer, chat template and image/audio processor configs: the base's, unchanged.
    for f in base_dir.iterdir():
        if f.suffix in (".json", ".jinja"):
            shutil.copy(f, out)
    print(f"merged model saved to {out}")


if __name__ == "__main__":
    main()
