"""Quick look at the tuned model: answer a few prompts with and without the adapter.

Not the eval (that's Step 6, through Ollama); just a sanity check right after training
that the adapter changed behavior in the direction we wanted and didn't break Korean.

    uv run python -m train.generate                               # default prompts, outputs/sft/final
    uv run python -m train.generate --adapter outputs/sft/checkpoint-300 "hablar 접속법 현재 nosotros?"
"""

import argparse
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoTokenizer, BitsAndBytesConfig

from data.common import SYSTEM_PROMPT
from train.sft import BASE, ROOT

# Not from the eval set: these are for eyeballing, the eval stays unseen.
PROMPTS = [
    "tener 단순과거 nosotros 형태? 답만요.",
    "Ayer yo va al cine con mi amiga. 이거 맞아요?",
    "Yo estoy cansada hoy. 틀린 데 있어요?",
    "por랑 para 차이를 한국어로 쉽게 설명해 주세요.",
    "ll 발음은 나라마다 다른가요?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts", nargs="*")
    ap.add_argument("--adapter", default=str(ROOT / "outputs" / "sft" / "final"))
    ap.add_argument("--max-new-tokens", type=int, default=700)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForMultimodalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16),
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    for prompt in args.prompts or PROMPTS:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                               return_dict=True).to(model.device)
        print(f"\n{'=' * 70}\nQ: {prompt}")
        for label, ctx in (("base", model.disable_adapter()), ("tuned", torch.no_grad())):
            with ctx, torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            text = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            print(f"\n--- {label} ---\n{text}")


if __name__ == "__main__":
    main()
