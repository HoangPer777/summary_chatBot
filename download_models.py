import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Ensure online mode for this download run by clearing offline flags
for _var in ["TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE", "HUGGINGFACE_HUB_OFFLINE"]:
    if os.getenv(_var):
        print(f"[INFO] Detected {_var}={os.getenv(_var)}; temporarily disabling to download.")
        os.environ.pop(_var, None)

BASE_MODEL_ID = "VietAI/vit5-base"
ADAPTER_ID = "lamdoanh2468/vit5-fast"
ADAPTER_REVISION = "ac6802f" # epoch 1

BASE_LOCAL_DIR = Path(os.getenv("VIT5_BASE_DIR", "models/vit5-base"))
ADAPTER_LOCAL_DIR = Path(os.getenv("VIT5_ADAPTER_DIR", "models/vit5-fast"))


def main():
    BASE_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTER_LOCAL_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading base model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_ID)
    tokenizer.save_pretrained(str(BASE_LOCAL_DIR))
    base.save_pretrained(str(BASE_LOCAL_DIR))

    print("Downloading adapter...")
    peft_model = PeftModel.from_pretrained(base, ADAPTER_ID, revision=ADAPTER_REVISION, is_trainable=False)
    peft_model.save_pretrained(str(ADAPTER_LOCAL_DIR))

    print("Done. Local cache:")
    print(f"  Base  -> {BASE_LOCAL_DIR}")
    print(f"  Adapter -> {ADAPTER_LOCAL_DIR}")


if __name__ == "__main__":
    main()
