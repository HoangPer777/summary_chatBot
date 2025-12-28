import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from .config import (
    BASE_MODEL_ID, ADAPTER_ID, ADAPTER_REVISION,
    BASE_LOCAL_DIR, ADAPTER_LOCAL_DIR, DEVICE,
    is_offline_mode
)

class ModelService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = DEVICE

    def load_model(self):
        """Load model from local cache if available; else download and persist locally."""
        print(f"Loading model on {self.device}...")
        
        is_base_local = BASE_LOCAL_DIR.exists() and (BASE_LOCAL_DIR / "config.json").exists()
        is_adapter_local = ADAPTER_LOCAL_DIR.exists() and (ADAPTER_LOCAL_DIR / "adapter_config.json").exists()

        # Load Base Model
        if is_base_local:
            print(f"Loading base from {BASE_LOCAL_DIR}")
            self.tokenizer = AutoTokenizer.from_pretrained(str(BASE_LOCAL_DIR))
            base = AutoModelForSeq2SeqLM.from_pretrained(str(BASE_LOCAL_DIR))
        else:
            if is_offline_mode():
                raise RuntimeError("Offline mode enabled but base model not found.")
            print(f"Downloading base from {BASE_MODEL_ID}")
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
            base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_ID)
            BASE_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
            self.tokenizer.save_pretrained(str(BASE_LOCAL_DIR))
            base.save_pretrained(str(BASE_LOCAL_DIR))

        # Load Adapter
        if is_adapter_local:
            print(f"Loading adapter from {ADAPTER_LOCAL_DIR}")
            peft_model = PeftModel.from_pretrained(base, str(ADAPTER_LOCAL_DIR), is_trainable=False)
        else:
            if is_offline_mode():
                raise RuntimeError("Offline mode enabled but adapter not found.")
            print(f"Downloading adapter from {ADAPTER_ID}")
            peft_model = PeftModel.from_pretrained(base, ADAPTER_ID, revision=ADAPTER_REVISION, is_trainable=False)
            ADAPTER_LOCAL_DIR.mkdir(parents=True, exist_ok=True)
            # Note: save_pretrained might only save the adapter config/weights
            peft_model.save_pretrained(str(ADAPTER_LOCAL_DIR))

        # Optional Clean Merge
        merge_flag = os.getenv("VIT5_MERGE", "0").lower() in {"1", "true", "yes"}
        if merge_flag:
            peft_model = peft_model.merge_and_unload()

        peft_model.to(self.device)
        peft_model.eval()
        self.model = peft_model
        print("Model loaded successfully.")

    def generate_summary(self, text: str, max_length: int = 128, min_length: int = 16) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded yet")

        inputs = self.tokenizer(
            text,
            max_length=256,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                early_stopping=True,
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

# Global instance
model_service = ModelService()
