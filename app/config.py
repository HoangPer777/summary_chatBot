import os
import torch
from pathlib import Path

# Model IDs
BASE_MODEL_ID = "VietAI/vit5-base"
ADAPTER_ID = "lamdoanh2468/vit5-fast"
ADAPTER_REVISION = "ac6802f"

# Local cache directories
BASE_LOCAL_DIR = Path(os.getenv("VIT5_BASE_DIR", "models/vit5-base"))
ADAPTER_LOCAL_DIR = Path(os.getenv("VIT5_ADAPTER_DIR", "models/vit5-fast"))

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def is_offline_mode() -> bool:
    """Detect offline mode via common env vars for Transformers/HF Hub."""
    flags = [
        os.getenv("TRANSFORMERS_OFFLINE"),
        os.getenv("HF_HUB_OFFLINE"),
        os.getenv("HUGGINGFACE_HUB_OFFLINE"),
    ]
    return any(str(v).lower() in {"1", "true", "yes"} for v in flags if v is not None)
