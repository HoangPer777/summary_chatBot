# ViT5 Summarization Chatbot

This project demonstrates a fine-tuned ViT5 model for Vietnamese text summarization, wrapped in a FastAPI web interface.

## Prerequisites

- **Python 3.10+**
- **CUDA-enabled GPU** (Recommended: NVIDIA GPU with drivers installed) for faster inference. 
- **4GB+ VRAM** (tested on RTX 3050 4GB).

## Setup

1.  **Clone the repository** (if not already):
    ```bash
    git clone https://github.com/HoangPer777/summary_chatBot.git
    cd summary_chatBot
    ```

2.  **Create and Activate Virtual Environment**:
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Linux/Mac
    source .venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    # For CUDA 12.1 support (adjust based on your driver):
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Install other requirements
    pip install -r requirements.txt
    ```

4.  **Download Models**:
    The system attempts to download models automatically on first run, but you can force it manually if needed:
    ```bash
    python download_models.py
    ```
    *Note: If you encounter `LoraConfig` errors, ensure your `adapter_config.json` is compatible with the installed `peft` version (unsupported keys like `lora_bias` should be removed).*

## Running the Application

1.  **Start the Server**:
    ```bash
    python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
    ```

2.  **Access the Web Interface**:
    Open your browser to: [http://localhost:8000](http://localhost:8000)

## API Usage

**Endpoint**: `POST /api/summarize`

**Body**:
```json
{
  "text": "Your long Vietnamese text here...",
  "max_length": 128,
  "min_length": 16
}
```

## Project Structure

- `main.py`: FastAPI backend and model inference logic.
- `templates/index.html`: Simple frontend UI.
- `models/`: Directory where models are cached locally.
  - **`vit5-base/`**: Base pre-trained model files.
    - `config.json`
    - `generation_config.json`
    - `model.safetensors`
    - `special_tokens_map.json`
    - `spiece.model`
    - `tokenizer.json`
    - `tokenizer_config.json`
  - **`vit5-fast/`**: Fine-tuned Adapter (LoRA) files.
    - `adapter_config.json`
    - `adapter_model.safetensors`
- `download_models.py`: Helper script to download models.
