# Parameter-Efficient Fine-Tuning Service for Banking77

This project provides a lightweight starter for training and serving a parameter-efficient fine-tuned model on the Banking77 intent-classification dataset.

## Goals
- Load and preprocess the Banking77 dataset
- Fine-tune a transformer model with PEFT/LoRA
- Evaluate on validation/test data
- Expose a simple inference API

## Repository structure
- `app/data.py` — dataset loading and preprocessing
- `app/train.py` — LoRA/PEFT training loop
- `app/inference.py` — prediction utilities
- `app/main.py` — FastAPI service endpoints
- `requirements.txt` — project dependencies
- `README.md` — project overview and usage

## Quick start

1. Create a virtual environment
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model
   ```bash
   python -m app.train
   ```

4. Run the inference service
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

5. Call the API
   ```bash
   curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "I need to check my account balance."}'
   ```

## Notes
- The default configuration uses a small transformer with LoRA adapters.
- You can swap in a larger backbone by changing the model name in `app/train.py`.
- Banking77 is a multi-class intent classification dataset; metrics such as accuracy and macro-F1 are recommended.
