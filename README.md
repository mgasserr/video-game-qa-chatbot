# PC Components Q&A Chatbot

A fine-tuned decoder-LLM chatbot that answers questions about PC hardware
(CPU, GPU, RAM, motherboard, PSU, case, cooler, storage, OS).

## 1. Problem & dataset

- **Domain:** PC hardware component specs (targets students/enthusiasts building or
  upgrading a PC and IT/e-commerce support use cases).
- **Dataset:** [PC Parts — Kaggle](https://www.kaggle.com/datasets/warcoder/pc-parts)
  ("Data of various pc parts with their main features and price", 25 categories:
  CPU, GPU, RAM, motherboard, PSU, case, cooler, storage, OS, etc.)
- This is a **spec table**, not a QA dataset out of the box. `dataset/build_dataset.py`
  converts every `(part, attribute, value)` triple into natural-language QA pairs
  using templates (e.g. *"What is the TDP of the RTX 4070?" → "The TDP of the RTX
  4070 is 200W."*), then splits into train/validation/test (80/10/10). This is the
  standard way to bootstrap a QA dataset from tabular data.

## 2. Model

- **GPT-2 (small)**, decoder-only, fine-tuned with **LoRA** (via `peft`) — chosen so
  training fits comfortably on a free Colab T4 GPU. Swap `MODEL_NAME` in the notebook
  for `Qwen/Qwen2.5-0.5B` or similar if you want a stronger base model and have GPU
  headroom (same LoRA recipe applies, just change `target_modules` per that model's
  attention layer names).

## 3. Project layout

```
pc-chatbot/
├── colab_finetune.ipynb     # Run this in Google Colab to train the model
├── dataset/
│   └── build_dataset.py     # Standalone/local version of the QA-pair generator
├── backend/
│   ├── main.py               # FastAPI app, POST /api/chat
│   ├── requirements.txt
│   └── model/                 # <- put your fine-tuned model here (see step 5)
└── frontend/
    ├── package.json
    ├── index.html
    └── src/ (App.jsx, App.css, main.jsx)
```

## 4. Train the model on Google Colab

1. Go to [colab.research.google.com](https://colab.research.google.com), upload
   `colab_finetune.ipynb`.
2. `Runtime → Change runtime type → T4 GPU`.
3. Run every cell top to bottom. The first Kaggle-download cell will prompt you to
   sign in / authorize Kaggle access the first time (it uses `kagglehub`, no manual
   API key file needed).
4. Training takes a few minutes on a T4. At the end you'll see the before/after
   fine-tuning comparison and Exact Match / ROUGE / BLEU / BERTScore numbers.
5. The last cell downloads `final_model.zip` to your computer automatically.

## 5. Full local setup (after training)

### Prerequisites
- Python 3.10+ and `pip`
- Node.js 18+ and `npm`
- ~2 GB free disk space for the model + Python packages

### 5.1 Get the project files onto your PC
Unzip this project folder anywhere, e.g. `C:\Projects\pc-chatbot` (Windows) or
`~/pc-chatbot` (Linux/Mac).

### 5.2 Place the fine-tuned model
Unzip `final_model.zip` (downloaded from Colab) and copy its contents into:
```
pc-chatbot/backend/model/
```
So you end up with files like `backend/model/config.json`,
`backend/model/model.safetensors`, `backend/model/tokenizer.json`, etc.

### 5.3 Backend (FastAPI)

```bash
cd pc-chatbot/backend

# create & activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt

uvicorn main:app --reload --port 8000
```

Check it's alive: open http://localhost:8000/ — you should see
`{"status":"ok","model_loaded":true}`. Interactive API docs: http://localhost:8000/docs

**GPU vs CPU:** the backend auto-detects CUDA (`torch.cuda.is_available()`) and falls
back to CPU automatically — GPT-2-small inference is fast enough on CPU, no GPU
required on your PC for serving. If you do have an NVIDIA GPU and want CUDA inference,
install the matching CUDA build of PyTorch from https://pytorch.org/get-started/locally/
instead of the plain `torch` in requirements.txt.

### 5.4 Frontend (React)

Open a **second terminal** (leave the backend running in the first one):

```bash
cd pc-chatbot/frontend
npm install
npm run dev
```

Open the URL it prints (default http://localhost:5173) in your browser. Type a
question and hit Send — the frontend calls `http://localhost:8000/api/chat`.

### 5.5 Common issues

| Symptom | Fix |
|---|---|
| "Couldn't get an answer... Is the FastAPI backend running?" | Make sure `uvicorn` is running on port 8000 in another terminal. |
| `model_loaded: false` on the health check | The `backend/model/` folder is missing or incomplete — re-copy the files from `final_model.zip`. |
| CORS error in the browser console | Confirm the frontend is on port 5173/3000 (already whitelisted in `backend/main.py`); add your port to `allow_origins` if different. |
| `pip install torch` very slow / large | Normal, PyTorch is a large package; use a wired connection or the CPU-only wheel if you don't need CUDA. |

## 6. API reference

**POST** `/api/chat`
```json
{ "question": "What is the price of the Corsair Vengeance RGB?" }
```
Response:
```json
{ "answer": "The Corsair Vengeance RGB costs $79.99." }
```
Empty/missing `question` → `400`. Model not loaded → `503`. Inference error → `500`.

## 7. Report checklist (for submission)

- Problem: PC-hardware QA assistant, target users = builders/upgraders and support desks.
- Dataset prep: see `dataset/build_dataset.py` — template-based QA generation from the
  Kaggle PC Parts spec table, 80/10/10 split.
- Model: GPT-2-small + LoRA (r=8, alpha=32, target `c_attn`).
- Fine-tuning config: 3 epochs, batch size 8, lr 2e-4, fp16 on GPU — see notebook section 8.
- Evaluation: eval/test loss + perplexity, Exact Match, ROUGE, BLEU, BERTScore, plus
  qualitative before/after fine-tuning answer comparisons — notebook sections 6, 9, 10.
- Limitations to mention: GPT-2-small has limited reasoning capacity; template-generated
  QA pairs are less varied than human-written questions; answers are only as accurate as
  the underlying spec table (no live price updates).
