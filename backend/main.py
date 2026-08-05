from pathlib import Path
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_DIR = Path(__file__).parent / "model"
BASE_MODEL_NAME = "Qwen/Qwen3.5-2B-Base"

app = FastAPI(title="Video Games Q&A Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.on_event("startup")
def load_model():
    global tokenizer, model
    if not MODEL_DIR.exists():
        print(
            f"[WARNING] Model folder not found at {MODEL_DIR}. "
            "Please copy your Colab adapter files into backend/model/."
        )
        return

    print(f"Loading custom tokenizer from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    print("Loading base Qwen model...")
    # Updated to use `dtype` instead of `torch_dtype`
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map=device,
        dtype=torch.bfloat16 if device == "cpu" else torch.float16,
    )

    print(f"Attaching and MERGING LoRA adapter from {MODEL_DIR}...")
    peft_model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    
    # Crucial for CPU performance
    model = peft_model.merge_and_unload()
    model.eval()
    
    print(f"Video Game Model successfully loaded and merged on {device}!")


class ChatRequest(BaseModel):
    question: str = Field(..., description="The user's question about a video game")


class ChatResponse(BaseModel):
    answer: str


@app.get("/")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = (request.question or "").strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    try:
        # NOTE: Update this prompt structure if you used a different format during training!
        prompt = f"Question: {question}\nAnswer:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        answer = tokenizer.decode(
            outputs[0][input_length:], skip_special_tokens=True
        ).strip()

        if not answer:
            answer = "Sorry, I couldn't generate an answer for that game."

        return ChatResponse(answer=answer)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc