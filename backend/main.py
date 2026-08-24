import re
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_NAME = "Qwen/Qwen3-1.7B"

# Must remain EXACTLY the same as the training script
SYSTEM_PROMPT = (
    "You are a friendly and helpful video game assistant. Answer clearly, and "
    "if you're uncertain about a fact, say so instead of guessing."
)

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
eos_ids = None
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.on_event("startup")
def load_model():
    global tokenizer, model, eos_ids
    print(f"Loading tokenizer from {BASE_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"Loading Qwen model ({BASE_MODEL_NAME}) on {device} as {dtype}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_ids = [tokenizer.eos_token_id]
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        eos_ids.append(im_end_id)

    print(f"Video Game Model successfully loaded on {device}!")

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
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=eos_ids,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        raw_answer = tokenizer.decode(
            outputs[0][input_length:], skip_special_tokens=True
        ).strip()

        # =====================================================================
        # RIGOROUS <think> TAG STRIPPING
        # Strips out any internal <think>...</think> reasoning blocks.
        # =====================================================================
        clean_answer = re.sub(r'<think>.*?</think>', '', raw_answer, flags=re.DOTALL).strip()
        
        # Fallback if a <think> tag is generated but cut off before </think>
        if "<think>" in clean_answer:
            clean_answer = clean_answer.split("<think>")[0].strip()

        if not clean_answer:
            clean_answer = "Sorry, I couldn't generate an answer for that game."

        return ChatResponse(answer=clean_answer)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc