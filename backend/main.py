from pathlib import Path
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Must match the training script exactly:
# - base model: Qwen/Qwen3-1.7B
# - MODEL_DIR should contain the copied contents of "qwen-videogames-final"
#   (the LoRA adapter + tokenizer saved by trainer.save_model / tokenizer.save_pretrained)
MODEL_DIR = Path(__file__).parent / "model"
BASE_MODEL_NAME = "Qwen/Qwen3-1.7B"

# Same system prompt used during training — the model was fine-tuned to
# respond in this persona/format, so inference must use it too.
SYSTEM_PROMPT = (
    "You are a friendly video game assistant. You answer questions using only "
    "the video game database you were trained on. If something isn't in your "
    "data, say so honestly instead of guessing."
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

    if not MODEL_DIR.exists():
        print(
            f"[WARNING] Model folder not found at {MODEL_DIR}. "
            "Copy the contents of 'qwen-videogames-final' into backend/model/."
        )
        return

    print(f"Loading tokenizer from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Precision: mirrors the training script's bf16_ok logic. On CPU we use
    # float32 (fp16/bf16 matmul on CPU is either unsupported or very slow).
    if device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32

    print(f"Loading base Qwen model ({BASE_MODEL_NAME}) on {device} as {dtype}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        device_map=device,
        dtype=dtype,
    )

    print(f"Attaching and merging LoRA adapter from {MODEL_DIR}...")
    peft_model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    model = peft_model.merge_and_unload()
    model.eval()

    # The model was trained to end assistant turns with <|im_end|>, so
    # generation must stop there too, not just at the base eos token.
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos_ids = [tokenizer.eos_token_id]
    if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
        eos_ids.append(im_end_id)

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
        # Build the same chat-template prompt used during training.
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
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
        answer = tokenizer.decode(
            outputs[0][input_length:], skip_special_tokens=True
        ).strip()

        if not answer:
            answer = "Sorry, I couldn't generate an answer for that game."

        return ChatResponse(answer=answer)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc