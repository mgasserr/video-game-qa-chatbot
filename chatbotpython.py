# Run this once in your terminal before executing the script:

# # 1. Create and activate a virtual env (Python 3.10/3.11)
# py -3.11 -m venv venv
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
# venv\Scripts\activate

# # 2. Install PyTorch with CUDA support (check pytorch.org for exact command matching your CUDA version)
# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132

# # 3. Install the rest
# pip install -U transformers datasets trl peft accelerate bitsandbytes kagglehub

# # 4. Run the script
# python chatbot.py

import os
import glob
import random

import numpy as np
import torch
import pandas as pd
import kagglehub

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def main():
    # =========================================================
    # 1) Download the FULL dataset via KaggleHub (no hard-coded path)
    # =========================================================
    path = kagglehub.dataset_download("ujjwalaggarwal402/video-games-dataset")
    print("Path to dataset files:", path)

    # Auto-discover the CSV inside the downloaded folder instead of assuming a filename.
    # We pick the largest CSV, since a dataset folder can sometimes contain small
    # auxiliary files alongside the main data file.
    csv_candidates = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
    if not csv_candidates:
        raise FileNotFoundError(f"No CSV file found under {path}")
    csv_path = max(csv_candidates, key=os.path.getsize)
    print("Using CSV file:", csv_path)

    df = pd.read_csv(csv_path)
    print("Columns found:", list(df.columns))
    print("Rows found:", len(df))

    # =========================================================
    # 2) Robustly map dataset columns (case-insensitive, several accepted
    #    names per field) instead of hard-coding exact column names.
    # =========================================================
    def find_col(candidates):
        lower_map = {c.lower().strip(): c for c in df.columns}
        for cand in candidates:
            if cand in lower_map:
                return lower_map[cand]
        return None

    game_col      = find_col(["title", "name", "game", "game_title"])
    release_col   = find_col(["release_date", "release date", "released", "year"])
    rating_col    = find_col(["critic_score", "critic score", "score", "metascore", "rating"])
    genre_col     = find_col(["genre", "genres"])
    platform_col  = find_col(["console", "platform", "platforms"])
    publisher_col = find_col(["publisher", "publishers"])
    developer_col = find_col(["developer", "developers", "dev"])

    if game_col is None:
        raise ValueError(
            "Could not find a game-title column. "
            f"Columns available: {list(df.columns)}"
        )

    print("Required  -> game_col:", game_col, "| release_col:", release_col, "| rating_col:", rating_col)
    print("Optional  -> genre:", genre_col, "platform:", platform_col,
        "publisher:", publisher_col, "developer:", developer_col)

    # =========================================================
    # 3) Keep the FULL dataset. We only drop rows with a missing game name,
    #    since every training example needs a title. We do NOT cap the
    #    dataset or drop rows just because one optional field is empty —
    #    each row simply contributes fewer QA pairs if some fields are missing.
    # =========================================================
    df = df.dropna(subset=[game_col]).reset_index(drop=True)
    print(f"Usable rows (full dataset, no 10K cap): {len(df)}")


    # =========================================================
    # 4) Build NATURAL, VARIED conversational training pairs
    #    (no rigid "Question: / Answer:" template — many phrasings per fact,
    #    and facts only come from the dataset itself).
    # =========================================================
    def clean(v):
        if pd.isna(v):
            return None
        s = str(v).strip()
        return s if s and s.lower() != "nan" else None

    release_q = [
        "When did {g} come out?", "What year was {g} released?",
        "Do you know the release date of {g}?", "When was {g} released?",
        "Can you tell me when {g} launched?",
    ]
    release_a = [
        "{g} was released on {v}.", "{g} came out on {v}.", "{g} launched on {v}.",
    ]

    score_q = [
        "What's the critic score for {g}?", "How was {g} rated by critics?",
        "What rating did {g} get?", "Do you know how well {g} scored with critics?",
        "What's the critic rating of {g}?",
    ]
    score_a = [
        "{g} has a critic score of {v}.", "{g} was rated {v} by critics.",
        "Critics gave {g} a score of {v}.",
    ]

    genre_q = [
        "What genre is {g}?", "What kind of game is {g}?", "Which genre does {g} belong to?",
    ]
    genre_a = [
        "{g} is a {v} game.", "{g} falls under the {v} genre.",
    ]

    platform_q = [
        "What platform can I play {g} on?", "Which platform is {g} available on?",
        "What console is {g} on?",
    ]
    platform_a = [
        "{g} is available on {v}.", "You can play {g} on {v}.",
    ]

    publisher_q = ["Who published {g}?", "What company published {g}?"]
    publisher_a = ["{g} was published by {v}."]

    developer_q = ["Who developed {g}?", "What studio made {g}?"]
    developer_a = ["{g} was developed by {v}."]

    about_q = [
        "Tell me about {g}.", "What do you know about {g}?",
        "Give me some info on {g}.", "Do you know anything about this game: {g}?",
    ]

    examples = []

    def add_pair(q_pool, a_pool, game, value):
        q = random.choice(q_pool).format(g=game)
        a = random.choice(a_pool).format(g=game, v=value)
        examples.append((q, a))

    for row in df.itertuples(index=False):
        game = clean(getattr(row, game_col))
        if not game:
            continue

        date      = clean(getattr(row, release_col)) if release_col else None
        score     = clean(getattr(row, rating_col)) if rating_col else None
        genre     = clean(getattr(row, genre_col)) if genre_col else None
        platform  = clean(getattr(row, platform_col)) if platform_col else None
        publisher = clean(getattr(row, publisher_col)) if publisher_col else None
        developer = clean(getattr(row, developer_col)) if developer_col else None

        if date:
            add_pair(release_q, release_a, game, date)
        if score:
            add_pair(score_q, score_a, game, score)
        if genre:
            add_pair(genre_q, genre_a, game, genre)
        if platform:
            add_pair(platform_q, platform_a, game, platform)
        if publisher and random.random() < 0.5:
            add_pair(publisher_q, publisher_a, game, publisher)
        if developer and random.random() < 0.5:
            add_pair(developer_q, developer_a, game, developer)

        # A natural "tell me about it" summary built from whichever facts exist
        facts = []
        if genre:
            facts.append(f"a {genre} game")
        if platform:
            facts.append(f"available on {platform}")
        if date:
            facts.append(f"released on {date}")
        if score:
            facts.append(f"holding a critic score of {score}")
        if developer:
            facts.append(f"developed by {developer}")
        if publisher:
            facts.append(f"published by {publisher}")

        if facts:
            if len(facts) > 1:
                summary = f"{game} is " + ", ".join(facts[:-1]) + " and " + facts[-1] + "."
            else:
                summary = f"{game} is " + facts[0] + "."
            q = random.choice(about_q).format(g=game)
            examples.append((q, summary))

    print(f"Generated {len(examples)} in-domain QA pairs from {len(df)} games.")

    # =========================================================
    # 5) Teach the model to stay in scope: a modest number of
    #    off-topic and unknown-game refusal examples, built only from
    #    generic phrasing (not fabricated game facts).
    # =========================================================
    off_topic_questions = [
        "What's the weather like today?", "Can you help me with my math homework?",
        "What's your favorite movie?", "Tell me a joke about cats.",
        "How do I bake a chocolate cake?", "What's the capital of France?",
        "Can you write me a poem about the ocean?", "What's the meaning of life?",
        "How do I fix my car engine?", "What's in the news today?",
    ]
    refusal_answers = [
        "I'm focused on video game info from my dataset, so I can't really help with that.",
        "That's outside what I know — I'm built specifically to answer questions about video games.",
        "I don't have information on that topic; I only know about the games in my dataset.",
        "Sorry, that's not something I can help with — my knowledge is limited to video games.",
    ]
    n_refusals = max(200, len(examples) // 200)
    for _ in range(n_refusals):
        q = random.choice(off_topic_questions)
        a = random.choice(refusal_answers)
        examples.append((q, a))

    fake_word_bank = [
        "Shadow", "Crystal", "Eternal", "Rogue", "Iron", "Silent", "Frozen", "Neon",
        "Lost", "Broken", "Realm", "Legends", "Odyssey", "Nexus", "Horizon",
        "Chronicles", "Empire", "Void", "Rising", "Storm",
    ]
    unknown_q = [
        "Do you know anything about {g}?", "What can you tell me about {g}?",
        "Have you heard of {g}?",
    ]
    unknown_a = [
        "I don't have any information about {g} in my dataset.",
        "{g} isn't in my dataset, so I can't tell you anything about it.",
        "I don't know anything about {g} — it's not part of the data I was trained on.",
    ]
    existing_titles = set(df[game_col].dropna().astype(str).str.lower())
    n_unknown = max(200, len(examples) // 300)
    made = 0
    while made < n_unknown:
        fake_game = f"{random.choice(fake_word_bank)} {random.choice(fake_word_bank)}"
        if fake_game.lower() in existing_titles:
            continue
        q = random.choice(unknown_q).format(g=fake_game)
        a = random.choice(unknown_a).format(g=fake_game)
        examples.append((q, a))
        made += 1

    print(f"Added {n_refusals} off-topic + {n_unknown} unknown-game refusal examples.")
    print(f"Total training examples: {len(examples)}")

    # =========================================================
    # 6) Assemble into chat-format conversations and split
    # =========================================================
    SYSTEM_PROMPT = (
        "You are a friendly video game assistant. You answer questions using only "
        "the video game database you were trained on. If something isn't in your "
        "data, say so honestly instead of guessing."
    )

    random.shuffle(examples)

    data_rows = [
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]
        }
        for q, a in examples
    ]

    dataset = Dataset.from_list(data_rows)
    split_dataset = dataset.train_test_split(test_size=0.02, seed=SEED)
    train_data = split_dataset["train"]
    val_data = split_dataset["test"]

    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")


    model_id = "Qwen/Qwen3-1.7B"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- Precision fix: detect what the actual Colab GPU supports instead of ---
    # --- blindly assuming BF16 (older GPUs like the T4 don't support it).    ---
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Using compute dtype: {compute_dtype} (bf16 supported: {bf16_ok})")

    # Configure 4-bit Quantization (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    # Load as a Causal Language Model for text-only chat
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    # Now that the tokenizer (and its chat template) is loaded, turn the
    # "messages" conversations built earlier into the actual training text.
    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    train_data = train_data.map(format_chat, remove_columns=["messages"])
    val_data = val_data.map(format_chat, remove_columns=["messages"])

    print(train_data[0]["text"][:400])


    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA parameters.
    # r=32 (alpha=64) targets ~35-40M trainable params on this ~1.7B model
    # (roughly 2% of base params) — a clear step up from the previous ~10M
    # without pushing into full-fine-tune territory or risking VRAM/overfitting
    # problems on a single Colab T4 for a ~64K-row dataset. See the printed
    # trainable-parameter count in the next cell.
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    model = get_peft_model(model, peft_config)


    model.print_trainable_parameters()


    from trl import SFTConfig, SFTTrainer

    # 1. Enforce cache disabling (Crucial for gradient checkpointing)
    model.config.use_cache = False

    training_args = SFTConfig(
        output_dir="./qwen-videogames-finetuned",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,       # effective batch size 32
        optim="paged_adamw_32bit",
        logging_steps=25,
        learning_rate=2e-4,
        bf16=bf16_ok,                        # precision picked automatically above
        fp16=not bf16_ok,
        max_grad_norm=0.3,
        num_train_epochs=1,                  # ~64K games x several QA pairs each is already
                                            # a large number of steps for one epoch
        eval_strategy="no",
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,                  # checkpoint periodically without filling the disk,
                                            # protects a long Colab run against disconnects
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        dataset_text_field="text",
        max_length=384,
        packing=True,                        # pack short QA examples together -> far less
                                            # wasted padding compute over 64K rows
        dataloader_num_workers=0,
        dataset_num_proc=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
        args=training_args,
    )

    # Execute Fine-Tuning
    trainer.train()

    # Save final artifacts
    trainer.save_model("qwen-videogames-final")
    tokenizer.save_pretrained("qwen-videogames-final")

    print("Video Game QA Fine-tuning complete!")

if __name__ == "__main__":
    main()