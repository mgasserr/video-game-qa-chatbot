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
import sys

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
    # 1) Download the FULL dataset via KaggleHub
    # =========================================================
    path = kagglehub.dataset_download("ujjwalaggarwal402/video-games-dataset")
    print("Path to dataset files:", path)

    csv_candidates = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
    if not csv_candidates:
        raise FileNotFoundError(f"No CSV file found under {path}")
    csv_path = max(csv_candidates, key=os.path.getsize)
    print("Using CSV file:", csv_path)

    df = pd.read_csv(csv_path)

    # =========================================================
    # 2) Robustly map dataset columns
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
        raise ValueError("Could not find a game-title column.")

    df = df.dropna(subset=[game_col]).reset_index(drop=True)
    print(f"Usable rows (full dataset, no 10K cap): {len(df)}")

    # Parse dates accurately to identify "original" vs "port" releases
    if release_col:
        df['parsed_date'] = pd.to_datetime(df[release_col], errors='coerce', dayfirst=True)
    else:
        df['parsed_date'] = pd.NaT

    # =========================================================
    # 3) LINGUISTIC POOLS (Intent over rigid templates)
    # =========================================================
    global_release_q = [
        "When did {g} come out?", "What year was {g} released?",
        "Do you know the release date of {g}?", "When was {g} released?",
        "Can you tell me when {g} launched?", "When did {g} drop?",
        "What was the original release date for {g}?",
        "When did the first {g} game come out?", "Tell me when {g} was first available.",
    ]
    global_release_a = [
        "The original release of {g} was on {date} for {consoles}.",
        "{g} first launched on {date} (available on {consoles}).",
        "It originally came out on {date} for {consoles}.",
        "According to the database, {g} was first released on {date} on {consoles}.",
    ]
    
    plat_release_q = [
        "When did {g} come out on {c}?", "When was {g} released for {c}?",
        "What was the {c} release date for {g}?", "When did the {c} version of {g} launch?",
        "Do you know when the {c} version of {g} came out?", "When was the {c} port of {g} released?",
        "When did {g} drop on {c}?", "What date did {g} launch for {c}?",
        "When did the {c} version drop?", "{g} {c} release date?",
    ]
    plat_release_a = [
        "The {c} version of {g} was released on {date}.",
        "For {c}, {g} launched on {date}.",
        "It came out on {c} on {date}.",
        "According to the database, the {c} release of {g} was {date}.",
        "{g} dropped on {c} on {date}.",
    ]
    
    plat_score_q = [
        "What is the critic score for {g} on {c}?", "How did {g} rate on {c}?",
        "What score did {g} get on {c}?", "Is the {c} version of {g} rated well?",
        "Critic rating for {g} {c}?",
    ]
    plat_score_a = [
        "{g} has a critic score of {score} on {c}.",
        "On {c}, {g} received a critic score of {score}.",
        "Critics gave the {c} version of {g} a score of {score}.",
    ]
    
    dev_q = ["Who developed {g}?", "What studio made {g}?", "Who created {g}?", "Which developer is behind {g}?"]
    dev_a = ["{g} was developed by {dev}.", "The studio behind {g} is {dev}.", "{dev} created {g}."]
    
    pub_q = ["Who published {g}?", "What company published {g}?", "Who is the publisher of {g}?"]
    pub_a = ["{g} was published by {pub}.", "The publisher for {g} is {pub}."]
    
    combo_q = [
        "Who developed {g} and what genre is it?",
        "Can you tell me the genre and developer of {g}?",
        "What kind of game is {g} and who made it?",
    ]
    combo_a = [
        "{g} is a {genre} game developed by {dev}.",
        "It was developed by {dev} and belongs to the {genre} genre.",
    ]
    
    about_q = ["Tell me about {g}.", "What do you know about {g}?", "Give me some info on {g}.", "What is {g}?"]
    
    follow_plat_q = [
        "What about the {c} version?", "When did it come out on {c}?",
        "Did it release on {c}?", "What was the {c} release date?",
        "When did the {c} port drop?", "And on {c}?",
    ]

    # =========================================================
    # 4) BUILD RELATIONAL QA PAIRS
    # =========================================================
    examples_single = []
    examples_multi = []
    validation_pool = []
    
    grouped = df.groupby(game_col)
    
    for game, group in grouped:
        game_str = str(game).strip()
        if not game_str or game_str.lower() == 'nan':
            continue
            
        group = group.sort_values('parsed_date')
        
        # Aggregate global facts across all platforms for this specific game
        devs = group[developer_col].dropna().unique() if developer_col else []
        dev = devs[0] if len(devs) > 0 else None
        
        pubs = group[publisher_col].dropna().unique() if publisher_col else []
        pub = pubs[0] if len(pubs) > 0 else None
        
        genres = group[genre_col].dropna().unique() if genre_col else []
        genre = genres[0] if len(genres) > 0 else None
        
        all_consoles = [str(c) for c in group[platform_col].dropna().unique() if str(c).lower() != 'all']
        all_consoles_str = ", ".join(all_consoles)
        
        # Identify the original/first release for generic queries
        valid_dates = group.dropna(subset=['parsed_date'])
        if not valid_dates.empty:
            first_row = valid_dates.iloc[0]
            first_date_str = first_row[release_col]
            first_parsed = first_row['parsed_date']
            first_consoles = valid_dates[valid_dates['parsed_date'] == first_parsed][platform_col].unique()
            first_consoles = [str(c) for c in first_consoles if str(c).lower() != 'all']
            first_consoles_str = ", ".join(first_consoles) if first_consoles else "various platforms"
        else:
            first_date_str = None
            first_consoles_str = None
            
        # -- GLOBAL QUESTIONS (No platform specified) --
        if first_date_str:
            q = random.choice(global_release_q).format(g=game_str)
            a = random.choice(global_release_a).format(g=game_str, date=first_date_str, consoles=first_consoles_str)
            examples_single.append((q, a))
            
        if dev:
            examples_single.append((random.choice(dev_q).format(g=game_str), random.choice(dev_a).format(g=game_str, dev=dev)))
        else:
            examples_single.append((random.choice(dev_q).format(g=game_str), f"I don't have the developer information for {game_str} in my database."))
            
        if pub:
            examples_single.append((random.choice(pub_q).format(g=game_str), random.choice(pub_a).format(g=game_str, pub=pub)))
            
        if dev and genre:
            examples_single.append((random.choice(combo_q).format(g=game_str), random.choice(combo_a).format(g=game_str, genre=genre, dev=dev)))

        # -- PLATFORM-SPECIFIC QUESTIONS --
        for _, row in group.iterrows():
            console = str(row[platform_col]) if platform_col and pd.notna(row[platform_col]) else ""
            if not console or console.lower() == 'all':
                continue
                
            date = str(row[release_col]) if release_col and pd.notna(row[release_col]) else None
            score = str(row[rating_col]) if rating_col and pd.notna(row[rating_col]) else None
            
            if date:
                q = random.choice(plat_release_q).format(g=game_str, c=console)
                a = random.choice(plat_release_a).format(g=game_str, c=console, date=date)
                examples_single.append((q, a))
                
                # Send 5% of platform queries to the validation pool
                if random.random() < 0.05:
                    validation_pool.append({
                        "q": q, "a": a, "expected_date": date, "game": game_str, "console": console
                    })
                    
            if score:
                q = random.choice(plat_score_q).format(g=game_str, c=console)
                a = random.choice(plat_score_a).format(g=game_str, c=console, score=score)
                examples_single.append((q, a))
            else:
                examples_single.append((random.choice(plat_score_q).format(g=game_str, c=console), f"I don't have a critic score recorded for {game_str} on {console}."))

        # -- MISSING DATA / NEGATIVE SAMPLES --
        fake_consoles = ["Nintendo Switch", "PlayStation 5", "Xbox Series X", "PC", "Mobile", "Atari", "GameCube"]
        if all_consoles:
            missing_console = random.choice(fake_consoles)
            if missing_console.lower() not in [c.lower() for c in all_consoles]:
                q = random.choice(plat_release_q).format(g=game_str, c=missing_console)
                a = f"I don't have any record of {game_str} releasing on {missing_console} in the database."
                examples_single.append((q, a))

        # -- CONVERSATIONAL MULTI-TURN (Contextual Pronouns) --
        if first_date_str and len(valid_dates) > 1:
            later_rows = valid_dates[valid_dates['parsed_date'] > first_parsed]
            if not later_rows.empty:
                later_row = later_rows.sample(1).iloc[0]
                later_console = str(later_row[platform_col])
                later_date = str(later_row[release_col])
                
                if later_console.lower() != 'all':
                    msg = [
                        {"role": "user", "content": random.choice(about_q).format(g=game_str)},
                        {"role": "assistant", "content": f"{game_str} is a {genre or 'video'} game. It originally launched on {first_date_str}."},
                        {"role": "user", "content": random.choice(follow_plat_q).format(c=later_console)},
                        {"role": "assistant", "content": random.choice(plat_release_a).format(g=game_str, c=later_console, date=later_date)}
                    ]
                    examples_multi.append(msg)

    # =========================================================
    # 5) ASSEMBLE MESSAGES & WORLD KNOWLEDGE REFUSALS
    # =========================================================
    SYSTEM_PROMPT = (
        "You are a friendly video game assistant. You answer questions using only "
        "the video game database you were trained on. If something isn't in your "
        "data, say so honestly instead of guessing."
    )

    messages_list = []
    
    for q, a in examples_single:
        messages_list.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a}
        ])
        
    for multi in examples_multi:
        messages_list.append([{"role": "system", "content": SYSTEM_PROMPT}] + multi)

    # Include off-topic generic refusals to block general world-knowledge
    off_topic_q = [
        "What's the weather like?", "How do I bake a cake?", 
        "Tell me a cat joke.", "Who is the current president of Egypt?",
        "What is the capital of France?", "Who won the World Cup?",
        "Explain quantum physics to me.", "How do I fix a car engine?",
        "What is the meaning of life?", "Can you write a poem about the ocean?"
    ]
    refusal_a = [
        "I'm focused strictly on video game info from my dataset, so I can't help with that.",
        "That is outside of my dataset. I only have information regarding video game releases, scores, and developers.",
        "I don't have information on that topic; I only know about the games in my database."
    ]
    for _ in range(300):
        messages_list.append([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": random.choice(off_topic_q)},
            {"role": "assistant", "content": random.choice(refusal_a)}
        ])

    print(f"\nGenerated {len(messages_list)} total conversations.")

    # =========================================================
    # 6) STRICT PRE-TRAINING VALIDATION AUDIT
    # =========================================================
    print("\n" + "="*60)
    print("PRE-TRAINING VALIDATION & SAMPLE REVIEW")
    print("="*60)
    
    passed = 0
    failed = 0
    
    sample = random.sample(validation_pool, min(10, len(validation_pool)))
    
    for ex in sample:
        print(f"User: {ex['q']}")
        print(f"Assistant: {ex['a']}")
        
        expected_date = str(ex['expected_date'])
        if expected_date in ex['a']:
            print(f"Validation: [VALID] Correctly mapped platform date: {expected_date}\n")
            passed += 1
        else:
            print(f"Validation: [INVALID] Expected to find date {expected_date} but didn't!\n")
            failed += 1
            
    print(f"Validation Summary: {passed} Passed, {failed} Failed.")
    
    if failed > 0:
        print("\nFATAL ERROR: Generated training data failed validation constraints.")
        print("Data relationship structures are compromised. Training aborted.")
        sys.exit(1)
        
    print("\nValidation successful! Here is a multi-turn conversation sample:")
    if examples_multi:
        sample_multi = random.choice(examples_multi)
        for turn in sample_multi[1:]:
            print(f"  {turn['role'].capitalize()}: {turn['content']}")
            
    print("="*60 + "\n")

    # =========================================================
    # 7) TOKENIZE AND TRAIN
    # =========================================================
    random.shuffle(messages_list)
    dataset = Dataset.from_list([{"messages": msgs} for msgs in messages_list])
    split_dataset = dataset.train_test_split(test_size=0.02, seed=SEED)
    train_data = split_dataset["train"]
    val_data = split_dataset["test"]

    model_id = "Qwen/Qwen3-1.7B"

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if bf16_ok else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    train_data = train_data.map(format_chat, remove_columns=["messages"])
    val_data = val_data.map(format_chat, remove_columns=["messages"])

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    model = get_peft_model(model, peft_config)

    training_args = SFTConfig(
        output_dir="./qwen-videogames-finetuned",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        optim="paged_adamw_32bit",
        logging_steps=25,
        learning_rate=2e-4,
        bf16=bf16_ok,
        fp16=not bf16_ok,
        max_grad_norm=0.3,
        num_train_epochs=1,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        dataset_text_field="text",
        max_length=384,
        packing=True,
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

    trainer.train()

    trainer.save_model("qwen-videogames-final")
    tokenizer.save_pretrained("qwen-videogames-final")
    print("Video Game QA Fine-tuning complete!")

if __name__ == "__main__":
    main()