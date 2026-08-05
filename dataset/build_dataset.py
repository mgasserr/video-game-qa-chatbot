"""
build_dataset.py

Turns the Kaggle "PC Parts" spec table (CPU, GPU, RAM, Motherboard, PSU, Case,
Cooler, Storage, OS, ...) into a question-answering dataset that a decoder LLM
can be fine-tuned on.

Dataset used: https://www.kaggle.com/datasets/warcoder/pc-parts
(25 categories of PC parts with their main specs and price - scraped tabular data,
no license restriction noted, "Data of various pc parts with their main features
and price").

The raw dataset is NOT a QA dataset out of the box - it's a spec table. This
script converts every (part, attribute, value) triple into one or more natural
language QA pairs using templates, which is a standard way to bootstrap a
QA dataset from structured/tabular data.

Usage (works locally OR in Colab, same code):
    python build_dataset.py --input pc_parts.csv --outdir ./qa_data

If you don't already have the CSV, download it first with kagglehub (see the
"download_dataset" cell in colab_finetune.ipynb) or manually from the Kaggle
page above and place it next to this script.
"""

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd

random.seed(42)

# Columns that identify the "name" of a part, in likely order of appearance.
NAME_CANDIDATES = ["name", "Name", "product_name", "Product_Name", "title", "Title"]

# Columns we never turn into a question (identifiers / noise), matched case-insensitively.
SKIP_COLUMNS = {"id", "index", "unnamed: 0", "url", "link", "image", "image_url"}

# A few hand-written phrasings per attribute so the model sees varied question forms.
# Falls back to a generic template for any attribute not listed here.
TEMPLATES = {
    "price": [
        "How much does the {name} cost?",
        "What is the price of the {name}?",
    ],
    "type": [
        "What type of component is the {name}?",
        "What category does the {name} belong to?",
    ],
}
GENERIC_TEMPLATES = [
    "What is the {attr} of the {name}?",
    "Tell me the {attr} for the {name}.",
    "What {attr} does the {name} have?",
]


def humanize(col: str) -> str:
    """Turn a column name like 'clock_speed_ghz' into 'clock speed ghz'."""
    s = re.sub(r"[_\-]+", " ", col).strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()


def find_name_column(df: pd.DataFrame) -> str:
    for c in NAME_CANDIDATES:
        if c in df.columns:
            return c
    # fallback: first object/string column
    for c in df.columns:
        if df[c].dtype == object:
            return c
    raise ValueError("Could not find a name/title column in the dataset.")


def clean_value(v) -> str:
    # Defensive: if v is somehow array-like (e.g. a duplicate-column selection
    # returning a Series instead of a scalar), reduce it to a single value first.
    if isinstance(v, (pd.Series, pd.DataFrame)):
        v = v.iloc[0] if not v.empty else None
    if v is None or pd.isna(v):
        return ""
    s = str(v).strip()
    return s


def dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename duplicate column labels with numeric suffixes instead of dropping data."""
    counts: dict[str, int] = {}
    new_cols = []
    for col in df.columns:
        if col not in counts:
            counts[col] = 0
            new_cols.append(col)
        else:
            counts[col] += 1
            new_cols.append(f"{col}.{counts[col]}")
    df = df.copy()
    df.columns = new_cols
    return df


def build_qa_pairs(df: pd.DataFrame, category_col: str | None) -> list[dict]:
    name_col = find_name_column(df)
    qa_pairs = []

    for _, row in df.iterrows():
        name = clean_value(row[name_col])
        if not name:
            continue

        for col in df.columns:
            if col == name_col:
                continue
            if col.lower() in SKIP_COLUMNS:
                continue

            value = clean_value(row[col])
            if not value or value.lower() in {"nan", "none", "n/a", "-"}:
                continue

            attr = humanize(col)
            key = attr if attr in TEMPLATES else None
            templates = TEMPLATES.get(key, GENERIC_TEMPLATES)
            question = random.choice(templates).format(name=name, attr=attr)

            if attr == "price" or "price" in attr:
                answer = f"The {name} costs {value}."
            else:
                answer = f"The {attr} of the {name} is {value}."

            qa_pairs.append(
                {
                    "part_name": name,
                    "category": clean_value(row[category_col]) if category_col else "",
                    "question": question,
                    "answer": answer,
                }
            )

        # A couple of "overview" questions per part, if enough fields exist.
        fields = [c for c in df.columns if c not in (name_col,) and c.lower() not in SKIP_COLUMNS]
        sample_fields = random.sample(fields, k=min(3, len(fields)))
        if sample_fields:
            desc_parts = []
            for c in sample_fields:
                v = clean_value(row[c])
                if v and v.lower() not in {"nan", "none"}:
                    desc_parts.append(f"{humanize(c)} is {v}")
            if desc_parts:
                qa_pairs.append(
                    {
                        "part_name": name,
                        "category": clean_value(row[category_col]) if category_col else "",
                        "question": f"Can you give me an overview of the {name}?",
                        "answer": f"The {name} - {', '.join(desc_parts)}.",
                    }
                )

    return qa_pairs


def split_and_save(qa_pairs: list[dict], outdir: Path):
    random.shuffle(qa_pairs)
    n = len(qa_pairs)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train = qa_pairs[:n_train]
    val = qa_pairs[n_train : n_train + n_val]
    test = qa_pairs[n_train + n_val :]

    outdir.mkdir(parents=True, exist_ok=True)
    for split_name, split_data in (("train", train), ("validation", val), ("test", test)):
        path = outdir / f"{split_name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for row in split_data:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split_name}: {len(split_data)} examples -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the raw pc-parts CSV")
    parser.add_argument("--outdir", default="./qa_data", help="Where to write train/val/test jsonl")
    parser.add_argument(
        "--category-col",
        default=None,
        help="Optional column name that stores the part category (CPU/GPU/RAM/...) if present",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = dedupe_columns(df)
    print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    category_col = args.category_col
    if category_col is None:
        for guess in ("type", "Type", "category", "Category"):
            if guess in df.columns:
                category_col = guess
                break

    qa_pairs = build_qa_pairs(df, category_col)
    print(f"Generated {len(qa_pairs)} QA pairs")

    split_and_save(qa_pairs, Path(args.outdir))


if __name__ == "__main__":
    main()
