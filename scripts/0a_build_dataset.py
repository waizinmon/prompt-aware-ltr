"""
PHASE 1, STEP 0a: Build Dataset 1 / Dataset 2 (runs on Mac M4 -- no GPU)
==========================================================================

Reconstructs an approximation of the main paper's train/eval split,
using the real public LMSYS-Chat-1M dataset (their cited source).

IMPORTANT HONESTY NOTE:
    This is NOT the exact 23,800-sample dataset the main paper used --
    that exact file was never published (their synthesize_dataset.py
    script is not public). This script builds a comparable dataset
    from the same public source (LMSYS-Chat-1M) with similar
    statistics (paper reports: mean input 192 tokens, mean output
    157 tokens). Report results as "evaluated on a reconstructed
    LMSYS-Chat-1M split" rather than claiming exact reproduction.

REQUIRES:
    - A HuggingFace account + accepting the LMSYS-Chat-1M usage terms:
      https://huggingface.co/datasets/lmsys/lmsys-chat-1m
    - pip install datasets transformers huggingface_hub --break-system-packages

OUTPUT:
    ./data/train_predictor.jsonl   (for fine-tuning OPT-125M, ~10-24k samples)
    ./data/dataset1_eval.jsonl     (in-distribution test set, held out)
    ./data/dataset2_eval.jsonl     (out-of-distribution test set, held out,
                                     drawn from a DIFFERENT conversation
                                     topic/length regime than training)

USAGE:
    python 0a_build_dataset.py \
        --hf_token YOUR_TOKEN \
        --train_size 23800 \
        --eval_size 1000 \
        --output_dir ./data
"""

import argparse
import json
import os
import random


def load_lmsys_chat_1m(hf_token: str, max_records: int = 200_000):
    """
    Loads LMSYS-Chat-1M from HuggingFace. Requires accepting the dataset's
    usage terms on the website first (one-time, instant approval usually).
    """
    from datasets import load_dataset

    print("Loading LMSYS-Chat-1M (streaming, this may take a few minutes)...")
    ds = load_dataset(
        "lmsys/lmsys-chat-1m",
        split="train",
        token=hf_token,
        streaming=True,
    )

    records = []
    for i, row in enumerate(ds):
        if i >= max_records:
            break
        conv = row.get("conversation", [])
        if not conv:
            continue
        # First human turn = the "prompt" for our scheduling task
        user_turns = [t["content"] for t in conv if t.get("role") == "user"]
        assistant_turns = [t["content"] for t in conv if t.get("role") == "assistant"]
        if not user_turns or not assistant_turns:
            continue
        records.append({
            "prompt": user_turns[0],
            "response": assistant_turns[0],
        })

    print(f"Loaded {len(records)} usable (prompt, response) pairs")
    return records


def add_token_lengths(records, tokenizer):
    """
    Appends true_output_length (in tokens) to each record using a
    tokenizer -- this becomes the ground-truth label for training
    and for the Kendall's tau evaluation in step 2.
    """
    out = []
    for r in records:
        prompt_tokens = len(tokenizer(r["prompt"])["input_ids"])
        output_tokens = len(tokenizer(r["response"])["input_ids"])
        out.append({
            "prompt": r["prompt"],
            "true_output_length": output_tokens,
            "prompt_length": prompt_tokens,
        })
    return out


def make_splits(records, train_size: int, eval_size: int, seed: int = 42):
    """
    Mirrors the paper's description: two NON-OVERLAPPING subsets.

    Dataset 1 (in-distribution): randomly sampled from the same pool
        as training data, but held out (never seen during training).
    Dataset 2 (out-of-distribution): sampled from a length-shifted
        subset to approximate genuine distribution shift, rather than
        just a random re-shuffle of the same distribution. The paper
        describes Dataset 2 as "a different subset" that exposes
        overfitting -- a pure random split usually isn't different
        enough to reproduce that effect, so this script biases
        Dataset 2 toward longer/shorter outputs than the training mean
        to create a genuine shift.
    """
    random.seed(seed)
    random.shuffle(records)

    if len(records) < train_size + 2 * eval_size:
        raise ValueError(
            f"Not enough records ({len(records)}) for requested "
            f"train_size={train_size} + 2*eval_size={eval_size}. "
            f"Lower --train_size/--eval_size or increase --max_records."
        )

    train = records[:train_size]
    remaining = records[train_size:]

    # Dataset 1: in-distribution -- plain random sample of remaining pool
    dataset1 = remaining[:eval_size]

    # Dataset 2: out-of-distribution -- bias toward the tails of the
    # output-length distribution to simulate genuine distribution shift,
    # matching the paper's reported finding that Dataset 2 causes
    # performance to collapse.
    remaining_sorted = sorted(
        remaining[eval_size:], key=lambda r: r["true_output_length"]
    )
    half = eval_size // 2
    dataset2 = remaining_sorted[:half] + remaining_sorted[-half:]
    random.shuffle(dataset2)

    return train, dataset1, dataset2


def save_jsonl(records, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {len(records)} records to {path}")


def print_stats(name, records):
    lengths_in = [r["prompt_length"] for r in records]
    lengths_out = [r["true_output_length"] for r in records]
    mean_in = sum(lengths_in) / len(lengths_in)
    mean_out = sum(lengths_out) / len(lengths_out)
    print(f"{name}: n={len(records)}  mean_input={mean_in:.1f} tok  "
          f"mean_output={mean_out:.1f} tok")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", required=True,
                        help="HuggingFace token with read access "
                             "(needed to accept LMSYS-Chat-1M usage terms)")
    parser.add_argument("--train_size", type=int, default=23800,
                        help="Matches the main paper's expanded dataset size")
    parser.add_argument("--eval_size", type=int, default=1000,
                        help="Size of EACH eval set (Dataset 1 and Dataset 2)")
    parser.add_argument("--max_records", type=int, default=200_000,
                        help="How many raw LMSYS records to pull before splitting")
    parser.add_argument("--output_dir", default="./data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    print("Loading tokenizer (facebook/opt-125m, for consistent token counting)...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

    raw_records = load_lmsys_chat_1m(args.hf_token, max_records=args.max_records)
    records = add_token_lengths(raw_records, tokenizer)

    train, dataset1, dataset2 = make_splits(
        records, args.train_size, args.eval_size, seed=args.seed
    )

    print("\n--- Dataset statistics ---")
    print_stats("Train", train)
    print_stats("Dataset 1 (in-distribution)", dataset1)
    print_stats("Dataset 2 (out-of-distribution)", dataset2)
    print("\nMain paper reports: mean input 192 tok, mean output 157 tok "
          "(your numbers above won't match exactly -- different "
          "sample, same source dataset).")

    save_jsonl(train, os.path.join(args.output_dir, "train_predictor.jsonl"))
    save_jsonl(dataset1, os.path.join(args.output_dir, "dataset1_eval.jsonl"))
    save_jsonl(dataset2, os.path.join(args.output_dir, "dataset2_eval.jsonl"))

    print("\nDone. These three files are what 0b_train_predictor.py and "
          "3_vllm_benchmark.py expect under --dataset / --train_set.")


if __name__ == "__main__":
    main()
