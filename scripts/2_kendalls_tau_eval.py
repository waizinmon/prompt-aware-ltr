"""
EVALUATION PART 1: Kendall's Tau (runs on Mac M4 -- no GPU required)
======================================================================

This script measures ranking quality WITHOUT needing vLLM or a GPU.
It only needs your trained OPT-125M predictor (CPU or MPS) and a
held-out test set of (prompt, true_output_length) pairs.

This is the piece of the evaluation you can complete entirely on a
Mac M4, per the plan discussed: train + measure tau locally, then
only use Colab for the final vLLM latency benchmark.

USAGE:
    python 2_kendalls_tau_eval.py \
        --predictor_path ./checkpoints/opt125m-ltr \
        --test_set ./data/dataset2_test.jsonl \
        --alpha 0.7 --beta 0.3

Expected test_set format (jsonl), one record per line:
    {"prompt": "...", "true_output_length": 143}
"""

import argparse
import json
from scipy.stats import kendalltau


def load_test_set(path: str):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_predictor(predictor_path: str):
    """
    Loads the fine-tuned OPT-125M predictor + tokenizer.
    Uses CUDA if available (Colab), then MPS (Mac M4), then CPU.

    NOTE: this loads the custom LTRPredictor architecture defined in
    0b_train_predictor.py (OPT backbone + linear score head), NOT a
    standard HF AutoModelForSequenceClassification -- the checkpoint
    saved by 0b_train_predictor.py is not in that format.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from importlib.util import spec_from_file_location, module_from_spec

    _spec = spec_from_file_location(
        "train_predictor_module",
        os.path.join(os.path.dirname(__file__), "0b_train_predictor.py"),
    )
    train_predictor_module = module_from_spec(_spec)
    _spec.loader.exec_module(train_predictor_module)
    LTRPredictor = train_predictor_module.LTRPredictor

    import torch
    from transformers import AutoTokenizer, AutoConfig

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Loading predictor on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(predictor_path)
    backbone_config = AutoConfig.from_pretrained(predictor_path)
    model = LTRPredictor(backbone_config.hidden_size)
    state_dict = torch.load(
        os.path.join(predictor_path, "pytorch_model.bin"), map_location=device
    )
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, tokenizer, device


def predict_score(model, tokenizer, device, prompt: str) -> float:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        score = model(inputs["input_ids"], inputs["attention_mask"])
    return float(score.squeeze().item())


def compute_combined_score(ltr_score: float, prompt_len: int, alpha: float, beta: float) -> float:
    prompt_len = max(prompt_len, 1)
    length_term = (1.0 / prompt_len) * 100.0  # same SCALE as in 1_scoring_patch.py
    # FIX: subtract instead of add -- same sign bug as in 1_scoring_patch.py's
    return alpha * ltr_score - beta * length_term


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor_path", required=True,
                        help="Path or HF hub id of your fine-tuned OPT-125M predictor")
    parser.add_argument("--test_set", required=True,
                        help="Path to jsonl test set: {prompt, true_output_length}")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--output", default="tau_results.json",
                        help="Path to write the tau results JSON to (e.g. result/tau_results.json)")
    args = parser.parse_args()

    records = load_test_set(args.test_set)
    print(f"Loaded {len(records)} test records from {args.test_set}")

    model, tokenizer, device = load_predictor(args.predictor_path)

    true_lengths = []
    ltr_only_scores = []
    combined_scores = []

    for r in records:
        prompt = r["prompt"]
        true_len = r["true_output_length"]

        ltr_score = predict_score(model, tokenizer, device, prompt)
        prompt_len = len(tokenizer(prompt)["input_ids"])
        combined = compute_combined_score(ltr_score, prompt_len, args.alpha, args.beta)

        true_lengths.append(true_len)
        ltr_only_scores.append(ltr_score)
        combined_scores.append(combined)

    tau_original, p_original = kendalltau(true_lengths, ltr_only_scores)
    tau_extended, p_extended = kendalltau(true_lengths, combined_scores)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Original LTR only        Kendall's tau = {tau_original:.4f}  (p={p_original:.4g})")
    print(f"LTR + Prompt Length       Kendall's tau = {tau_extended:.4f}  (p={p_extended:.4g})")
    print(f"Delta                     {tau_extended - tau_original:+.4f}")
    print("=" * 60)

    out_path = args.output
    with open(out_path, "w") as f:
        json.dump({
            "tau_original": tau_original,
            "tau_extended": tau_extended,
            "delta": tau_extended - tau_original,
            "alpha": args.alpha,
            "beta": args.beta,
            "n_test_records": len(records),
        }, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
