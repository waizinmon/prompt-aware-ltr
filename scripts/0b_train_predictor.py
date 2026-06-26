"""
PHASE 1, STEP 0b: Fine-tune OPT-125M Predictor (runs on Mac M4 -- MPS)
==========================================================================

Trains the LTR predictor on the dataset produced by 0a_build_dataset.py.
Supports BOTH loss functions so you can produce the original-paper
baseline AND your Extension #6 (pairwise margin loss) from the same
script -- just flip --loss_type.

This is genuinely trainable on a Mac M4. OPT-125M is small; with MPS
acceleration, 10 epochs over ~24k samples should take roughly 1-3
hours depending on batch size and sequence length, not days.

USAGE:
    # Original paper's setup (ListMLE, listwise)
    python 0b_train_predictor.py \
        --train_set ./data/train_predictor.jsonl \
        --output_dir ./checkpoints/opt125m-ltr-original \
        --loss_type listmle \
        --epochs 10

    # Extension #6 (pairwise margin ranking loss)
    python 0b_train_predictor.py \
        --train_set ./data/train_predictor.jsonl \
        --output_dir ./checkpoints/opt125m-ltr-marginloss \
        --loss_type margin \
        --margin 1.0 \
        --pair_filter_delta 0.20 \
        --epochs 10
"""

import argparse
import json
import os
import random
from itertools import combinations


def get_device():
    import torch
    if torch.cuda.is_available():
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon GPU acceleration)")
        return torch.device("mps")
    print("No GPU available (no CUDA, no MPS), falling back to CPU (will be slower)")
    return torch.device("cpu")


def load_train_records(path):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


import torch
import torch.nn as nn


class LTRPredictor(nn.Module):
    """
    OPT-125M + a linear head mapping hidden states to a single
    ranking score, matching the architecture described in the
    L2Rank paper and Hao AI Lab's blog post (OPT + appended MLP).

    Defined at module level (not nested inside a function) so it can
    be imported by scoring_patch_helper.py and 3_vllm_benchmark.py --
    this is a custom architecture, NOT a standard HF
    AutoModelForSequenceClassification, so it must be loaded with
    this exact class, not transformers' Auto* loaders.
    """
    def __init__(self, hidden_size):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained("facebook/opt-125m")
        self.score_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = out.last_hidden_state  # [B, L, H]
        # Use the last non-padded token's hidden state (EOS-style pooling)
        seq_lengths = attention_mask.sum(dim=1) - 1
        pooled = last_hidden[
            torch.arange(last_hidden.size(0)), seq_lengths
        ]
        score = self.score_head(pooled).squeeze(-1)  # [B]
        return score


def build_model(device):
    from transformers import AutoTokenizer, AutoConfig

    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone_config = AutoConfig.from_pretrained("facebook/opt-125m")
    hidden_size = backbone_config.hidden_size

    model = LTRPredictor(hidden_size).to(device)
    return model, tokenizer


def batch_iterator(records, batch_size, seed):
    random.seed(seed)
    indices = list(range(len(records)))
    random.shuffle(indices)
    for i in range(0, len(indices), batch_size):
        yield [records[j] for j in indices[i:i + batch_size]]


def encode_batch(batch, tokenizer, device, max_length=512):
    import torch
    prompts = [r["prompt"] for r in batch]
    enc = tokenizer(
        prompts, return_tensors="pt", truncation=True,
        max_length=max_length, padding=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    true_lengths = torch.tensor(
        [r["true_output_length"] for r in batch], dtype=torch.float, device=device
    )
    return enc, true_lengths


def listmle_loss(scores, true_lengths):
    """
    Listwise ranking loss matching the original paper's ListMLE setup.
    Permutes by descending true length, then maximizes the likelihood
    of that ordering under the predicted scores.
    """
    import torch
    # Order indices by descending true output length (ground truth ranking)
    _, order = torch.sort(true_lengths, descending=True)
    ordered_scores = scores[order]

    # ListMLE: -sum_i [ s_i - log(sum_{j>=i} exp(s_j)) ]
    cumsum_exp = torch.flip(
        torch.cumsum(torch.flip(torch.exp(ordered_scores), dims=[0]), dim=0),
        dims=[0],
    )
    log_cumsum = torch.log(cumsum_exp + 1e-10)
    loss = -(ordered_scores - log_cumsum).sum() / ordered_scores.size(0)
    return loss


def margin_ranking_loss(scores, true_lengths, margin, pair_filter_delta):
    """
    Extension #6: Pairwise margin ranking loss with noisy-pair filtering.

    Only trains on pairs whose true output lengths differ by at least
    `pair_filter_delta` (relative, e.g. 0.20 = 20%) -- discards
    near-equal pairs that add noise without useful ranking signal.
    """
    import torch
    n = scores.size(0)
    pair_losses = []

    for i, j in combinations(range(n), 2):
        len_i, len_j = true_lengths[i].item(), true_lengths[j].item()
        if max(len_i, len_j) == 0:
            continue
        rel_diff = abs(len_i - len_j) / max(len_i, len_j)
        if rel_diff < pair_filter_delta:
            continue  # skip noisy near-equal pair

        if len_i > len_j:
            target = 1.0   # score_i should be > score_j by `margin`
            s_a, s_b = scores[i], scores[j]
        else:
            target = 1.0
            s_a, s_b = scores[j], scores[i]

        pair_loss = torch.clamp(margin - (s_a - s_b), min=0.0)
        pair_losses.append(pair_loss)

    if not pair_losses:
        # All pairs in this batch were too similar -- return zero loss
        # rather than crashing; happens occasionally with small batches.
        return scores.sum() * 0.0

    return torch.stack(pair_losses).mean()


def train(args):
    import torch

    device = get_device()
    records = load_train_records(args.train_set)
    print(f"Loaded {len(records)} training records")

    model, tokenizer = build_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"\nTraining with loss_type={args.loss_type} for {args.epochs} epochs")
    print(f"Batch size: {args.batch_size}  |  Device: {device}\n")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in batch_iterator(records, args.batch_size, seed=args.seed + epoch):
            if len(batch) < 2:
                continue  # need at least 2 samples to rank

            enc, true_lengths = encode_batch(batch, tokenizer, device)
            scores = model(enc["input_ids"], enc["attention_mask"])

            if args.loss_type == "listmle":
                loss = listmle_loss(scores, true_lengths)
            elif args.loss_type == "margin":
                loss = margin_ranking_loss(
                    scores, true_lengths, args.margin, args.pair_filter_delta
                )
            else:
                raise ValueError(f"Unknown loss_type: {args.loss_type}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch + 1}/{args.epochs}  avg_loss={avg_loss:.4f}")

        # Original paper notes overfitting beyond 10 epochs -- warn if exceeded
        if epoch + 1 > 10:
            print("  [WARNING] Past 10 epochs: original paper reports "
                  "overfitting beyond this point. Monitor a held-out "
                  "validation set if continuing.")

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(args.output_dir)

    # Save the OPT-125M backbone's config so the checkpoint folder is
    # self-describing. NOTE: this is a custom architecture (OPT +
    # appended score_head), NOT a standard HF AutoModelForSequenceClassification.
    # Load it with LTRPredictor directly (see scoring_patch_helper.py),
    # not transformers' Auto* loaders -- the config.json here is for
    # reference/debugging, not for AutoModel.from_pretrained() to use.
    model.backbone.config.save_pretrained(args.output_dir)

    # Save a small config so 2_kendalls_tau_eval.py / 3_vllm_benchmark.py
    # know how this checkpoint was trained.
    with open(os.path.join(args.output_dir, "ltr_training_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nSaved checkpoint to {args.output_dir}")
    print("This is the folder path to use as --predictor_path in later scripts.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_set", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--loss_type", choices=["listmle", "margin"], default="listmle")
    parser.add_argument("--margin", type=float, default=1.0,
                        help="Margin for margin ranking loss (Extension #6 only)")
    parser.add_argument("--pair_filter_delta", type=float, default=0.20,
                        help="Min relative length difference to keep a pair "
                             "(Extension #6 only)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Smaller than the paper's batch 64 -- reasonable "
                             "default for Mac M4 memory; raise if you have headroom")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
