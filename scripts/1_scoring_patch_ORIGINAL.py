"""
EXTENSION: Add Prompt Length as a Secondary Ranking Signal
============================================================

WHERE THIS GOES:
    In the vllm-ltr repo (https://github.com/hao-ai-lab/vllm-ltr),
    this logic lives wherever the LTR predictor score is turned into
    a scheduling priority -- typically inside the custom scheduler
    policy file under vllm/core/scheduler.py (or wherever the repo's
    LTR hook computes `request.priority` / `request.score`).

    Search the repo for the function that calls the OPT-125M predictor
    and assigns a score to each waiting request. That function is what
    you are patching below.

WHY:
    - SARATHI (Agrawal et al.) shows prefill cost scales with prompt
      length -> long prompts cause more Head-Of-Line blocking.
    - L2Rank (Fu et al., Appendix G) confirms prompt length is known
      BEFORE scheduling, with zero prediction cost.

WHAT CHANGES:
    Nothing about training. Nothing about the model. Only the function
    that converts a predictor score into a final scheduling priority.
"""

from typing import List


def predict_with_model(ltr_model, tokenizer, prompt: str) -> float:
    """
    Bridges a raw LTRPredictor (defined in 0b_train_predictor.py, which
    only implements forward(input_ids, attention_mask) -- a tensor-in,
    tensor-out interface) with the string-in/score-out interface the
    rest of this file expects.

    If ltr_model exposes a .predict(prompt) method (e.g. a dummy/mock
    model used for standalone testing, see __main__ below), that is
    used directly. Otherwise this does the tokenize -> forward pass ->
    extract-scalar-score steps for a real torch LTRPredictor checkpoint.
    """
    if hasattr(ltr_model, "predict"):
        return ltr_model.predict(prompt)

    import torch

    device = next(ltr_model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    enc = {k: v.to(device) for k, v in enc.items()}

    ltr_model.eval()
    with torch.no_grad():
        score = ltr_model(enc["input_ids"], enc["attention_mask"])

    return float(score.squeeze().item())


# ---------------------------------------------------------------------------
# BEFORE (original paper's behavior)
# ---------------------------------------------------------------------------
def compute_score_original(ltr_model, tokenizer, prompt: str) -> float:
    """
    Original scoring function used in the main paper.
    Lower score = higher scheduling priority (shorter predicted output first).
    """
    ltr_score = predict_with_model(ltr_model, tokenizer, prompt)
    return ltr_score


# ---------------------------------------------------------------------------
# AFTER (this extension)
# ---------------------------------------------------------------------------
def compute_score_with_prompt_length(
    ltr_model,
    tokenizer,
    prompt: str,
    alpha: float = 0.7,
    beta: float = 0.3,
) -> float:
    """
    Extended scoring function: blends the existing LTR predicted score
    with a free, always-accurate prompt-length signal.

    alpha : weight on the learned LTR score (output-length prediction)
    beta  : weight on the prompt-length term (input-length signal)

    alpha + beta should sum to 1.0 so the combined score stays in a
    comparable range to the original ltr_score for any downstream
    sorting / threshold logic that assumes that range.
    """
    ltr_score = predict_with_model(ltr_model, tokenizer, prompt)

    # Prompt length is known exactly -- zero prediction cost.
    prompt_len = len(tokenizer(prompt)["input_ids"])
    prompt_len = max(prompt_len, 1)  # guard against div-by-zero

    # Inverse length: shorter prompts -> larger term -> higher priority
    # when combined with a "lower score = scheduled sooner" convention.
    length_term = 1.0 / prompt_len

    # Normalize length_term roughly into the same scale as ltr_score.
    # NOTE: tune this scaling constant empirically once you have real
    # ltr_score distributions from your trained OPT-125M predictor.
    SCALE = 100.0
    score = alpha * ltr_score + beta * (length_term * SCALE)
    return score


# ---------------------------------------------------------------------------
# Drop-in replacement for a batch of waiting requests
# ---------------------------------------------------------------------------
def rank_requests(
    ltr_model,
    tokenizer,
    requests: List[str],
    use_prompt_length: bool = True,
    alpha: float = 0.7,
    beta: float = 0.3,
) -> List[int]:
    """
    Returns indices of `requests` sorted by scheduling priority
    (index of the request that should run first comes first).

    Set use_prompt_length=False to reproduce the original paper's
    behavior exactly -- useful for the FCFS/LTR-only baseline runs
    in your evaluation.
    """
    if use_prompt_length:
        scores = [
            compute_score_with_prompt_length(ltr_model, tokenizer, p, alpha, beta)
            for p in requests
        ]
    else:
        scores = [compute_score_original(ltr_model, tokenizer, p) for p in requests]

    # Lower score = scheduled sooner (shorter predicted output / prompt)
    ranked_indices = sorted(range(len(requests)), key=lambda i: scores[i])
    return ranked_indices


if __name__ == "__main__":
    # Minimal smoke test you can run on Mac M4 with no GPU.
    class DummyLTRModel:
        """Stand-in for the fine-tuned OPT-125M predictor."""
        def predict(self, prompt: str) -> float:
            # Pretend the model is slightly wrong about length ordering
            # for short prompts, on purpose, to show the effect of beta.
            return len(prompt) * 0.05

    class DummyTokenizer:
        def __call__(self, text):
            return {"input_ids": text.split()}

    model = DummyLTRModel()
    tok = DummyTokenizer()

    requests = [
        "Paris is",
        "How are you doing today, can you explain in detail",
        "1+1=",
        "Write a long essay about the history of Rome and its empire",
    ]

    print("Original LTR-only ranking:")
    for i in rank_requests(model, tok, requests, use_prompt_length=False):
        print(f"  -> {requests[i]!r}")

    print("\nLTR + Prompt Length ranking:")
    for i in rank_requests(model, tok, requests, use_prompt_length=True):
        print(f"  -> {requests[i]!r}")
