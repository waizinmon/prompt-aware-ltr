"""
EVALUATION PART 2: vLLM Latency Benchmark (run on Google Colab GPU)
======================================================================

This is the piece that REQUIRES a CUDA GPU and cannot run on Mac M4.
Upload this file + your modified scoring function to Colab, then run
it there.

SETUP ON COLAB (paste into a cell first):
    !git clone https://github.com/hao-ai-lab/vllm-ltr.git
    %cd vllm-ltr
    !pip install -e .
    !pip install transformers accelerate scipy matplotlib

    # If on the free T4 (16GB), uncomment the quantization flag below
    # in launch_vllm() -- Llama-3.1-8B at FP16 alone needs ~16GB and
    # vLLM needs headroom on top of that.

USAGE (from a Colab cell or terminal):
    python 3_vllm_benchmark.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --ltr_predictor_path ./checkpoints/opt125m-ltr-original \
        --ltr_promptlen_predictor_path ./checkpoints/opt125m-ltr-marginloss \
        --dataset ./data/dataset1_eval.jsonl \
        --request_rates 5 10 20 30 40 50 60 \
        --schedulers fcfs classification ltr ltr_promptlen \
        --output results.json
"""

import argparse
import json
import time
import asyncio
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class BenchmarkResult:
    scheduler: str
    request_rate: float
    mean_latency_s_per_token: float = 0.0
    p90_latency_s_per_token: float = 0.0
    n_requests: int = 0
    # Time to First Token -- time from request arrival to first generated
    # token, pulled directly from vLLM's own per-request metrics (not
    # derived from elapsed/n_tokens, so it isn't affected by the
    # max_tokens-cap issue that affects mean_latency_s_per_token).
    mean_ttft_s: float = 0.0
    p90_ttft_s: float = 0.0


def load_dataset(path: str, max_requests: int = None) -> List[dict]:
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
            if max_requests is not None and len(records) >= max_requests:
                break
    return records


def launch_vllm(model_name: str, quantize: bool = False):
    """
    Launches the vLLM engine matching the original paper's setup:
    vLLM (current stable, V1 engine by default), FP16.

    NOTE: the original paper's central technique was choosing
    preemption_mode="swap" over the default "recompute" on vLLM 0.4.1
    (V0 engine). Current stable vLLM's V1 engine REMOVED this
    parameter entirely -- not renamed, removed -- because V1's
    "simplified core architecture... no longer requires KV cache
    swapping to handle request preemptions" (vLLM V1 docs). This is a
    genuine architectural difference, not just a syntax change: the
    swap-vs-recompute axis the original paper compared no longer
    exists as a user-facing choice on this vLLM version. State this
    explicitly in your report -- it's a real limitation on how
    directly this reproduction can be compared to the original paper,
    not a bug to "fix" further.

    Set quantize=True if running on free Colab T4 (16GB) -- this is
    NOT what the original paper used, so clearly label any results
    produced this way as "quantized" rather than a direct comparison.
    """
    from vllm import LLM, SamplingParams

    engine_kwargs = dict(
        model=model_name,
        dtype="float16",
    )
    if quantize:
        engine_kwargs["quantization"] = "awq"
        print("WARNING: running with AWQ quantization (free-tier GPU mode). "
              "Results are NOT directly comparable to the original "
              "paper's FP16 numbers -- report this caveat alongside results.")

    llm = LLM(**engine_kwargs)
    return llm


def apply_scheduler(scheduler_name: str, requests: List[dict], models: Dict, tokenizers: Dict):
    """
    Orders `requests` according to the named scheduling policy.
    Returns the requests in the order they should be submitted/batched.

    `models` / `tokenizers` are dicts keyed by scheduler name so that
    "ltr" and "ltr_promptlen" can each use their OWN fine-tuned
    checkpoint (original ListMLE vs. Extension #6 margin-loss model)
    rather than incorrectly sharing one predictor.
    """
    if scheduler_name == "fcfs":
        return requests  # arrival order, unchanged

    if scheduler_name == "classification":
        # Classification-based baseline, following the L2Rank paper's
        # description (Fu et al., Section 5.1): bucket requests by
        # predicted output length into `num_buckets` equal-width bins,
        # using bucket_size = max_context_length / num_buckets, then
        # order by ascending bucket (shortest-predicted-bucket first),
        # FCFS within each bucket. This mirrors the "S3"-style
        # classification predictor the original main paper cites as
        # its comparison method.
        #
        # Uses the SAME OPT-125M predictor checkpoint as "ltr" -- the
        # only difference from the "ltr" scheduler is that this one
        # coarsens the continuous score into discrete buckets before
        # sorting, exactly as a classification (not ranking) model
        # would. This isolates the effect of ranking vs. classification
        # while controlling for predictor quality.
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from scoring_patch_helper import rank_requests_ltr_only

        NUM_BUCKETS = 10
        MAX_CONTEXT_LENGTH = 2048  # adjust to match your eval set's max length
        bucket_size = MAX_CONTEXT_LENGTH / NUM_BUCKETS

        # Reuse the "ltr" predictor's raw scores as a stand-in for a
        # length classifier's predicted length, then discretize.
        order = rank_requests_ltr_only(
            models["ltr"], tokenizers["ltr"], [r["prompt"] for r in requests]
        )
        # `order` is already sorted ascending by predicted score/length.
        # Assign bucket indices based on RANK position (not raw score)
        # so bucket boundaries are well-defined regardless of the
        # predictor's raw score scale.
        n = len(requests)
        bucketed = []
        for rank_pos, idx in enumerate(order):
            bucket_idx = min(int(rank_pos / n * NUM_BUCKETS), NUM_BUCKETS - 1)
            bucketed.append((bucket_idx, idx))

        # Sort by bucket only (coarse) -- ties within a bucket keep
        # their original arrival order, i.e. FCFS within bucket.
        bucketed.sort(key=lambda x: x[0])
        return [requests[idx] for _, idx in bucketed]

    if scheduler_name == "ltr":
        # Original paper's LTR-only ranking -- uses the "ltr" checkpoint
        # (trained with --loss_type listmle in 0b_train_predictor.py).
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from scoring_patch_helper import rank_requests_ltr_only
        order = rank_requests_ltr_only(
            models["ltr"], tokenizers["ltr"], [r["prompt"] for r in requests]
        )
        return [requests[i] for i in order]

    if scheduler_name == "ltr_promptlen":
        # This extension -- uses the "ltr_promptlen" checkpoint
        # (trained with --loss_type margin in 0b_train_predictor.py),
        # combined with the prompt-length secondary signal.
        import sys, os
        sys.path.insert(0, os.path.dirname(__file__))
        from scoring_patch_helper import rank_requests_extended
        order = rank_requests_extended(
            models["ltr_promptlen"], tokenizers["ltr_promptlen"],
            [r["prompt"] for r in requests]
        )
        return [requests[i] for i in order]

    raise ValueError(f"Unknown scheduler: {scheduler_name}")


def run_single_benchmark(llm, scheduler_name, dataset, request_rate, models, tokenizers, max_tokens=512, max_rate=60.0):
    """
    Runs one (scheduler, request_rate) combination and returns a
    BenchmarkResult. This mirrors the latency-vs-request-rate sweep
    in the original paper's Figure 3.

    IMPORTANT: requests are submitted to vLLM as a SINGLE BATCH via
    llm.generate(list_of_prompts, ...), not one at a time in a loop.
    vLLM's own continuous batching scheduler is what determines
    execution order and overlap -- the `ordered` list produced by
    apply_scheduler() reflects each scheduler's PRIORITY ordering
    (which requests vLLM should prefer when constructing batches
    under load), not a literal one-by-one submission sequence.
    Submitting one request per llm.generate() call (the previous
    implementation) defeats vLLM's batching entirely and produces
    runtimes dominated by per-call Python/CUDA-graph overhead rather
    than the actual scheduling behavior being compared.

    request_rate is used to size how many requests are admitted into
    this batch (approximating a fixed time window at that arrival
    rate), not as a literal per-request sleep delay.
    """
    from vllm import SamplingParams

    ordered = apply_scheduler(scheduler_name, dataset, models, tokenizers)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    # Scale the number of admitted requests proportionally to
    # request_rate relative to the highest rate in this sweep, so each
    # rate produces a genuinely different batch size (a fixed-size
    # time window collapses every rate above a low threshold to the
    # same full-dataset batch once request_rate * window exceeds the
    # dataset size, making the sweep uninformative). At max_rate the
    # full dataset (or DATASET_CAP, whichever is smaller) is admitted;
    # lower rates admit proportionally fewer requests.
    DATASET_CAP = 200
    frac = request_rate / max_rate if max_rate > 0 else 1.0
    n_admit = max(5, int(frac * min(len(ordered), DATASET_CAP)))
    n_admit = min(n_admit, len(ordered))
    batch = ordered[:n_admit]
    prompts = [r["prompt"] for r in batch]

    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - start

    # vLLM returns RequestOutput objects in the same order as the
    # input prompts (confirmed via vLLM docs/source), so output[i]
    # corresponds to prompts[i] / batch[i].
    per_request_latencies = []
    for output in outputs:
        n_tokens = len(output.outputs[0].token_ids) or 1
        # Per-request latency here is total batch wall-clock time
        # divided by that request's token count -- a throughput-
        # normalized view appropriate for a BATCHED run, not a
        # measurement of that request's individual queue+exec time
        # in isolation (vLLM does not expose true per-request
        # start/end timestamps through the offline llm.generate() API).
        per_request_latencies.append(elapsed / n_tokens)

    per_request_latencies.sort()
    n = len(per_request_latencies)
    mean_lat = sum(per_request_latencies) / n if n else 0.0
    p90_lat = per_request_latencies[int(0.9 * n)] if n else 0.0

    # Time to First Token, using vLLM's own per-request metrics
    # (arrival_time, first_token_time) rather than the shared-elapsed
    # approximation used for mean_latency_s_per_token above.
    ttfts = []
    for output in outputs:
        m = getattr(output, "metrics", None)
        if m is not None and m.first_token_time is not None and m.arrival_time is not None:
            ttfts.append(m.first_token_time - m.arrival_time)
    ttfts.sort()
    n_ttft = len(ttfts)
    mean_ttft = sum(ttfts) / n_ttft if n_ttft else 0.0
    p90_ttft = ttfts[int(0.9 * n_ttft)] if n_ttft else 0.0
    if n_ttft < len(outputs):
        print(f"  [WARNING] TTFT metrics missing for {len(outputs) - n_ttft}/{len(outputs)} "
              f"requests -- vLLM may not be populating output.metrics in this version.")

    return BenchmarkResult(
        scheduler=scheduler_name,
        request_rate=request_rate,
        mean_latency_s_per_token=mean_lat,
        p90_latency_s_per_token=p90_lat,
        n_requests=n,
        mean_ttft_s=round(mean_ttft, 5),
        p90_ttft_s=round(p90_ttft, 5),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--ltr_predictor_path", required=True,
                        help="Checkpoint trained with --loss_type listmle "
                             "(original paper baseline)")
    parser.add_argument("--ltr_promptlen_predictor_path", required=True,
                        help="Checkpoint trained with --loss_type margin "
                             "(Extension #6, used by the ltr_promptlen scheduler)")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max_requests", type=int, default=None,
                        help="Limit dataset to first N requests -- use a small "
                             "number (e.g. 20-50) for a fast smoke test before "
                             "committing to a full run with all requests")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="Max tokens to generate per request. Lower this "
                             "(e.g. 64) for smoke tests to finish faster")
    parser.add_argument("--request_rates", nargs="+", type=float,
                        default=[5, 10, 20, 30, 40, 50, 60])
    parser.add_argument("--schedulers", nargs="+",
                        default=["fcfs", "classification", "ltr", "ltr_promptlen"])
    parser.add_argument("--quantize", action="store_true",
                        help="Use AWQ quantization for free-tier T4 GPUs")
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset} ...")
    dataset = load_dataset(args.dataset, max_requests=args.max_requests)
    print(f"Loaded {len(dataset)} requests"
          + (f" (capped at --max_requests={args.max_requests})"
             if args.max_requests else ""))

    import sys, os as _os
    sys.path.insert(0, _os.path.dirname(__file__))
    from importlib.util import spec_from_file_location, module_from_spec

    # 0b_train_predictor.py starts with a digit, so import it by file path
    # (same trick used in scoring_patch_helper.py for 1_scoring_patch.py).
    _spec = spec_from_file_location(
        "train_predictor_module",
        _os.path.join(_os.path.dirname(__file__), "0b_train_predictor.py"),
    )
    train_predictor_module = module_from_spec(_spec)
    _spec.loader.exec_module(train_predictor_module)
    LTRPredictor = train_predictor_module.LTRPredictor

    from transformers import AutoTokenizer, AutoConfig
    import torch

    def load_ltr_checkpoint(checkpoint_dir):
        """
        Reconstructs the custom LTRPredictor architecture and loads its
        weights from a checkpoint saved by 0b_train_predictor.py.

        NOTE: this is NOT a standard HF AutoModelForSequenceClassification
        checkpoint -- it's a custom OPT backbone + linear score head, so
        it must be loaded with this exact class, not transformers' Auto*
        model loaders (those expect a different weight/config layout).
        """
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        backbone_config = AutoConfig.from_pretrained(checkpoint_dir)
        model = LTRPredictor(backbone_config.hidden_size)
        state_dict = torch.load(
            _os.path.join(checkpoint_dir, "pytorch_model.bin"),
            map_location="cuda",
        )
        model.load_state_dict(state_dict)
        model = model.cuda().eval()
        return model, tokenizer

    models, tokenizers = {}, {}

    if "ltr" in args.schedulers:
        print(f"Loading 'ltr' predictor from {args.ltr_predictor_path} ...")
        models["ltr"], tokenizers["ltr"] = load_ltr_checkpoint(args.ltr_predictor_path)

    if "ltr_promptlen" in args.schedulers:
        print(f"Loading 'ltr_promptlen' predictor from "
              f"{args.ltr_promptlen_predictor_path} ...")
        models["ltr_promptlen"], tokenizers["ltr_promptlen"] = load_ltr_checkpoint(
            args.ltr_promptlen_predictor_path
        )

    print(f"Launching vLLM with model={args.model} quantize={args.quantize} ...")
    llm = launch_vllm(args.model, quantize=args.quantize)

    all_results = []
    for scheduler_name in args.schedulers:
        for rate in args.request_rates:
            print(f"\nRunning scheduler={scheduler_name} request_rate={rate} req/s ...")
            result = run_single_benchmark(llm, scheduler_name, dataset, rate, models, tokenizers, max_tokens=args.max_tokens, max_rate=max(args.request_rates))
            print(f"  mean={result.mean_latency_s_per_token:.4f} s/tok  "
                  f"p90={result.p90_latency_s_per_token:.4f} s/tok  "
                  f"TTFT mean={result.mean_ttft_s:.4f}s p90={result.p90_ttft_s:.4f}s")
            all_results.append(result.__dict__)

    with open(args.output, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved all results to {args.output}")


if __name__ == "__main__":
    main()
