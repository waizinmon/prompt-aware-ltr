"""
EVALUATION PART 4: Head-of-Line (HOL) Blocking Analysis
==========================================================================

Estimates Head-of-Line blocking reduction across scheduling policies --
this directly tests the project's core hypothesis (Section 4.3 of the
proposal): that incorporating prompt length as a secondary ranking
signal reduces HOL blocking caused by long-prompt requests, relative to
the output-length-only LTR baseline.

WHY THIS METRIC MATTERS:
    The project's other four metrics (Average Latency, Throughput,
    Kendall's Tau, GPU Utilization) are useful proxies, but none of them
    directly measure whether HOL blocking specifically improved. This
    script fills that gap using two indicators computable entirely from
    the existing benchmark result files -- no additional GPU work
    required:

      1. p90/mean latency ratio at high load -- a high ratio means a
         small fraction of requests are experiencing disproportionately
         long waits (a signature of HOL blocking: a few long-prompt
         requests delaying many short ones). Lower ratio = less HOL
         blocking.

      2. Latency-vs-request-rate slope -- how steeply mean latency
         grows as request rate increases. A scheduler that controls HOL
         blocking should show a FLATTER slope (latency degrades more
         gracefully under load). Steeper slope = blocking is compounding
         as load increases.

LIMITATION:
    vLLM's offline llm.generate() API does not expose true per-request
    queue-entry or prefill-completion timestamps, so this is an
    estimation based on aggregate latency statistics rather than a
    direct trace of blocking events. State this explicitly in your
    report alongside any figures produced here.

USAGE:
    python scripts/6_hol_blocking_eval.py \
        --results_dir ./result \
        --data_dir ./data \
        --output hol_blocking_metrics.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import List, Dict


def load_results(results_dir: str) -> Dict[str, List[dict]]:
    """Load both dataset result files from the result/ directory."""
    loaded = {}
    for fname in ["results_dataset1.json", "results_dataset2.json"]:
        path = os.path.join(results_dir, fname)
        if os.path.exists(path):
            with open(path) as f:
                loaded[fname] = json.load(f)
            print(f"Loaded {len(loaded[fname])} records from {fname}")
        else:
            print(f"[WARNING] {fname} not found at {path} -- skipping")
    return loaded


def load_dataset(path: str) -> List[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} dataset records from {path}")
    return records


def compute_hol_blocking(
    results_for_scheduler: List[dict],
    dataset: List[dict],
    long_prompt_percentile: float = 0.75,
) -> dict:
    """
    Estimates Head-of-Line (HOL) blocking by comparing latency
    distribution shape across request rates for a given scheduler.

    Args:
        results_for_scheduler: list of result dicts for ONE scheduler
        dataset: the evaluation JSONL records (used to report long-prompt
            fraction for context, not directly in the HOL calculation)
        long_prompt_percentile: requests above this prompt-length
            percentile are classified as "long prompt" (default: top 25%)

    Returns:
        dict with HOL blocking indicators
    """
    prompt_lengths = sorted(
        r.get("prompt_length", len(r["prompt"].split())) for r in dataset
    )
    threshold = prompt_lengths[int(long_prompt_percentile * len(prompt_lengths))]
    long_prompt_fraction = sum(
        1 for r in dataset
        if r.get("prompt_length", len(r["prompt"].split())) >= threshold
    ) / len(dataset)

    # p90/mean ratio across request rates -- higher = more tail latency = more HOL
    ratios = []
    high_load_results = [r for r in results_for_scheduler if r.get("request_rate", 0) >= 30]
    for r in high_load_results:
        mean = r.get("mean_latency_s_per_token", 0)
        p90 = r.get("p90_latency_s_per_token", 0)
        if mean > 0:
            ratios.append(p90 / mean)

    mean_tail_ratio = sum(ratios) / len(ratios) if ratios else 0.0

    # Latency slope: how steeply latency grows with request rate
    sorted_results = sorted(results_for_scheduler, key=lambda x: x.get("request_rate", 0))
    if len(sorted_results) >= 2:
        low = sorted_results[0]
        high = sorted_results[-1]
        rate_delta = high.get("request_rate", 1) - low.get("request_rate", 1)
        lat_delta = (high.get("mean_latency_s_per_token", 0) -
                     low.get("mean_latency_s_per_token", 0))
        latency_slope = lat_delta / rate_delta if rate_delta > 0 else 0.0
    else:
        latency_slope = 0.0

    return {
        "long_prompt_threshold_tokens": threshold,
        "long_prompt_fraction_in_dataset": round(long_prompt_fraction, 4),
        "mean_p90_to_mean_ratio_high_load": round(mean_tail_ratio, 4),
        "latency_slope_per_req_rate": round(latency_slope, 6),
        "note": (
            "Lower p90/mean ratio and flatter latency slope indicate "
            "less HOL blocking. Compare across schedulers at the same "
            "request rates for a fair comparison."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="HOL blocking evaluation")
    parser.add_argument("--results_dir", default="./result",
                        help="Directory containing results_dataset1.json and results_dataset2.json")
    parser.add_argument("--data_dir", default="./data",
                        help="Directory containing dataset1_eval.jsonl and dataset2_eval.jsonl")
    parser.add_argument("--output", default="hol_blocking_metrics.json",
                        help="Output file for HOL blocking metrics")
    args = parser.parse_args()

    print("\n=== HOL Blocking Analysis ===\n")

    dataset_paths = {
        "dataset1_eval": os.path.join(args.data_dir, "dataset1_eval.jsonl"),
        "dataset2_eval": os.path.join(args.data_dir, "dataset2_eval.jsonl"),
    }
    datasets = {}
    for label, path in dataset_paths.items():
        if os.path.exists(path):
            datasets[label] = load_dataset(path)
        else:
            print(f"[WARNING] {path} not found -- skipping {label}")

    if not datasets:
        print("ERROR: Neither dataset1_eval.jsonl nor dataset2_eval.jsonl found "
              f"in {args.data_dir}")
        sys.exit(1)

    all_results = load_results(args.results_dir)
    if not all_results:
        print("ERROR: No result files found. Run 3_vllm_benchmark.py first.")
        sys.exit(1)

    output = {}

    for fname, results in all_results.items():
        dataset_label = fname.replace(".json", "").replace("results_", "") + "_eval"
        print(f"\n--- Processing {dataset_label} ---")

        dataset = datasets.get(dataset_label)
        if dataset is None:
            print(f"[WARNING] No matching dataset loaded for {dataset_label} -- skipping")
            continue

        by_scheduler = defaultdict(list)
        for r in results:
            by_scheduler[r["scheduler"]].append(r)

        dataset_output = {}

        for scheduler_name, sched_results in by_scheduler.items():
            hol = compute_hol_blocking(sched_results, dataset)
            dataset_output[scheduler_name] = hol
            print(f"  Scheduler: {scheduler_name}")
            print(f"    p90/mean ratio (high load): {hol['mean_p90_to_mean_ratio_high_load']:.4f}")
            print(f"    latency slope: {hol['latency_slope_per_req_rate']:.6f}")

        print(f"\n  HOL Blocking Comparison ({dataset_label}):")
        print(f"  {'Scheduler':<28} {'P90/Mean ratio':>16} {'Latency slope':>15}")
        print(f"  {'-'*62}")
        for sname in ["fcfs", "classification", "ltr", "ltr_promptlen"]:
            if sname in dataset_output:
                hol = dataset_output[sname]
                print(f"  {sname:<28} "
                      f"{hol['mean_p90_to_mean_ratio_high_load']:>16.4f} "
                      f"{hol['latency_slope_per_req_rate']:>15.6f}")

        output[dataset_label] = dataset_output

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nHOL blocking metrics saved to {args.output}")


if __name__ == "__main__":
    main()
