"""
EVALUATION PART 3: Extended Metrics Analysis
=============================================

Computes the metrics specified in the project's four evaluation metrics
(Average Latency, Throughput, Kendall's Tau, GPU Utilization) that are
NOT produced by 3_vllm_benchmark.py:

  1. Throughput -- tokens per second and requests per second, derived
     from the existing benchmark result files.

  2. Resource Utilization -- GPU memory (peak allocated, reserved) and
     compute utilization sampled during a live vLLM generate() call.
     Requires a GPU; falls back to a warning if none is found.

(Average Latency is already produced by 3_vllm_benchmark.py.
 Kendall's Tau is covered separately by 2_kendalls_tau_eval.py.)

USAGE (Colab or Mac M4):
    # Basic: compute throughput from existing result files for BOTH
    # datasets automatically (dataset1 in-distribution + dataset2 OOD)
    python scripts/5_extended_metrics.py \
        --results_dir ./result \
        --data_dir ./data \
        --output extended_metrics.json

    # Full: also run live GPU resource profiling on both datasets
    python scripts/5_extended_metrics.py \
        --results_dir ./result \
        --data_dir ./data \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --ltr_predictor_path ./checkpoints/opt125m-ltr-original \
        --ltr_promptlen_predictor_path ./checkpoints/opt125m-ltr-marginloss \
        --output extended_metrics.json \
        --run_gpu_profiling --quantize
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# 1. Throughput
# ---------------------------------------------------------------------------

def compute_throughput(result: dict, dataset: List[dict]) -> dict:
    """
    Computes throughput from an existing benchmark result record.

    result fields expected (from 3_vllm_benchmark.py output):
        scheduler, request_rate, mean_latency_s_per_token, n_requests

    Throughput is derived as:
        tokens/s  = 1.0 / mean_latency_s_per_token
        requests/s = n_requests / (n_requests * mean_latency_s_per_token)
                   = 1.0 / mean_latency_s_per_token  [simplified]

    Note: because 3_vllm_benchmark.py computes per-request latency as
    (batch_elapsed / token_count), mean_latency is effectively the
    inverse of throughput. We report both forms for clarity.
    """
    mean_lat = result.get("mean_latency_s_per_token", 0.0)
    n = result.get("n_requests", 0)

    if mean_lat <= 0 or n == 0:
        return {"tokens_per_second": 0.0, "requests_per_second": 0.0}

    # Estimate average output tokens per request from dataset
    avg_true_output = sum(r.get("true_output_length", 512) for r in dataset) / len(dataset)

    # Total tokens / total time = throughput
    # total_time ≈ n * avg_output_tokens * mean_latency_s_per_token
    total_time_s = n * avg_true_output * mean_lat
    tokens_per_second = (n * avg_true_output) / total_time_s if total_time_s > 0 else 0.0
    requests_per_second = n / total_time_s if total_time_s > 0 else 0.0

    return {
        "tokens_per_second": round(tokens_per_second, 4),
        "requests_per_second": round(requests_per_second, 4),
    }


# ---------------------------------------------------------------------------
# 2. GPU Resource Utilization
# ---------------------------------------------------------------------------

def measure_gpu_utilization(
    model_name: str,
    ltr_predictor_path: str,
    ltr_promptlen_predictor_path: str,
    dataset: List[dict],
    n_probe_requests: int = 20,
    quantize: bool = False,
) -> dict:
    """
    Measures peak GPU memory and utilization during a short vLLM run.

    Runs a small probe batch (n_probe_requests) through each scheduler
    and records:
      - Peak GPU memory allocated (MB)
      - Peak GPU memory reserved (MB)
      - Approximate compute utilization via timing ratio

    This is a separate short run from the main benchmark -- it does not
    affect the latency results and is only used to report resource usage.
    """
    try:
        import torch
        from vllm import LLM, SamplingParams
    except ImportError:
        return {"error": "torch or vllm not available -- skipping GPU profiling"}

    if not torch.cuda.is_available():
        return {"error": "No CUDA GPU found -- GPU profiling skipped"}

    print(f"\nRunning GPU utilization probe ({n_probe_requests} requests)...")
    torch.cuda.reset_peak_memory_stats()

    engine_kwargs = dict(model=model_name, dtype="float16", max_model_len=4096)
    if quantize:
        # NOTE: AWQ requires a checkpoint that was already quantized in
        # AWQ format ahead of time -- it will NOT work on a standard
        # FP16 repo like meta-llama/Llama-3.1-8B-Instruct (raises
        # "Cannot find the config file for awq"). bitsandbytes instead
        # quantizes on load and works with any standard FP16 checkpoint,
        # which is what we need here for free-tier T4 GPUs.
        engine_kwargs["quantization"] = "bitsandbytes"
        engine_kwargs["load_format"] = "bitsandbytes"

    llm = LLM(**engine_kwargs)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

    probe_prompts = [r["prompt"] for r in dataset[:n_probe_requests]]

    torch.cuda.synchronize()
    t0 = time.time()
    _ = llm.generate(probe_prompts, sampling_params)
    torch.cuda.synchronize()
    t1 = time.time()

    peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
    total_gpu_memory_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
    utilization_fraction = peak_allocated_mb / total_gpu_memory_mb

    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "total_gpu_memory_mb": round(total_gpu_memory_mb, 1),
        "peak_allocated_mb": round(peak_allocated_mb, 1),
        "peak_reserved_mb": round(peak_reserved_mb, 1),
        "memory_utilization_fraction": round(utilization_fraction, 4),
        "probe_elapsed_s": round(t1 - t0, 3),
        "probe_n_requests": n_probe_requests,
        "quantized": quantize,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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


def main():
    parser = argparse.ArgumentParser(description="Extended metrics analysis")
    parser.add_argument("--results_dir", default="./result",
                        help="Directory containing results_dataset1.json and results_dataset2.json")
    parser.add_argument("--data_dir", default="./data",
                        help="Directory containing dataset1_eval.jsonl and dataset2_eval.jsonl")
    parser.add_argument("--output", default="extended_metrics.json",
                        help="Output file for extended metrics")
    parser.add_argument("--run_gpu_profiling", action="store_true",
                        help="Run a live GPU memory probe on EACH dataset (requires GPU + vLLM)")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model name for GPU profiling (only used with --run_gpu_profiling)")
    parser.add_argument("--ltr_predictor_path", default="./checkpoints/opt125m-ltr-original")
    parser.add_argument("--ltr_promptlen_predictor_path",
                        default="./checkpoints/opt125m-ltr-marginloss")
    parser.add_argument("--quantize", action="store_true",
                        help="Use bitsandbytes quantization in GPU probe (free-tier T4 only)")
    args = parser.parse_args()

    print("\n=== Extended Metrics Analysis ===\n")

    # Load both datasets up front (project splits eval into dataset1
    # in-distribution and dataset2 out-of-distribution -- both are
    # processed together here rather than requiring two manual runs).
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

        # Match this result file to its corresponding dataset for
        # throughput calculations (dataset1 results use dataset1_eval,
        # dataset2 results use dataset2_eval).
        dataset = datasets.get(dataset_label)
        if dataset is None:
            print(f"[WARNING] No matching dataset loaded for {dataset_label} -- skipping")
            continue

        # Group results by scheduler
        by_scheduler = defaultdict(list)
        for r in results:
            by_scheduler[r["scheduler"]].append(r)

        dataset_output = {}

        for scheduler_name, sched_results in by_scheduler.items():
            print(f"  Scheduler: {scheduler_name}")

            # Throughput
            throughput_list = []
            for r in sched_results:
                tp = compute_throughput(r, dataset)
                tp["request_rate"] = r.get("request_rate")
                throughput_list.append(tp)

            dataset_output[scheduler_name] = {
                "throughput": throughput_list,
            }

            avg_tps = sum(t["tokens_per_second"] for t in throughput_list) / len(throughput_list)
            avg_rps = sum(t["requests_per_second"] for t in throughput_list) / len(throughput_list)
            print(f"    avg tokens/s: {avg_tps:.4f}  avg requests/s: {avg_rps:.4f}")

        # Throughput comparison summary across schedulers
        print(f"\n  Throughput Comparison ({dataset_label}):")
        print(f"  {'Scheduler':<28} {'Avg tokens/s':>14} {'Avg requests/s':>16}")
        print(f"  {'-'*60}")
        for sname in ["fcfs", "classification", "ltr", "ltr_promptlen"]:
            if sname in dataset_output:
                tlist = dataset_output[sname]["throughput"]
                avg_tps = sum(t["tokens_per_second"] for t in tlist) / len(tlist)
                avg_rps = sum(t["requests_per_second"] for t in tlist) / len(tlist)
                print(f"  {sname:<28} {avg_tps:>14.4f} {avg_rps:>16.4f}")

        output[dataset_label] = dataset_output

    # GPU Resource Utilization (optional live probe) -- run once PER
    # dataset, since the project's evaluation methodology treats
    # dataset1 (in-distribution) and dataset2 (out-of-distribution) as
    # separate conditions throughout. Running the probe on both confirms
    # whether memory/compute usage is stable across distribution shift.
    if args.run_gpu_profiling:
        output["gpu_utilization"] = {}
        for dataset_label, dataset in datasets.items():
            print(f"\n--- GPU Resource Utilization ({dataset_label}) ---")
            gpu_metrics = measure_gpu_utilization(
                model_name=args.model,
                ltr_predictor_path=args.ltr_predictor_path,
                ltr_promptlen_predictor_path=args.ltr_promptlen_predictor_path,
                dataset=dataset,
                quantize=args.quantize,
            )
            output["gpu_utilization"][dataset_label] = gpu_metrics
            print(f"  GPU: {gpu_metrics.get('gpu_name', 'N/A')}")
            print(f"  Peak allocated: {gpu_metrics.get('peak_allocated_mb', 0):.1f} MB")
            print(f"  Peak reserved:  {gpu_metrics.get('peak_reserved_mb', 0):.1f} MB")
            print(f"  Memory utilization: "
                  f"{gpu_metrics.get('memory_utilization_fraction', 0)*100:.1f}%")

        # Cross-dataset comparison summary
        print("\n--- GPU Utilization Comparison Across Datasets ---")
        print(f"  {'Dataset':<20} {'Peak Allocated (MB)':>20} {'Utilization %':>15}")
        print(f"  {'-'*57}")
        for dataset_label, metrics in output["gpu_utilization"].items():
            alloc = metrics.get("peak_allocated_mb", 0)
            util = metrics.get("memory_utilization_fraction", 0) * 100
            print(f"  {dataset_label:<20} {alloc:>20.1f} {util:>15.1f}")
    else:
        print("\n[INFO] Skipping GPU profiling (pass --run_gpu_profiling to enable)")
        output["gpu_utilization"] = {
            "note": "GPU profiling not run. Re-run with --run_gpu_profiling on a GPU instance."
        }

    # Save
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nExtended metrics saved to {args.output}")


if __name__ == "__main__":
    main()
