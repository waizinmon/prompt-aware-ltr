# prompt-aware-ltr

Capstone extension of ["Efficient LLM Scheduling by Learning to Rank"](https://arxiv.org/abs/2408.15792) (Fu et al., 2024 — [hao-ai-lab/vllm-ltr](https://github.com/hao-ai-lab/vllm-ltr)).

This project adds two extensions on top of the original vllm-ltr predictor:

- **Prompt length as a secondary ranking signal** — `scripts/1_scoring_patch.py` combines the learned predictor score with prompt length when computing scheduling priority.
- **Pairwise margin-ranking loss** — `scripts/0b_train_predictor.py` can train the OPT-125M predictor with either the original paper's ListMLE loss or a new pairwise margin loss (`--loss_type margin`).

## Repository layout

```
prompt-aware-ltr/
├── README.md
├── src/            # reserved for shared library code (see src/README.md)
├── scripts/        # the experiment pipeline, run in numeric order
├── data/           # datasets — see data/README.md (not committed, see .gitignore)
├── checkpoints/    # trained predictors — see checkpoints/README.md (not committed)
└── .gitignore
```

## Dependencies

This pipeline runs against a vanilla, pip-installed `vllm` — it does **not**
clone or patch the vllm-ltr engine source. `scripts/1_scoring_patch.py`
pre-computes a request order from the predictor score (+ prompt length),
then submits that order as a single batch to `vllm.LLM(...).generate(...)`;
vLLM's own continuous-batching scheduler handles the rest.

```bash
conda create -n prompt-aware-ltr python=3.10
conda activate prompt-aware-ltr
pip install "transformers==4.44.2" "tokenizers==0.19.1" vllm datasets scipy matplotlib scikit-learn huggingface_hub
```

`transformers`/`tokenizers` are pinned together to avoid a version conflict
with `vllm`'s own pins. `vllm` itself is left unpinned — newer releases avoid
a broken `pyairports`/`outlines` dependency present in `vllm==0.5.4`'s
guided-decoding feature.

## Setup

Large files (trained checkpoints and datasets) are stored on Google Drive, not in this repo.

1. Download the models and datasets:
   https://drive.google.com/... *(replace with your share link)*

2. Place them in:
   ```
   data/
   checkpoints/
   ```

3. Run:
   ```bash
   python scripts/0b_train_predictor.py
   ```
   (run from the repository root — the scripts use relative paths like `./data/...` and `./checkpoints/...`)

## Pipeline

Run in order from the repository root:

| step | script | what it does | where it runs |
|---|---|---|---|
| 0a | `scripts/0a_build_dataset.py` | Builds train/eval splits from LMSYS-Chat-1M | CPU (e.g. Mac) |
| 0b | `scripts/0b_train_predictor.py` | Fine-tunes the OPT-125M predictor (`--loss_type listmle` for the original baseline, `--loss_type margin` for this project's extension) | CPU/MPS/CUDA |
| 1 | `scripts/1_scoring_patch.py` + `scripts/scoring_patch_helper.py` | Defines the prompt-length-aware request ranking used to pre-order the batch in step 3 | n/a (library, imported by step 3) |
| 2 | `scripts/2_kendalls_tau_eval.py` | Evaluates ranking quality via Kendall's Tau | CPU/MPS |
| 3 | `scripts/3_vllm_benchmark.py` | Runs the vLLM latency-vs-request-rate benchmark across schedulers | CUDA GPU (e.g. Colab) |
| 4 | `scripts/4_plot_results.py` | Plots the latency comparison chart from step 3's output | CPU |
| 5 | `scripts/5_extended_metrics.py` | Computes throughput (from step 3's result files) and, optionally, live GPU memory/utilization profiling | CPU (GPU optional, for `--run_gpu_profiling`) |
| 6 | `scripts/6_hol_blocking_eval.py` | Estimates Head-of-Line blocking reduction (p90/mean latency ratio, latency-vs-rate slope) from step 3's result files | CPU |

## Citation

```
@article{fu2024efficient,
  title={Efficient LLM Scheduling by Learning to Rank},
  author={Fu, Yichao and Zhu, Siqi and Su, Runlong and Qiao, Aurick and Stoica, Ion and Zhang, Hao},
  journal={arXiv preprint arXiv:2408.15792},
  year={2024}
}
```
