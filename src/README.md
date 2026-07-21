# src

Reserved for shared library code.

Right now the whole pipeline (`scripts/0a`–`6`, plus the `plot_*.py` figure
scripts) is self-contained: `scoring_patch_helper.py`, `1_scoring_patch.py`,
`0b_train_predictor.py`, `2_kendalls_tau_eval.py`, `3_vllm_benchmark.py`,
`5_extended_metrics.py`, and `6_hol_blocking_eval.py` load each other via
same-directory relative-path tricks (`os.path.dirname(__file__)`), so they
all live together in `scripts/`.

If pipeline logic gets factored out into an installable/importable module later,
it belongs here.
