# src

Reserved for shared library code.

Right now the whole pipeline (`scripts/0a`–`4`) is self-contained: `scoring_patch_helper.py`,
`1_scoring_patch.py`, `0b_train_predictor.py`, `2_kendalls_tau_eval.py`, and
`3_vllm_benchmark.py` load each other via same-directory relative-path tricks
(`os.path.dirname(__file__)`), so they all live together in `scripts/`.

If pipeline logic gets factored out into an installable/importable module later,
it belongs here.
