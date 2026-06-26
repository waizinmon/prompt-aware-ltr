"""
Thin wrapper around 1_scoring_patch.py so 3_vllm_benchmark.py can
import clean function names without Python's import-with-leading-digit
problem (module names can't start with a digit).

Keep this file in the same directory as 1_scoring_patch.py.
"""

import importlib.util
import os

_here = os.path.dirname(__file__)
_spec = importlib.util.spec_from_file_location(
    "scoring_patch", os.path.join(_here, "1_scoring_patch.py")
)
scoring_patch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scoring_patch)


def rank_requests_ltr_only(ltr_model, tokenizer, prompts):
    return scoring_patch.rank_requests(
        ltr_model, tokenizer, prompts, use_prompt_length=False
    )


def rank_requests_extended(ltr_model, tokenizer, prompts, alpha=0.7, beta=0.3):
    return scoring_patch.rank_requests(
        ltr_model, tokenizer, prompts,
        use_prompt_length=True, alpha=alpha, beta=beta
    )
