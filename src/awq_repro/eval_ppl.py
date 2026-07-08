"""Shared perplexity evaluator.

All methods should flow through this module so FP16, RTN, and AWQ differ only
in model weights, not in evaluation mechanics.
"""


def evaluate_perplexity(*args, **kwargs):
    """Evaluate causal language-model perplexity."""
    raise NotImplementedError("TODO: implement perplexity evaluation")
