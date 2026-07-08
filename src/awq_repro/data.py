"""Dataset boundaries for calibration and perplexity evaluation.

Expected responsibilities:
- Load tokenizer-compatible text datasets.
- Build calibration samples.
- Build evaluation token streams/windows.
"""


def load_calibration_data(*args, **kwargs):
    """Load calibration examples for activation collection."""
    raise NotImplementedError("TODO: implement calibration data loading")


def load_eval_data(*args, **kwargs):
    """Load evaluation examples for perplexity measurement."""
    raise NotImplementedError("TODO: implement eval data loading")
