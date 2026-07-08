"""AWQ scale-search and application boundaries."""


def search_awq_scales(*args, **kwargs):
    """Search activation-aware scaling factors for target layers."""
    raise NotImplementedError("TODO: implement AWQ scale search")


def apply_awq_fake_quantization(*args, **kwargs):
    """Apply AWQ scaling plus weight-only fake quantization to a model."""
    raise NotImplementedError("TODO: implement AWQ fake quantization")
