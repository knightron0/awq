"""Experiment orchestration entrypoint.

Intended flow:
1. Read a config.
2. Iterate over model names and methods.
3. Build the requested model variant.
4. Evaluate perplexity with the shared evaluator.
5. Write CSV/JSON results.
"""


def main():
    """Run the configured table reproduction."""
    raise NotImplementedError("TODO: implement experiment orchestration")


if __name__ == "__main__":
    main()
