# awq

[repro] activation-aware weight quantization

This repo is a small reproduction harness for the main AWQ perplexity result,
adapted to small Pythia models first.

Target experiment shape:

| model | fp16 | rtn_w4a16 | awq_w4a16 |
| --- | --- | --- | --- |
| `EleutherAI/pythia-14m` | TODO | TODO | TODO |
| `EleutherAI/pythia-31m` | TODO | TODO | TODO |
| `EleutherAI/pythia-70m` | TODO | TODO | TODO |

## Repo Layout

```text
configs/
  pythia_small.yaml        # model/dataset/method settings
main.py                    # keep the first implementation here
tests/
  test_scaffold.py         # placeholder test file
results/
  .gitkeep                 # output directory for CSV/JSON results
```

For now, put everything in `main.py`:

- WikiText-2 loading
- Pythia model/tokenizer loading
- FP16 perplexity evaluation
- later: RTN fake quantization
- later: AWQ fake quantization

Only split code into more files once `main.py` becomes painful to edit.

## Intended First Milestone

Get one FP16 perplexity number first:

```text
EleutherAI/pythia-14m on WikiText-2 test
```

Then build the fake-quantized reproduction path:

1. Evaluate FP16 perplexity.
2. Apply RTN weight-only fake quantization and evaluate perplexity.
3. Collect calibration activations.
4. Apply AWQ scaling + fake quantization and evaluate perplexity.
5. Write one row per `(model, method)` to `results/table3_pythia.csv`.

## Non-Goals For The First Pass

- No packed int4 CUDA kernels.
- No real quantized `Linear` replacement.
- No throughput benchmarking.
- No exact reproduction of the OPT table values.
- No mixed-precision `1% FP16` ablation until the core rows work.
