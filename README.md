# awq

[repro] activation-aware weight quantization

This repo is structured as a small reproduction harness for the main AWQ
perplexity result, adapted to small Pythia models first.

Target experiment shape:

| model | fp16 | rtn_w4a16 | awq_w4a16 |
| --- | --- | --- | --- |
| `EleutherAI/pythia-14m` | TODO | TODO | TODO |
| `EleutherAI/pythia-31m` | TODO | TODO | TODO |
| `EleutherAI/pythia-70m` | TODO | TODO | TODO |

## Repo Layout

```text
configs/
  pythia_small.yaml        # first reproduction config
src/awq_repro/
  data.py                  # calibration/evaluation dataset loading
  eval_ppl.py              # shared perplexity evaluator
  model.py                 # model/tokenizer loading boundaries
  quant.py                 # RTN/fake-quant boundaries
  collect_acts.py          # activation collection boundaries
  awq.py                   # AWQ scale-search boundaries
  run_table.py             # experiment orchestration entrypoint
tests/
  test_scaffold.py         # placeholder test file
results/
  .gitkeep                 # output directory for CSV/JSON results
```

## Intended First Milestone

Build only the fake-quantized reproduction path:

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
