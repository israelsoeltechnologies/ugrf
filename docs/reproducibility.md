# Reproducibility

The repository contains the frozen benchmark reproduction script:

```text
research/paper1/reproduce_benchmark.py
```

## Run the benchmark

From the repository root:

```bash
pip install -e .
python research/paper1/reproduce_benchmark.py --download
```

To use another data directory:

```bash
python research/paper1/reproduce_benchmark.py \
    --data-dir ./my_data \
    --download
```

## Frozen benchmark protocol

| Dataset | Shape | Utility origins | Final origin | Horizon |
|---|---:|---|---:|---:|
| Car Parts | `2674 × 51` | `30, 33, 36, 39` | `45` | `6` |
| AUTO | `3000 × 24` | `12` | `18` | `6` |
| RAF | `5000 × 84` | `48, 54, 60, 66, 72` | `78` | `6` |

## Frozen target ratios

| Dataset | UGRF / TSB |
|---|---:|
| Car Parts | `0.9918971759288252` |
| AUTO | `0.9761486034991342` |
| RAF | `0.8905672729887494` |

The reproduction guard uses absolute tolerance `5e-6` and zero relative tolerance.

## Why benchmark code is separate

The stable public API stays small, while dataset downloads, exact benchmark windows, and publication guards remain under `research/paper1/` to preserve provenance without making benchmark assumptions part of normal package use.
