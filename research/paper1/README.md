# Paper-1 frozen benchmark reproduction

This directory is deliberately outside the stable `ugrf` package API.

Run:

```bash
python research/paper1/reproduce_benchmark.py --download --data-dir ./data
```

The script evaluates only the canonical TSB, Renewal, and UGRF components needed for the frozen-ratio guard. It uses the published panel shapes and benchmark split locations:

- Car Parts: utility origins 30, 33, 36, 39; final origin 45
- AUTO: utility origin 12; final origin 18
- RAF: utility origins 48, 54, 60, 66, 72; final origin 78

Expected UGRF/TSB total-absolute-error ratios:

- Car Parts: 0.9918971759288252
- AUTO: 0.9761486034991342
- RAF: 0.8905672729887494

The assertion tolerance is `5e-6` absolute.
