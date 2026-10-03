# MODEL COMPARISON

Protocol versions are compared only when the tokenizer, dataset fingerprint and evaluation split are identical; otherwise rows are marked **NOT COMPARABLE**.

| Variant | File bytes | Human | Round-trip rel. RMSE | Max abs error |
| :--- | ---: | ---: | ---: | ---: |
| fp32 | 10233776 | 9.76 MB | n/a | n/a |
| fp16 | 5118480 | 4.88 MB | 0.000211 | 0.000488 |
| int8 | 2600976 | 2.48 MB | 0.006483 | 0.001408 |
| int4 | 1451272 | 1.38 MB | 0.097674 | 0.02538 |

- fp32 file: **9.76 MB** (under 50 MB)

## Held-out measurements

- `EXP-002-CORRECTED-NANO:fp32` on `challenge`: 232 samples, mean accuracy 0.0
- `EXP-002-CORRECTED-NANO:fp32` on `test`: 60 samples, mean accuracy 0.0
- `EXP-002-CORRECTED-NANO:fp16` on `test`: 60 samples, mean accuracy 0.0
- `EXP-002-CORRECTED-NANO:int8` on `test`: 60 samples, mean accuracy 0.0
- `EXP-002-CORRECTED-NANO:int4` on `test`: 60 samples, mean accuracy 0.0

Rows for models trained on a different dataset fingerprint or tokenizer are **NOT COMPARABLE** even if they appear in the same report; check `manifest.json` before drawing conclusions.
