# MODEL COMPARISON

Protocol versions are compared only when the tokenizer, dataset fingerprint and evaluation split are identical; otherwise rows are marked **NOT COMPARABLE**.

| Variant | File bytes | Human | Round-trip rel. RMSE | Max abs error |
| :--- | ---: | ---: | ---: | ---: |
| fp32 | 10233776 | 9.76 MB | n/a | n/a |
| fp16 | 5118480 | 4.88 MB | 0.000212 | 0.000488 |
| int8 | 2600976 | 2.48 MB | 0.006438 | 0.001417 |
| int4 | 1451272 | 1.38 MB | 0.096716 | 0.02541 |

- fp32 file: **9.76 MB** (under 50 MB)

## Held-out measurements

- `EXP-004-TOOL-SFT-V3:fp32` on `challenge`: 176 samples, mean accuracy 0.1
- `EXP-004-TOOL-SFT-V3:fp16` on `challenge`: 176 samples, mean accuracy 0.1
- `EXP-004-TOOL-SFT-V3:int8` on `challenge`: 176 samples, mean accuracy 0.1
- `EXP-004-TOOL-SFT-V3:int4` on `challenge`: 176 samples, mean accuracy 0.0
- `EXP-004-TOOL-SFT-V3:fp32` on `test`: 2158 samples, mean accuracy 0.186
- `EXP-004-TOOL-SFT-V3:fp32` (tool suite, 25 cases): argument_accuracy=0.6, argument_validity=1.0, citation_validity=0.0, error_recovery_success=0.0
- `EXP-004-TOOL-SFT-V3:fp16` (tool suite, 25 cases): argument_accuracy=0.6, argument_validity=1.0, citation_validity=0.0, error_recovery_success=0.0
- `EXP-004-TOOL-SFT-V3:int8` (tool suite, 25 cases): argument_accuracy=0.6667, argument_validity=1.0, citation_validity=0.0, error_recovery_success=0.0
- `EXP-004-TOOL-SFT-V3:int4` (tool suite, 25 cases): argument_accuracy=0.0, argument_validity=1.0, citation_validity=0.0, error_recovery_success=0.0

Rows for models trained on a different dataset fingerprint or tokenizer are **NOT COMPARABLE** even if they appear in the same report; check `manifest.json` before drawing conclusions.
