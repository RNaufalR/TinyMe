# Evaluation report — EXP-002-CORRECTED-NANO (test split)

- Generated: 2026-10-02 04:50:14
- Models: EXP-002-CORRECTED-NANO:fp32
- Data: held-out `test`/`challenge` splits only (no training rows)

| Domain | EXP-002-CORRECTED-NANO:fp32 |
| :--- | ---: |
| language | ppl=259.2335 |
| logic | 0.000 (0/93) |
| math | 0.000 (0/178) |
| algorithmic_reasoning | — |
| code | 0.000 (0/140) |
| debugging | 0.000 (0/96) |
| code_generation | 0.000 (0/100) |
| instruction_following | 0.000 (0/42) |
| tool_selection | 0.000 (0/47) |
| tool_arguments | 0.000 (0/47) |
| tool_syntax | 0.000 (0/47) |
| tool_execution | 0.000 (0/47) |
| grounding | 0.000 (0/47) |
| generalization | — |

## Notes

- Exact-answer domains use normalised string equality on the extracted `<|final|>` answer.
- Code domains execute the generated program together with the record's real assertions
  inside the sandboxed runner (network-disabled); `parseable` counts candidates that pass
  `ast.parse`, `executed` counts runs that actually reached the assertions.
- Tool domains are reported separately: syntax, tool-name accuracy, argument match,
  tool execution success and the answer given after the real result is fed back.
- Accuracy denominators are the full held-out sets; no `[:20]`-style caps are applied.
