# Evaluation report — EXP-002-CORRECTED-NANO (challenge split)

- Generated: 2026-10-02 04:50:19
- Models: EXP-002-CORRECTED-NANO:fp32
- Data: held-out `test`/`challenge` splits only (no training rows)

| Domain | EXP-002-CORRECTED-NANO:fp32 |
| :--- | ---: |
| language | — |
| logic | 0.000 (0/14) |
| math | 0.000 (0/19) |
| algorithmic_reasoning | — |
| code | — |
| debugging | 0.000 (0/13) |
| code_generation | — |
| instruction_following | — |
| tool_selection | 0.000 (0/14) |
| tool_arguments | 0.000 (0/14) |
| tool_syntax | 0.000 (0/14) |
| tool_execution | 0.000 (0/14) |
| grounding | 0.000 (0/14) |
| generalization | 0.000 (0/116) |

## Notes

- Exact-answer domains use normalised string equality on the extracted `<|final|>` answer.
- Code domains execute the generated program together with the record's real assertions
  inside the sandboxed runner (network-disabled); `parseable` counts candidates that pass
  `ast.parse`, `executed` counts runs that actually reached the assertions.
- Tool domains are reported separately: syntax, tool-name accuracy, argument match,
  tool execution success and the answer given after the real result is fed back.
- Accuracy denominators are the full held-out sets; no `[:20]`-style caps are applied.
