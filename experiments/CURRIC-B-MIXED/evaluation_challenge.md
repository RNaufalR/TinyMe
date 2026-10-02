# Evaluation report — CURRIC-B-MIXED (challenge split)

- Generated: 2026-10-02 13:03:24
- Models: CURRIC-B-MIXED:fp32
- Data: held-out `test`/`challenge` splits only (no training rows)

| Domain | CURRIC-B-MIXED:fp32 |
| :--- | ---: |
| language | ppl=582.5454 |
| logic | 0.000 (0/2) |
| math | 0.000 (0/3) |
| algorithmic_reasoning | — |
| code | — |
| debugging | — |
| code_generation | 0.000 (0/3) |
| instruction_following | 0.000 (0/2) |
| tool_selection | 0.000 (0/2) |
| tool_arguments | 0.000 (0/2) |
| tool_syntax | 0.000 (0/2) |
| tool_execution | 0.000 (0/2) |
| grounding | 0.000 (0/2) |
| generalization | 0.000 (0/22) |

## Notes

- Exact-answer domains use normalised string equality on the extracted `<|final|>` answer.
- Code domains execute the generated program together with the record's real assertions
  inside the sandboxed runner (network-disabled); `parseable` counts candidates that pass
  `ast.parse`, `executed` counts runs that actually reached the assertions.
- Tool domains are reported separately: syntax, tool-name accuracy, argument match,
  tool execution success and the answer given after the real result is fed back.
- Accuracy denominators are the full held-out sets; no `[:20]`-style caps are applied.
