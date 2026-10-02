# Evaluation report — EXP-004-TOOL-SFT-V3 (challenge split)

- Generated: 2026-10-02 12:20:48
- Models: EXP-004-TOOL-SFT-V3:fp32
- Data: held-out `test`/`challenge` splits only (no training rows)

| Domain | EXP-004-TOOL-SFT-V3:fp32 |
| :--- | ---: |
| language | ppl=779.7545 |
| logic | 0.000 (0/8) |
| math | 0.000 (0/12) |
| algorithmic_reasoning | — |
| code | — |
| debugging | — |
| code_generation | 0.000 (0/8) |
| instruction_following | 0.000 (0/8) |
| tool_selection | 0.000 (0/8) |
| tool_arguments | 0.000 (0/8) |
| tool_syntax | 1.000 (8/8) |
| tool_execution | 0.000 (0/8) |
| grounding | 0.000 (0/8) |
| generalization | 0.100 (0/84) |

## Notes

- Exact-answer domains use normalised string equality on the extracted `<|final|>` answer.
- Code domains execute the generated program together with the record's real assertions
  inside the sandboxed runner (network-disabled); `parseable` counts candidates that pass
  `ast.parse`, `executed` counts runs that actually reached the assertions.
- Tool domains are reported separately: syntax, tool-name accuracy, argument match,
  tool execution success and the answer given after the real result is fed back.
- Accuracy denominators are the full held-out sets; no `[:20]`-style caps are applied.
