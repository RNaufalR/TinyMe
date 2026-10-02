# Evaluation report — EXP-004-TOOL-SFT-V2 (test split)

- Generated: 2026-10-02 12:02:18
- Models: EXP-004-TOOL-SFT-V2:fp32
- Data: held-out `test`/`challenge` splits only (no training rows)

| Domain | EXP-004-TOOL-SFT-V2:fp32 |
| :--- | ---: |
| language | ppl=620.7797 |
| logic | 0.252 (55/218) |
| math | 0.003 (1/407) |
| algorithmic_reasoning | — |
| code | 0.000 (0/140) |
| debugging | — |
| code_generation | 0.000 (0/200) |
| instruction_following | 0.000 (0/142) |
| tool_selection | 0.020 (3/150) |
| tool_arguments | 0.000 (0/150) |
| tool_syntax | 0.207 (31/150) |
| tool_execution | 0.147 (0/150) |
| grounding | 0.000 (0/150) |
| generalization | — |

## Notes

- Exact-answer domains use normalised string equality on the extracted `<|final|>` answer.
- Code domains execute the generated program together with the record's real assertions
  inside the sandboxed runner (network-disabled); `parseable` counts candidates that pass
  `ast.parse`, `executed` counts runs that actually reached the assertions.
- Tool domains are reported separately: syntax, tool-name accuracy, argument match,
  tool execution success and the answer given after the real result is fed back.
- Accuracy denominators are the full held-out sets; no `[:20]`-style caps are applied.
