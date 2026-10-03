# Tool-mechanism coverage — dataset_v8

Every count below is produced by `scripts/tool_coverage_report.py` from the split JSONL files; a mechanism counts as covered when the train split holds at least 20 records with at least 5 distinct prompt shapes (digits and code blocks masked), so a passing check cannot be satisfied by repeating one template.

| Mechanism | Held-out family | train records | train shapes | validation | test | challenge |
| :--- | :--- | --: | --: | --: | --: | --: |
| M1 — answer with no tool | A | 15204 / 6732 | 1954 / 698 | 2844 / 1554 | 52 / 24 |
| M2 — compute (copied expression) | B | 1200 / 398 | 73 / 37 | 75 / 35 | 10 / 1 |
| M3 — search → grounded answer | C | 2042 / 1869 | 492 / 440 | 479 / 435 | 8 / 1 |
| M4 — search → fetch (multi-step) | D | 2042 / 1869 | 492 / 440 | 479 / 435 | 8 / 1 |
| M5 — fetch a named source id | D3 | 190 / 163 | 0 / 0 | 0 / 0 | 0 / 0 |
| M6 — run a supplied program | E | 1026 / 388 | 254 / 64 | 246 / 60 | 0 / 0 |
| M7 — repair failing code | F | 398 / 385 | 66 / 63 | 63 / 59 | 0 / 0 |
| M8 — failure → corrected call | G | 835 / 469 | 169 / 148 | 157 / 142 | 0 / 0 |
| M9 — citation in the final answer | H | 1262 / 460 | 88 / 52 | 85 / 45 | 28 / 3 |

Train-split floor gaps: none (floor: 20 records / 5 shapes)

Raw: `docs/audit_evidence/tool_coverage.json`

