# SPLIT SUMMARY

- **train**: 1680 records / 1586 template families
- **validation**: 253 records / 147 template families
- **test**: 234 records / 212 template families
- **challenge**: 60 records / 8 template families

## Category distribution

| split | algorithm | code_explain | code_gen | code_repair | instruction | language | logic | math | programming | tool_use |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| train | 6 | 372 | 7 | 4 | 7 | 144 | 4 | 24 | 1098 | 14 |
| validation | 0 | 47 | 16 | 1 | 2 | 18 | 1 | 4 | 161 | 3 |
| test | 0 | 47 | 2 | 2 | 1 | 18 | 5 | 15 | 140 | 4 |
| challenge | 0 | 0 | 8 | 8 | 8 | 8 | 8 | 12 | 0 | 8 |

## Category coverage per split

- **train**: algorithm, code_explain, code_gen, code_repair, instruction, language, logic, math, programming, tool_use
- **validation**: code_explain, code_gen, code_repair, instruction, language, logic, math, programming, tool_use
- **test**: code_explain, code_gen, code_repair, instruction, language, logic, math, programming, tool_use
- **challenge**: code_gen, code_repair, instruction, language, logic, math, tool_use

## Contamination

- TRAIN/VALIDATION/TEST CONTAMINATION: PASS
  - `train_vs_validation`: exact=0 normalized=0 minhash=0 code=0 template=0
  - `train_vs_test`: exact=0 normalized=0 minhash=0 code=0 template=0
  - `train_vs_challenge`: exact=0 normalized=0 minhash=0 code=0 template=0
  - `validation_vs_test`: exact=0 normalized=0 minhash=0 code=0 template=0
  - `validation_vs_challenge`: exact=0 normalized=0 minhash=0 code=0 template=0
