# SPLIT SUMMARY

- **train**: 20088 records / 2020 template families
- **validation**: 2886 records / 25 template families
- **test**: 3768 records / 27 template families
- **challenge**: 90 records / 11 template families

## Category distribution

| split | algorithm | code_explain | code_gen | code_repair | instruction | language | logic | math | programming | tool_use |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| train | 1151 | 542 | 180 | 102 | 10440 | 164 | 125 | 520 | 1397 | 5467 |
| validation | 251 | 12 | 20 | 25 | 1413 | 15 | 33 | 24 | 1 | 1092 |
| test | 318 | 12 | 15 | 13 | 2290 | 1 | 41 | 15 | 1 | 1062 |
| challenge | 0 | 0 | 8 | 8 | 8 | 8 | 8 | 12 | 0 | 38 |

## Category coverage per split

- **train**: algorithm, code_explain, code_gen, code_repair, instruction, language, logic, math, programming, tool_use
- **validation**: algorithm, code_explain, code_gen, code_repair, instruction, language, logic, math, programming, tool_use
- **test**: algorithm, code_explain, code_gen, code_repair, instruction, language, logic, math, programming, tool_use
- **challenge**: code_gen, code_repair, instruction, language, logic, math, tool_use

## Contamination

- TRAIN/VALIDATION/TEST CONTAMINATION: PASS
  - `train_vs_validation`: exact=0 normalized=0 minhash=0 code=0 template=0
  - `train_vs_test`: exact=0 normalized=0 minhash=0 code=0 template=0
  - `train_vs_challenge`: exact=0 normalized=0 minhash=0 code=0 template=0
  - `validation_vs_test`: exact=0 normalized=0 minhash=0 code=0 template=0
  - `validation_vs_challenge`: exact=0 normalized=0 minhash=0 code=0 template=0
