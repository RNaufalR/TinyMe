# SPLIT SUMMARY

- **train**: 14787 records / 1714 template families
- **validation**: 2389 records / 142 template families
- **test**: 2265 records / 224 template families
- **challenge**: 90 records / 11 template families

## Category distribution

| split | algorithm | code_explain | code_gen | code_repair | instruction | language | logic | math | programming | tool_use |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| train | 1283 | 449 | 140 | 76 | 6439 | 144 | 125 | 393 | 1098 | 4640 |
| validation | 256 | 60 | 35 | 25 | 1034 | 18 | 33 | 64 | 161 | 703 |
| test | 181 | 57 | 40 | 39 | 1006 | 18 | 41 | 102 | 140 | 641 |
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
