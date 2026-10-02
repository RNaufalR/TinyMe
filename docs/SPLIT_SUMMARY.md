# SPLIT SUMMARY

- **train**: 4227 records / 1631 template families
- **validation**: 741 records / 114 template families
- **test**: 810 records / 206 template families
- **challenge**: 60 records / 5 template families

## Category distribution

| split | algorithm | code_explain | code_gen | code_repair | instruction | language | logic | math | programming | tool_use |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| train | 300 | 515 | 350 | 191 | 190 | 144 | 300 | 977 | 1098 | 162 |
| validation | 0 | 68 | 65 | 63 | 36 | 18 | 95 | 154 | 161 | 81 |
| test | 0 | 96 | 100 | 96 | 42 | 18 | 93 | 178 | 140 | 47 |
| challenge | 0 | 0 | 0 | 13 | 0 | 0 | 14 | 19 | 0 | 14 |

## Category coverage per split

- **train**: algorithm, code_explain, code_gen, code_repair, instruction, language, logic, math, programming, tool_use
- **validation**: code_explain, code_gen, code_repair, instruction, language, logic, math, programming, tool_use
- **test**: code_explain, code_gen, code_repair, instruction, language, logic, math, programming, tool_use
- **challenge**: code_repair, logic, math, tool_use

## Contamination

- TRAIN/VALIDATION/TEST CONTAMINATION: PASS
  - `train_vs_validation`: exact=0 normalized=0 minhash=0 code=0 template=0
  - `train_vs_test`: exact=0 normalized=0 minhash=0 code=0 template=0
  - `train_vs_challenge`: exact=0 normalized=0 minhash=0 code=0 template=0
  - `validation_vs_test`: exact=0 normalized=0 minhash=0 code=0 template=0
  - `validation_vs_challenge`: exact=0 normalized=0 minhash=0 code=0 template=0
