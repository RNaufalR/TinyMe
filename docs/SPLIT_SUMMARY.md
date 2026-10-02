# SPLIT SUMMARY

- **train**: 7220 records / 1641 template families
- **validation**: 1422 records / 112 template families
- **test**: 1558 records / 198 template families
- **challenge**: 60 records / 8 template families

## Category distribution

| split | algorithm | code_explain | code_gen | code_repair | instruction | language | logic | math | programming | tool_use |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| train | 600 | 697 | 700 | 382 | 588 | 144 | 552 | 1911 | 1098 | 548 |
| validation | 0 | 124 | 115 | 127 | 107 | 18 | 189 | 264 | 161 | 317 |
| test | 0 | 92 | 200 | 191 | 142 | 18 | 218 | 407 | 140 | 150 |
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
