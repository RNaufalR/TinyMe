# SPLIT SUMMARY

- **train**: 5139 records / 1850 template families
- **validation**: 360 records / 9 template families
- **test**: 708 records / 10 template families
- **challenge**: 60 records / 8 template families

## Category distribution

| split | algorithm | code_explain | code_gen | code_repair | instruction | language | logic | math | programming | tool_use |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| train | 0 | 632 | 450 | 254 | 302 | 139 | 300 | 1287 | 1372 | 403 |
| validation | 0 | 1 | 50 | 63 | 71 | 15 | 92 | 51 | 1 | 16 |
| test | 300 | 30 | 15 | 32 | 63 | 1 | 100 | 15 | 1 | 151 |
| challenge | 0 | 0 | 8 | 8 | 8 | 8 | 8 | 12 | 0 | 8 |

## Category coverage per split

- **train**: code_explain, code_gen, code_repair, instruction, language, logic, math, programming, tool_use
- **validation**: code_explain, code_gen, code_repair, instruction, language, logic, math, programming, tool_use
- **test**: algorithm, code_explain, code_gen, code_repair, instruction, language, logic, math, programming, tool_use
- **challenge**: code_gen, code_repair, instruction, language, logic, math, tool_use

## Contamination

- TRAIN/VALIDATION/TEST CONTAMINATION: PASS
  - `train_vs_validation`: exact=0 normalized=0 minhash=0 code=0 template=0
  - `train_vs_test`: exact=0 normalized=0 minhash=0 code=0 template=0
  - `train_vs_challenge`: exact=0 normalized=0 minhash=0 code=0 template=0
  - `validation_vs_test`: exact=0 normalized=0 minhash=0 code=0 template=0
  - `validation_vs_challenge`: exact=0 normalized=0 minhash=0 code=0 template=0
