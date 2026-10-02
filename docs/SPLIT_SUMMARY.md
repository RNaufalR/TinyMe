# SPLIT SUMMARY

- **train**: 3062 records / 1592 template families
- **validation**: 259 records / 258 template families
- **test**: 259 records / 97 template families
- **challenge**: 60 records / 5 template families

## Category distribution

| split | algorithm | code_explain | code_gen | code_repair | instruction | language | logic | math | programming | tool_use |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| train | 300 | 380 | 23 | 5 | 4 | 152 | 29 | 911 | 1169 | 89 |
| validation | 0 | 67 | 2 | 1 | 1 | 24 | 0 | 0 | 161 | 3 |
| test | 0 | 23 | 0 | 0 | 0 | 4 | 0 | 163 | 69 | 0 |
| challenge | 0 | 0 | 0 | 13 | 0 | 0 | 14 | 19 | 0 | 14 |

## Contamination

- TRAIN/VALIDATION/TEST CONTAMINATION: PASS
  - `train_vs_validation`: exact=0 normalized=0 minhash=0 code=0 template=0
  - `train_vs_test`: exact=0 normalized=0 minhash=0 code=0 template=0
  - `train_vs_challenge`: exact=0 normalized=0 minhash=0 code=0 template=0
  - `validation_vs_test`: exact=0 normalized=0 minhash=0 code=0 template=0
  - `validation_vs_challenge`: exact=0 normalized=0 minhash=0 code=0 template=0
