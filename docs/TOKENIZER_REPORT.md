# TOKENIZER REPORT

- **Algorithm:** byte-level BPE (`tokenizers`), ByteLevel pre-tokenizer/decoder
- **Tokenizer version:** tok-v2
- **Vocabulary size:** 4096
- **Serialized `tokenizer.json` size:** 266,195 bytes (260.0 KiB) — **counts toward the <50 MB artifact**
- **Special tokens:** <|pad|>, <|bos|>, <|eos|>, <|unk|>, <|endoftext|>, <|system|>, <|user|>, <|assistant|>, <|thought|>, <|answer|>, <|code|>, <|endcode|>, <|tool_call|>, <|end_tool_call|>, <|tool_result|>, <|end_tool_result|>, <|final|>, <|endtool_call|>, <|endtool_result|>

## Per-domain efficiency

| Domain | Samples | Total tokens | Tokens/sample | Chars/token | UNK rate |
| :--- | ---: | ---: | ---: | ---: | ---: |
| all_train | 3062 | 567543 | 185.35 | 3.7149 | 0.000000 |
| code | 1877 | 457324 | 243.646 | 3.7397 | 0.000000 |
| prose | 152 | 32279 | 212.362 | 3.2131 | 0.000000 |
| math | 940 | 66510 | 70.755 | 3.8541 | 0.000000 |
| tool_call_json | 1 | 21 | 21.0 | 3.1905 | 0.000000 |

## Interpretation

- A high chars/token ratio on code means fewer tokens are needed to represent a line of Python, which directly increases the effective context available to a <50 MB model.
- The UNK rate is exactly 0.0 because ByteLevel pre-tokenization falls back to raw bytes, so no input can ever be out-of-vocabulary.
