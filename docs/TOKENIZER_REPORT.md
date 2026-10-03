# TOKENIZER REPORT

- **Algorithm:** byte-level BPE (`tokenizers`), ByteLevel pre-tokenizer/decoder
- **Tokenizer version:** tok-v3
- **Vocabulary size:** 4096
- **Serialized `tokenizer.json` size:** 266,299 bytes (260.1 KiB) — **counts toward the <50 MB artifact**
- **Special tokens:** <|pad|>, <|bos|>, <|eos|>, <|unk|>, <|endoftext|>, <|system|>, <|user|>, <|assistant|>, <|thought|>, <|answer|>, <|code|>, <|endcode|>, <|tool_call|>, <|end_tool_call|>, <|tool_result|>, <|end_tool_result|>, <|final|>, <|endtool_call|>, <|endtool_result|>

## Per-domain efficiency

| Domain | Samples | Total tokens | Tokens/sample | Chars/token | UNK rate |
| :--- | ---: | ---: | ---: | ---: | ---: |
| all_train | 20088 | 4135713 | 205.88 | 3.177 | 0.000000 |
| code | 2000 | 368827 | 184.413 | 3.1262 | 0.000000 |
| prose | 164 | 36402 | 221.963 | 3.1074 | 0.000000 |
| math | 645 | 51948 | 80.54 | 3.2231 | 0.000000 |
| tool_call_json | 1 | 19 | 19.0 | 3.5263 | 0.000000 |

## Interpretation

- A high chars/token ratio on code means fewer tokens are needed to represent a line of Python, which directly increases the effective context available to a <50 MB model.
- The UNK rate is exactly 0.0 because ByteLevel pre-tokenization falls back to raw bytes, so no input can ever be out-of-vocabulary.
