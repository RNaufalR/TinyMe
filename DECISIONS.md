# ENGINEERING & ARCHITECTURAL DECISIONS (`DECISIONS.md`)

## DEC-001: Compute & Autodiff Backend Selection
- **Context:** Sandbox has 2 vCPUs (Intel Xeon Ice Lake @ 2.60GHz with AVX-512 & VNNI), 3.8 GiB RAM, no GPU, and `download.pytorch.org/whl/cpu` is blocked by the sandbox TLS firewall while standard PyPI `torch` downloads >2.5 GB of unused CUDA packages.
- **Decision:** Use **JAX 0.10.2 + XLA CPU (`jaxlib 0.10.2`) + Optax 0.2.8 + Safetensors 0.8.0 + NumPy 2.4.6** for training, and support both **JAX JIT** and **standalone pure NumPy + Safetensors** for inference and quantization.
- **Rationale:** XLA compiles the entire Transformer forward, backward, and AdamW step into fused AVX-512 native machine code with minimal memory overhead, while `.safetensors` + pure NumPy guarantees the final `release/` artifact can run anywhere without heavy dependencies.
- **Status:** `VERIFIED`

## DEC-002: Model Architecture (`TinyMe-Reasoner` Decoder-Only Transformer)
- **Context:** Must satisfy `FINAL_MODEL_SIZE < 50 MB` (target 10–42 MB across FP32/FP16/INT8/INT4) while maximizing computational reasoning, code understanding, and algorithmic problem solving per byte.
- **Decision:** Use a modern **Llama-3 / Qwen-2.5 style Pre-Norm Decoder-Only Transformer** with:
  - Rotary Position Embeddings (**RoPE**, $\theta = 10000.0$)
  - **RMSNorm** (root-mean-square layer normalization)
  - **SwiGLU** gated feed-forward networks
  - **Grouped-Query / Multi-Head Causal Attention**
  - **Weight Tying** between input token embeddings and output LM head (`tie_word_embeddings = True`), saving `vocab_size * d_model` parameters (~2.0M–4.1M parameters) and improving representation alignment on compact corpora.
- **Status:** `VERIFIED`

## DEC-003: Dual-Path External Data Ingestion Strategy
- **Context:** Sandbox TLS filter permits `api.github.com`, `github.com`, and `pypi.org`, while blocking direct socket connections to `huggingface.co`. However, the environment provides `fetch_page` which reaches `datasets-server.huggingface.co` and `huggingface.co`.
- **Decision:**
  1. Ingest real Hugging Face dataset slices (`roneneldan/TinyStories` [CDLA-Sharing-1.0], `openai/gsm8k` [MIT], `google-research-datasets/mbpp` [CC-BY-4.0]) via `datasets-server.huggingface.co` into `datasets/raw/huggingface/` and record their exact URLs, licenses, and hashes in `DATA_PROVENANCE.json`.
  2. Ingest real GitHub repositories and datasets (`TheAlgorithms/Python` [MIT], `openai/human-eval` [MIT], `keon/algorithms` [MIT]) directly via `api.github.com`.
  3. Extract local Python 3.11 standard library algorithms & documentation (`PSF-2.0`).
  4. Generate execution-verified synthetic curriculum data across all 9 categories (`A` through `I`) using deterministic problem/solution/validator pipelines.
- **Status:** `VERIFIED`
