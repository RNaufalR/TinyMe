# ARCHITECTURE SELECTION & DECISION RECORD (`ARCHITECTURE_DECISION.md`)

**Date:** 2026-10-01  
**Status:** `VERIFIED`  
**Hard Constraint:** `FINAL_MODEL_SIZE < 50 MB` (Target safety margin: 35–45 MB in full precision, <20 MB compressed)

---

## 1. Evaluation of Candidate Paradigms & Training Approaches

Before selecting the final architecture and training pipeline, all nine required approaches from §3 of `TINY_AI_TRAINING_LAB.md` were evaluated against our measured hardware constraints (2 vCPU Intel Xeon Ice Lake @ 2.60 GHz with AVX-512, 3.8 GiB RAM, no GPU, sandbox TLS filter blocking `huggingface.co` direct socket weights download):

| Approach | Feasibility in Sandbox | Pros | Cons / Constraints | Decision |
| :--- | :--- | :--- | :--- | :--- |
| **1. Tiny Transformer from Scratch** | **High** | Full control over vocabulary, layer geometry, RoPE, SwiGLU, and parameter budget (<50 MB guaranteed). | Must maximize data information density and curriculum structure. | **SELECTED (Core)** |
| **2. Compact Decoder-Only Language Model** | **High** | Unified autoregressive formulation (`<|system|>...<|user|>...<|thought|>...<|assistant|>...`) supports language, math, logic, code generation, code repair, and explanation in one model. | Autoregressive decoding requires KV-caching or fast matrix ops for low latency. | **SELECTED (Core)** |
| **3. Knowledge Distillation (Teacher → Student)** | **Medium (Sequence-Level)** | Running a multi-billion parameter teacher live in 3.8 GiB CPU RAM is infeasible; however, **sequence-level distillation** (training on verified structured solutions and reasoning traces from curated datasets and programmatic solvers) transfers teacher structure with zero runtime VRAM cost. | Live logit-level KL distillation from a 7B+ LLM is blocked by 3.8 GiB RAM & no GPU. | **SELECTED (Sequence-Level Distillation)** |
| **4. Continued Pretraining** | **High** | Enables multi-phase training: foundational language & syntax pretraining followed by domain-focused code & algorithmic continued pretraining. | Requires careful learning-rate warmup/decay transitions. | **SELECTED (Stage 1–6)** |
| **5. Supervised Fine-Tuning (SFT)** | **High** | Essential for aligning the pretrained model to structured problem solving (`Input → Transformation → Output`), code repair, and instruction following. | Can overfit if dataset lacks diversity; mitigated via dropout & weight decay. | **SELECTED (Stage 7–10)** |
| **6. Mixed Objective Training** | **High** | Combining standard causal language modeling on prose/code with targeted loss masking on structured `<|thought|>...<|answer|>` and `<|code|>` completion segments sharpens reasoning accuracy per parameter. | Slightly more complex data collator. | **SELECTED** |
| **7. Parameter-Efficient Training (LoRA / Adapters)** | **Medium** | Useful when adapting frozen base weights during continual learning (`dataset_v2`, `dataset_v3`) or targeted weakness repair without catastrophic forgetting. | For initial from-scratch training of a <10M param model, full-parameter updates are required. | **SELECTED (Supported for Continual Learning & Targeted Repair)** |
| **8. Quantization-Aware Approaches** | **High** | RMSNorm + bounded weight decay (`0.01`) + outlier-free SwiGLU scaling keeps weight distributions compact and resilient to INT8/INT4 quantization. | Full simulated QAT adds CPU overhead during pretraining; weight-regularized training + calibration is faster on CPU. | **SELECTED** |
| **9. Post-Training Quantization (PTQ)** | **High** | Per-channel symmetric INT8 and group-wise INT4 weight-only quantization reduce model footprint by 2x–8x with minimal perplexity or reasoning degradation. | INT4 can degrade sensitive arithmetic layers if group size is too large (solved via group-size 64 per-channel scaling). | **SELECTED (FP16, INT8, INT4)** |

---

## 2. Candidate Model Architectures Evaluated

We designed a modern **Pre-Norm Decoder-Only Transformer (`TinyMe-Reasoner`)** incorporating:
- **RMSNorm** ($\epsilon = 10^{-5}$) before attention and feed-forward sublayers
- **Rotary Position Embeddings (RoPE, $\theta = 10000.0$)** applied to Query and Key projections (0 additional parameters, strong length generalization for code/algorithms)
- **Causal Multi-Head Self-Attention** (`q_proj`, `k_proj`, `v_proj`, `o_proj`, bias-free)
- **SwiGLU Feed-Forward Network** ($\text{FFN}(x) = (\text{SiLU}(x W_{\text{gate}}) \odot x W_{\text{up}}) W_{\text{down}}$, bias-free)
- **Tied Input/Output Embeddings** (`wte` shared between token embedding lookup and final LM head projection)

### Architectural Candidates Comparison Table

| Metric / Specification | Candidate A: `TinyMe-Nano` (Ablation Baseline) | Candidate B: `TinyMe-Base` (**Selected Primary**) | Candidate C: `TinyMe-Medium` (Oversized FP32) |
| :--- | :--- | :--- | :--- |
| **Parameter Count** | `2,557,632` (2.56M) | **`8,933,440` (8.93M)** | `17,308,032` (17.31M) |
| **Embedding Size (`d_model`)** | `192` | **`320`** | `384` |
| **Number of Layers (`n_layers`)** | `4` | **`6`** | `8` |
| **Attention Heads (`n_heads`)** | `6` (`head_dim = 32`) | **`8` (`head_dim = 40`)** | `8` (`head_dim = 48`) |
| **Feed-Forward Dim (`d_ff`)** | `512` (SwiGLU) | **`896` (SwiGLU)** | `1024` (SwiGLU) |
| **Vocabulary Size (`vocab_size`)** | `4096` | **`4096`** | `8192` |
| **Context Length (`max_seq_len`)** | `256` (RoPE extensible to `512`) | **`256` (RoPE extensible to `512`)** | `256` |
| **Training Precision** | `FP32` (XLA AVX-512) | **`FP32` (XLA AVX-512)** | `FP32` |
| **Estimated Raw Size (FP32)** | `9.76 MiB` (`10.23 MB`) | **`34.08 MiB` (`35.73 MB`)** | `66.03 MiB` (`69.23 MB` — exceeds 50 MB in FP32) |
| **Serialized Size (FP16 `.safetensors`)** | `4.88 MiB` (`5.12 MB`) | **`17.04 MiB` (`17.87 MB`)** | `33.01 MiB` (`34.62 MB`) |
| **Quantized Size (INT8 `.safetensors`)** | `2.46 MiB` (`2.58 MB`) | **`8.56 MiB` (`8.98 MB`)** | `16.55 MiB` (`17.35 MB`) |
| **Quantized Size (INT4 `.safetensors`)** | `1.26 MiB` (`1.32 MB`) | **`4.36 MiB` (`4.57 MB`)** | `8.40 MiB` (`8.81 MB`) |
| **Tokenizer Footprint** | `~140 KB` (`tokenizer.json`) | **`~140 KB` (`tokenizer.json`)** | `~270 KB` (`tokenizer.json`) |
| **Inference RAM Requirement** | `< 60 MiB` | **`< 120 MiB`** | `< 220 MiB` |
| **CPU Training Speed (2 vCPU AVX-512)** | ~3,800 tok/sec | **~1,450 tok/sec** | ~620 tok/sec |

---

## 3. Final Selection Rationale (`TinyMe-Base` + `TinyMe-Nano` Ablation)

1. **Strict Compliance with `< 50 MB` and the 35–45 MB Safety Margin (§1)**:
   - `TinyMe-Base` has **8,933,440 parameters**.
   - Even in **uncompressed FP32**, its raw weight size is **35.73 MB (34.08 MiB)** — fitting directly inside the specification's recommended **35–45 MB** target window!
   - When exported in **FP16 (`17.87 MB`)**, **INT8 (`8.98 MB`)**, or **INT4 (`4.57 MB`)**, the entire deployable `release/` package (including `model_fp32.safetensors` OR compressed weights + `tokenizer.json` + inference code) is strictly under 50 MB.
2. **Cache-Resident Execution on Target CPU**:
   - The host Intel Xeon processor has a **54 MiB L3 cache**. Both `TinyMe-Base` FP32 (`34.08 MiB`) and FP16/INT8 (`17.04 MiB` / `8.56 MiB`) fit **entirely inside CPU L3 cache**, avoiding DRAM bandwidth bottlenecks during both XLA-compiled training and standalone NumPy inference.
3. **Vocabulary Efficiency (`vocab_size = 4096`)**:
   - A 32,000-token vocabulary at `d_model = 320` would consume `32000 * 320 * 4 = 40.96 MB` just for the embedding table, starving the reasoning layers.
   - A custom byte-level BPE tokenizer with `vocab_size = 4096` (plus structural special tokens `<|pad|>`, `<|bos|>`, `<|eos|>`, `<|unk|>`, `<|system|>`, `<|user|>`, `<|thought|>`, `<|assistant|>`, `<|code|>`, `<|endcode|>`) consumes only `1.31M` parameters (`5.24 MB` FP32), dedicating **85.3% of all parameters (`7.62M`) to the 6 deep Transformer reasoning layers** while maintaining zero unknown tokens (`0.0% UNK rate`) via byte fallback.
4. **Empirical Ablation Support**:
   - Both `TinyMe-Nano` (`2.56M` params) and `TinyMe-Base` (`8.93M` params), as well as tokenizer vocabulary sizes (`2048` vs `4096`), are implemented and empirically compared in our ablation suite (§21).
