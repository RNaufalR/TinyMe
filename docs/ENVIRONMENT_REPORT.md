# ENVIRONMENT REPORT — FEASIBILITY AUDIT

**Audit Date:** 2026-10-01  
**Repository:** `RNaufalR/TinnyMe` (`/home/user/TinnyMe`)  
**Branch:** `arena/01a0f722-tinnyme`  
**Status:** `VERIFIED`

---

## 1. Hardware Capabilities

| Resource | Measured Value | Details |
| :--- | :--- | :--- |
| **CPU Model** | Intel(R) Xeon(R) Processor @ 2.60GHz | Family 6, Model 106 (Ice Lake Server), `x86_64` |
| **vCPUs / Cores** | 2 logical vCPUs (1 physical core, 2 threads/core) | 1 Socket, NUMA node 0 (`0,1`) |
| **SIMD / Vector Extensions** | AVX, AVX2, AVX-512 (`avx512f`, `avx512bw`, `avx512vl`, `avx512_vnni`, `avx512_vbmi2`) | Full AVX-512 + VNNI support for accelerated FP32/BF16/INT8 matrix operations |
| **Cache Hierarchy** | L1d: 48 KiB, L1i: 32 KiB, L2: 1.3 MiB, **L3: 54 MiB** | Large 54 MiB L3 cache fits an entire <50 MB model inside CPU cache |
| **RAM** | 3.8 GiB total (~3.6 GiB available) | No swap configured (`Swap: 0B`) |
| **Disk Storage** | 21 GiB total (~19 GiB available on `/dev/root`) | Shared memory `/dev/shm`: 2.0 GiB |
| **GPU / Accelerator** | **None** | `nvidia-smi: command not found`; CPU-only environment |

---

## 2. Software & Runtime Stack

| Component | Version / Status | Notes |
| :--- | :--- | :--- |
| **OS / Kernel** | Debian GNU/Linux 12 (bookworm), `x86_64` | KVM virtualization |
| **Python** | Python 3.11.2 | `/usr/bin/python3` |
| **C/C++ Toolchain** | `gcc`, `g++`, `make` installed | Available at `/usr/bin/gcc`, `/usr/bin/g++` |
| **Git & GitHub CLI** | `git`, `gh` installed and authenticated | `gh api rate_limit` verified: 5,000 req/hr |
| **ML Compute Engine** | `jax 0.10.2` + `jaxlib 0.10.2` (CPU XLA JIT) | Compiles forward/backward/optimizer steps into fused AVX-512 XLA kernels |
| **Optimizer Library** | `optax 0.2.8` | AdamW, cosine decay, warmup, gradient clipping, multi-step accumulation |
| **Numerical & ML Libs** | `numpy 2.4.6`, `scipy 1.17.1`, `scikit-learn 1.9.1` | Installed and verified |
| **Tokenizer Engine** | `tokenizers 0.23.2` (Rust BPE), `sentencepiece 0.2.2` | Fast Rust-backed BPE training and encoding |
| **Serialization** | `safetensors 0.8.0` (`safetensors.numpy`) | Safe, zero-copy weight serialization without pickle vulnerabilities |
| **Inference Runtime** | `jax`, `numpy`, `onnxruntime 1.30.0` | Pure NumPy + Safetensors standalone inference supported without heavy dependencies |
| **Testing Framework** | `pytest 9.1.1` | Automated test suite runner |

### Why JAX + XLA CPU + NumPy/Safetensors over PyPI `torch`?
- Standard `pypi.org` `torch 2.14.1` depends on >2.5 GB of `nvidia-*` CUDA wheels, while `download.pytorch.org/whl/cpu` is blocked by the sandbox TLS firewall (`SSL_ERROR_SYSCALL`).
- `jax 0.10.2` + `jaxlib 0.10.2` installs cleanly (<100 MB) and compiles transformer forward/backward passes via XLA CPU with AVX-512 vectorization.
- Furthermore, model weights are stored in `.safetensors` (FP32, FP16, INT8, INT4 packed) and our inference engine supports both **XLA-accelerated JAX** and **zero-dependency pure NumPy**, ensuring the `release/` package runs anywhere.

---

## 3. Network & External Data Access Audit

| Endpoint / Source | Sandbox `curl`/`python` Access | Agent `fetch_page` Access | Strategy |
| :--- | :--- | :--- | :--- |
| `pypi.org` / `files.pythonhosted.org` | **OPEN** (HTTP 200) | OPEN | Package installation via `pip` |
| `api.github.com` / `github.com` | **OPEN** (HTTP 200, 5000 req/hr) | OPEN | Direct repository & dataset ingestion (`TheAlgorithms/Python`, `google-research/mbpp`, `openai/grade-school-math`, `openai/human-eval`, etc.) |
| `huggingface.co` / `datasets-server.huggingface.co` | Blocked by sandbox TLS egress (`SSL_ERROR_SYSCALL`) | **OPEN** (Verified JSON row API & dataset cards) | Ingest verified HF dataset slices via `datasets-server.huggingface.co` into `datasets/raw/huggingface/` + fallback to cached snapshots & GitHub mirrors in `data_sources/huggingface_adapter.py` |
| Local Python 3.11 Standard Library & Docs | **LOCAL** (`/usr/lib/python3.11`) | N/A | Extract PSF-2.0 licensed Python stdlib algorithms, docstrings, and unit patterns |

---

## 4. Resource-Aware Training Strategy

Given **2 vCPUs (AVX-512)** and **3.8 GiB RAM**:
1. **Model Footprint**:
   - Target a compact decoder-only Transformer with RoPE, RMSNorm, SwiGLU, and GQA/MHA:
     - **Primary Architecture (`TinyCoder-Reasoner`)**: ~8.5M–12.5M parameters.
     - Raw FP32 size: ~34–48 MB (already <50 MB!).
     - FP16 serialized size: ~17–24 MB (comfortably inside the 35–45 MB or <25 MB target!).
     - INT8 weight-only quantized size: ~9–13 MB (plus ~250 KB tokenizer).
     - INT4 weight-only quantized size: ~5–7 MB.
2. **Sequence Length & Batching**:
   - Context window: `seq_len = 128` to `256` tokens during training (supports up to `512` at inference via RoPE).
   - Micro-batch size: `16`–`32` sequences with gradient accumulation (`1`–`4` steps) to keep peak RAM under 1.2 GiB (well below the 3.6 GiB available).
3. **High Information Density Corpus**:
   - Prioritize clean, verified, high-signal examples across all 9 required categories (Language, Logic, Mathematics, Algorithmic Reasoning, Programming, Code Repair, Code Generation, Code Explanation, Synthetic Curriculum).
   - Use XLA-compiled training loops (`@jax.jit`) for maximum tokens/second on Intel Xeon AVX-512 CPU.

## Dependency sets (audit §15, §18) — added during the CI audit

Two files are pinned on purpose and both are installed exactly as written:

| File | Contents | Who installs it |
| :--- | :--- | :--- |
| `requirements.txt` | **runtime**: `numpy`, `tokenizers`, `safetensors` | the release environment — `release/inference.py` must run with nothing else (§14) |
| `requirements-training.txt` | `-r requirements.txt` + `jax==0.10.2` + `optax==0.2.8` | CI and anyone who runs the training code or the test suite |

This split was made after auditing a **fresh clone**: installing only
`requirements.txt` gives 10 collection errors in the suite (`import jax` /
`import optax`), because the research stack genuinely needs them while the
released artefact genuinely does not. CI previously installed only the runtime
file, so its test job could never have passed; it now installs
`requirements-training.txt`, and the lint job additionally imports the runtime
set on its own to keep the release-only claim honest.
