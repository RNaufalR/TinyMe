# TinyMe — FINAL REPORT (audit execution on branch `arena/01a0fa8e-tinyme`)

Specification: `TinyMeTask.md` (md5 `cf38c0c8b66134ea618b78fdd007abe6`, 29 sections).
Status vocabulary: **PLANNED · EXECUTED · VERIFIED · BLOCKED · FAILED · SUPERSEDED**.
Every number below was produced by a command in this repository; the file that
holds it is named next to it. Nothing is estimated, and no failed experiment is
hidden or deleted.

---

## 1. Headline result (read this first)

The engineering audit **passes**: the corrected training pipeline, data contract,
evaluation harness, sandbox, evidence engine, release packager and clean-room
release gate are all implemented, executed and verified with recorded artefacts.

The **capability** goal **fails**, and it is reported as `FAILED` rather than
papered over with a good loss curve:

| Model stage | Val ppl | Tool-suite `task_completion` | Verdict |
| :--- | --: | --: | :--- |
| `EXP-002-CORRECTED-NANO` (pretrain, 400 steps) | 122.02 | n/a | baseline trained, valid |
| `EXP-004-TOOL-SFT` (SFT, 320 steps) | 7.21 | 0.00 | rejected |
| `EXP-004-TOOL-SFT-V2` (SFT, 700 steps) | 5.41 | 0.00 | rejected |
| `EXP-004-TOOL-SFT-V3` (SFT, 600 steps) | **4.62** | **0.00** | **rejected — published as `release/` reference** |

The model produces syntactically valid tool calls that the runtime executes and
that return *real* computed values, then answers with memorised template text
instead of the computed result:

```
call   {'name': 'compute', 'arguments': {'expression': '69 + 45 - 19'}}
result {'ok': True, 'value': 95, 'exact': '95', 'method': 'python-ast-exact', 'citation': 'calc-51a69d1f'}
final  'Implement the specification and check the edge cases.'   <-- memorised, wrong
```

## 2. Root cause, measured

| Finding | Measurement | Artefact |
| :--- | :--- | :--- |
| One target sentence dominated the corpus | the code-repair thought string appeared **382 times with exactly 1 distinct value** = 5.7 % of all 6,652 target segments | target-segment census over `dataset_v3/train` |
| Fixed at the data level | thought pool of 6 phrasings per family; distinct thoughts 1 → 6 per family; `dataset_v3` **rev4** rebuilt (train 7,220 records, 718,077 active tokens, contamination PASS) | `data_sources/synthetic_v2.py`, `datasets/versions/dataset_v3/manifest.json` |
| Training data is ~50× too small | 718,077 active tokens = **1.40 %** of the Chinchilla-optimal 20×N budget (51,152,640) for 2,557,632 parameters | `docs/CAPACITY_ANALYSIS.md` |
| Capacity cannot fix it | Base (8,933,440 params) is compute-feasible (1,000.8 tok/s measured, 49.6 h per 20N pass) but the corpus caps it at the same deficit | `experiments/EXP-003-BASE-PILOT/summary.json` |

Loss was never accepted as evidence of capability (§22): all three SFT
iterations improved loss and remained at exactly 0.00 task completion.

## 3. What is verified (with the artefact that proves it)

| Audit section | Status | Evidence |
| :--- | :--- | :--- |
| §1 audit matrix | VERIFIED | `docs/AUDIT_MATRIX.md` (every row cites a runnable artefact) |
| §3 strict splits + contamination | VERIFIED | `dataset_v3` manifest: train 7,220 / val 1,422 / test 1,558 / challenge 60, exact + normalized + MinHash + code-shingle + template overlap all 0, `contamination_free: true` |
| §5 evidence engine | VERIFIED | `tests/test_evidence.py` — valid, unknown source, forged URL, forged id, unsupported quote, no evidence, multiple, mixed |
| §6 architecture (runtime/tools/search/retrieval/evidence/sandbox + providers) | VERIFIED | `src/agent/`, `src/tools/`, `src/sandbox/`; providers report `network=false`; no fake live search |
| §7 retrieval quality | VERIFIED | `tests/test_retrieval_quality.py` (12 tests, hit-rate gate, provenance, deterministic ranking) |
| §8 sandbox security matrix | VERIFIED | `docs/audit_evidence/sandbox_escape_suite.out.txt` — 18 cases, 18 PASS, 16 blocked-or-contained, label `namespace(net+mount+pid)` |
| §9 documented limits | VERIFIED | `docs/SANDBOX.md` — blocked / contained / allowed-by-design / not-tested / env-dependent |
| §11 sizes on serialized files | VERIFIED | `release/manifest.json`: fp32 10,233,776 B · fp16 5,118,480 · int8 2,600,976 · int4 1,451,272; directory 13 files / 19,688,665 B (18.78 MiB); `< 50 MB` true for all four |
| §12 release contract | VERIFIED | `release/` contains `model_{fp32,fp16,int8,int4}.safetensors`, `tokenizer.json`, `config.json`, `manifest.json`, `checksums.txt`, `evaluation_report.md`, `model_comparison.md`, `provenance.md`, `README.md`, `inference.py` |
| §13 packaging audit | VERIFIED | strict config load (no fallback), write-order (config before verify, inventory before README), checksums recomputed, Base-vs-Nano handled; `tests/test_release_contract.py` |
| §14 clean-environment gate | VERIFIED (portability) | `env -i PATH=/usr/bin:/bin HOME=/tmp PYTHONPATH=` in a directory containing **only** release files: `python3 inference.py --prompt ...` loads `[weights=float32 params=2,557,632]` and generates — no JAX, no Optax, no `src/`, no checkpoints, no dataset sources |
| §15 pinned deps / reproducibility | VERIFIED | `requirements.txt`, `docs/ENVIRONMENT_REPORT.md`, versions recorded inside `manifest.json: environment` |
| §16 test-suite completeness | VERIFIED | 33 test modules; `python3 -m pytest tests/ -q` → **273 passed** |
| §17 no false-pass tests | VERIFIED | audited skips (each gated on an artefact, none unconditional); source-string-only and self-fulfilling assertions removed |
| §18 CI | VERIFIED (file) | `.github/workflows/ci.yml`: lint → unit → integration → eval smoke → package smoke; no full training job |
| §19 synthetic generate→solve→verify→reject | VERIFIED | `data_sources/synthetic_v2.py`; counters requested/generated/verified/rejected/duplicate/malformed/accepted in the manifest; generator and verifier are independent code paths |
| §20 data scale/quality + exact token count | VERIFIED | `train_tokens_active` reconstructed from `labels != -100`, not from a config field |
| §24 inference audit | VERIFIED | `tests/test_inference.py`, `tests/test_kv_cache.py`: truncation, context window, BOS/EOS, stop set, top-p, temperature, seed, cache parity, long/empty/short prompts, max-context generation |
| §25 NumPy ≈ JAX, full ≈ cached | VERIFIED | max abs diff **1.53e-05**, argmax agreement 1.0 (`tests/test_forward_numpy_jax.py`, `tests/test_kv_cache.py`) |
| §26 independent RoPE verification | VERIFIED | `docs/audit_evidence/rope_pairing_probe_current.out.txt` — explicit NumPy rotation reference, worst-case **4.287e-07** vs tolerance 1e-5, norm preservation 1.471e-07; no self-reference |
| §27 resume verification | VERIFIED | `tests/test_checkpoint_resume.py`: params, optimizer, scheduler, global step, sampler, RNG, fingerprints, dataset/tokenizer compatibility; resumed == uninterrupted; no warmup restart |
| §28 gradient-accumulation equivalence | VERIFIED | `tests/test_grad_accumulation.py`: micro×K vs single large batch; params frozen until the update |
| §29 documentation sync | VERIFIED | `STATE.md`, `TODO.md`, `DECISIONS.md`, `EXPERIMENT_LOG.{md,jsonl}`, this report, `docs/AUDIT_MATRIX.md` |

### 2b. What the data intervention actually bought (test split, same 2,158 samples)

| Metric | V2 (rev3) | V3 (rev4) | Δ |
| :--- | --: | --: | --: |
| mean accuracy | 0.0571 | **0.1860** | **+0.1289** |
| logic | 0.2523 | **0.7385** | +0.4862 |
| tool_syntax | 0.2067 | **0.9867** | +0.7800 |
| tool_execution | 0.1467 | 0.2067 | +0.0600 |
| code_repair | 0.0000 | 0.0942 | +0.0942 |
| math | 0.0025 | 0.0000 | -0.0025 |
| tool_selection | 0.0200 | 0.0200 | 0.0000 |

Removing the repeated target sentence produced the largest single measured gain
of the whole audit **in-domain**, while the held-out-template §4 suite stayed at
0.00 `task_completion`. Both facts belong in the report: the intervention was
real, and it was not sufficient.

## 4. What failed

| Audit section | Status | The measurement that decides it |
| :--- | :--- | :--- |
| §2A tool SFT capability | **FAILED** | `experiments/EXP-004-TOOL-SFT-V3/evaluation_tools_best.json`: `task_completion` 0.00, `multi_step_success` 0.00, `error_recovery_success` 0.00, `citation_validity` 0.00 |
| §4 tool evaluation A–H | **FAILED (partial)** | working: `argument_validity` 1.00, `tool_syntax_validity` 0.84, `execution_success` 0.50, `tool_needed_accuracy` 0.30. Broken: multi-step, error recovery, grounded final answer, citations |
| §2C per-domain accuracy | **FAILED (partial)** | test split, 2,158 samples on `EXP-004-TOOL-SFT-V3`: mean accuracy **0.1860**; working: tool_syntax **0.9867**, logic **0.7385**, tool_execution 0.2067, code_repair 0.0942; broken: math 0.0000, code 0.0000, code_generation 0.0000, grounding 0.0000, instruction_following 0.0000, tool_arguments 0.0000 |
| challenge (held-out templates) | **FAILED** | mean accuracy 0.1000 (only `generalization` scores 10 %); math/logic/code_repair/instruction/tool_selection/grounding all 0.00 — while the *same* model scores logic 0.7385 on the test split. That gap is the §23 in-domain↑ / held-out-template↓ signature |
| §23 overfit curve | EXECUTED, no failure | V2 val_ppl 5.18 (400) → 5.29 (500) → 5.35 (600): mild overfit after step 400, `best` = step 400. V3: train 0.695 / val ppl 4.62 (no divergence), but test 0.1860 vs challenge 0.1000 shows the generalisation gap |
| §22 acceptance | **REJECTED** | all three SFT iterations rejected on capability, never on loss |

## 5. Quantisation outcome (§10)

| Variant | Bytes | Round-trip rel. RMSE | Max abs error | Tool syntax | Execution |
| :--- | --: | --: | --: | --: | --: |
| fp32 | 10,233,776 | 0 (reference) | 0 | 0.84 | 0.50 |
| fp16 | 5,118,480 | 2.12e-04 | 4.88e-04 | ≈ fp32 | ≈ fp32 |
| int8 | 2,600,976 | 6.44e-03 | 1.42e-03 | 0.80 | 0.45 |
| int4 | 1,451,272 | 9.67e-02 | 2.54e-02 | **0.40** | **0.00** |

Selected by measured functional regression: **fp32 is the reference**; int4 is
published as "smallest, measurably degraded". Full tables in
`docs/QUANTIZATION_REPORT.md`.

## 6. Latency / throughput (§10, §24)

| Measurement | Value |
| :--- | --: |
| NumPy engine, 32 greedy tokens | 409.9 tok/s |
| JAX engine, same workload, cold | 1.9 tok/s (per-call compilation) |
| Served generation, test split (1,491 calls) | 57,127 tokens / 316.8 s → **180.3 tok/s**, mean 0.2125 s, p50 0.042 s, p95 1.547 s |
| Served generation, challenge split (60 calls) | 3,419 tokens / 3.73 s → 916.6 tok/s, p50 0.062 s, p95 0.123 s |
| Training throughput | 663.9–849.4 tok/s (nano), 1,000.8 tok/s (base pilot) |

## 7. Honest statement of remaining limits

1. **Capability is not achieved.** A 2.56 M-parameter model trained on 0.72 M
   active tokens (1.4 % of budget) does not compose behaviours. The release is
   mechanically perfect and functionally weak; the two claims are never merged.
2. **No live network.** Search/fetch run against the shipped offline BM25 index
   and are labelled as such; providers report `network=false`. No live-search
   result is ever simulated.
3. **Sandbox is namespace-based, not a VM.** Residual risks (`/etc` shared,
   same-uid processes visible) are documented, not hidden.
4. **Base/Medium are not trained to budget.** Compute extrapolation says Base
   needs 49.6 h for a 20N pass on this box *if* the corpus existed; it does not.
5. **The §14 gate proves portability, not capability** — it passed while the
   model answered `-799` to `4837 * 962 + 71`.

## 8. Reproduce everything

```bash
bash scripts/run_audit_pipeline.sh EXP-004-TOOL-SFT-V3 best
python scripts/evaluate_tools.py --experiment EXP-004-TOOL-SFT-V3 --checkpoint best \
    --variants fp32,fp16,int8,int4 --max-new-tokens 96 --backend numpy
python scripts/evaluate.py --experiment EXP-004-TOOL-SFT-V3 --checkpoint best \
    --dataset dataset_v3 --split test --backend numpy --variants fp32
python scripts/evaluate.py --experiment EXP-004-TOOL-SFT-V3 --checkpoint best \
    --dataset dataset_v3 --split challenge --backend numpy --variants fp32,fp16,int8,int4
python scripts/package_model.py --experiment EXP-004-TOOL-SFT-V3 --checkpoint best --dataset dataset_v3
python -m pytest tests/ -q          # 273 passed
```
