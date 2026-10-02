# TinyMe — Audit matrix (§1 / §30 / §31)

Status vocabulary: **PLANNED · EXECUTED · VERIFIED · BLOCKED · FAILED · SUPERSEDED**.
A row is `VERIFIED` only when the implementation exists, has been *executed* on the
real artefacts, and an independent check confirms the measured result. Rows are
re-audited from the repository, not from memory.

Evidence column points at the file/command that produced the number, so a hostile
auditor can reproduce every claim.

| # | Requirement | Implementation | Execution evidence | Verification | Status |
| :-- | :--- | :--- | :--- | :--- | :--- |
| §1 | Audit matrix exists | this file | `docs/AUDIT_MATRIX.md` | every row cites a runnable artefact | VERIFIED |
| §2A | `EXP-004-TOOL-SFT` on corrected data | `scripts/train.py --stage sft` | 3 iterations: `EXP-004-TOOL-SFT` (320 steps), `-V2` (700), `-V3` (600) | `experiments/*/summary.json`, `evaluation_tools_best.json` | **EXECUTED, capability FAILED** — see §2A/§4 rows below |
| §2B | `EXP-003` Base feasibility | `scripts/train.py --arch base` | `experiments/EXP-003-BASE-PILOT/summary.json` (25 steps, 8,933,440 params, 1,000.8 tok/s) | `docs/CAPACITY_ANALYSIS.md` | EXECUTED — feasible in wall-clock, **BLOCKED by corpus size** (0.72 M active tokens = 1.4 % of 20N) |
| §2C | Per-domain evaluator (no collapsed score) | `src/evaluation/evaluator.py` | `evaluation_test.json`, `evaluation_challenge.json` | per-domain metrics + latency/throughput | VERIFIED |
| §3 | Strict splits + contamination | `src/data/splits.py`, `scripts/prepare_data_v2.py` | `datasets/versions/dataset_v3/manifest.json` (`train_validation_test_contamination: PASS`) | `tests/test_split_contamination.py` | VERIFIED |
| §4 | Tool-use evaluation A–H | `src/evaluation/tool_cases.py`, `scripts/evaluate_tools.py` | `experiments/EXP-004-TOOL-SFT-V3/evaluation_tools_best.{json,md}` | 25 independent cases, §5 citation rule | **FAILED** — task_completion 0.00, multi_step 0.00, error_recovery 0.00, citation_validity 0.00 |
| §5 | Evidence engine audit | `src/agent/evidence.py` | `tests/test_evidence.py` (10 tests) | forged-URL / forged-id / unsupported-quote cases | VERIFIED |
| §6 | Architecture: runtime + engines + providers | `src/agent/`, `src/tools/`, `src/sandbox/` | `tests/test_tool_registry.py`, `tests/test_tools.py` | no fake live search: providers report `network=false` | VERIFIED |
| §7 | Retrieval quality + provenance | `src/tools/retrieval.py` | `tests/test_retrieval_quality.py` (12 tests, hit-rate gate) | measured hit rate ≥ 0.875 | VERIFIED |
| §8 | Sandbox security matrix (≥14 cases) | `src/sandbox/*`, `scripts/audit_sandbox.py` | `docs/audit_evidence/sandbox_escape_suite.out.txt` (18 cases) | `tests/test_sandbox.py` (23 tests) | VERIFIED |
| §9 | Documented limits (honest labels) | `docs/SANDBOX.md` | escape-suite labels (`visible`, `allowed`, `contained`) | rows re-checked against live probe | VERIFIED |
| §10 | `EXP-005-QUANT` FP32/FP16/INT8/INT4 | `scripts/quantize.py`, `scripts/evaluate.py --variants` | `experiments/EXP-004-TOOL-SFT-V3/evaluation_{tools,challenge}.json` | `docs/QUANTIZATION_REPORT.md` | MEASURED — fp16/int8 ≈ fp32, **int4 degrades** (tool syntax 0.84 → 0.40) |
| §11 | Size measured on serialized files (MB & MiB) | `scripts/package_model.py` | `release/manifest.json`: fp32 10,233,776 B, fp16 5,118,480, int8 2,600,976, int4 1,451,272; dir 13 files / 19,688,665 B | `tests/test_release_contract.py`, `tests/test_package_size.py` (14 passed) | VERIFIED — `< 50 MB` for all four variants |
| §12 | Release file contract | `scripts/package_model.py` | `release/` — 13 files incl. all 4 variants, tokenizer, config, manifest, checksums, evaluation_report, model_comparison, provenance, README, inference.py | `tests/test_release_contract.py` | VERIFIED (contents) — the rendered `evaluation_report.md` was hollow and is now fixed, see defect table |
| §13 | Packaging audit (order, no fallback config) | `scripts/package_model.py` | strict `InferenceEngine(..., strict_config=True)` | regression tests in `tests/test_package_size.py` | VERIFIED |
| §14 | Clean-environment release gate | `release/inference.py` | manual probe `/tmp/cleanrel2` (`env -i`, empty `PYTHONPATH`, only release files): loads 2,557,632 params, generates | `tests/test_release_contract.py::test_clean_environment_release_gate` | VERIFIED for **portability**; the generated answer is wrong — portability and capability are reported separately |
| §15/§25 | Pinned deps + reproducibility | `requirements.txt` | versions recorded in `docs/ENVIRONMENT_REPORT.md` | `pip install -r requirements.txt` used by CI | VERIFIED |
| §16 | Test-suite completeness (24+ modules) | `tests/` (33 modules) | `pytest -q` → **274 passed** locally; **260 passed / 0 failed** in a fresh clone (CI layout) after the fixture fix below | no empty/existence-only module; the 5 remaining skips are artefact-gated with explicit reasons | VERIFIED |
| §17 | No false-pass tests | `tests/` | skip inventory: 5 skips, each gated by `requires_shards` with a printed reason (never unconditional); CI lint gate `\|\| true` **removed** so it can fail again | `pytest -rs` output in a fresh clone; `ruff check --select E9,F63,F7,F82,F811,F841` clean | VERIFIED |
| §18 | CI pipeline | `.github/workflows/ci.yml` | **run 37020181438 on commit `2ffc656`: all five jobs green** — lint 30 s, unit tests 2 m 43 s, integration 58 s, evaluation smoke 42 s, packaging + release smoke 28 s | remote evidence; three CI defects were found and fixed to get there (see below) | VERIFIED |
| §19 | Synthetic generate→solve→verify→reject | `data_sources/synthetic_v2.py` | `datasets/versions/dataset_v3/manifest.json` (`verified_samples`) | independent solver/verifier per family | VERIFIED |
| §20 | Data scale/quality + exact token count | `scripts/prepare_data_v2.py` | manifest `train_tokens_active` | reconstructed from `labels != -100` | VERIFIED |
| §21 | Curriculum A vs B pilots | `scripts/train.py --stage {pretrain,mixed}` | `experiments/CURRIC-A-STAGED/summary.json` (300 steps, ppl 101.65), `experiments/CURRIC-B-MIXED/summary.json` (300 steps, ppl 86.19), `evaluation_*.json` per split | per-domain + tool-suite comparison, 16 samples/domain | EXECUTED — **B (mixed) ≥ A (staged) on every capability measure**; both still 0.00 `task_completion`. Neither accepted |
| §22 | Iterative failure→data→retrain loop | `EXPERIMENT_LOG.jsonl` | v1 → V2 → V3, each launched from an independent-eval failure | 8-point record in `docs/TRAINING_REPORT.md` | EXECUTED — 3 iterations, never accepted on loss alone |
| §23 | Overfitting detection | `scripts/evaluate.py` + metrics | V3 val_ppl 4.62 (best, step 600); V2 showed 5.18 → 5.29 → 5.35 after step 400 (mild overfit) | `docs/TRAINING_REPORT.md` §6 | EXECUTED — plateau/overfit identified, `best` selected from it |
| §24 | Inference audit (edge cases) | `src/inference/engine.py` | `tests/test_inference.py`, `tests/test_kv_cache.py` | BOS/EOS/stop/temperature/top-p/cache | VERIFIED |
| §25 | NumPy≈JAX and full≈cached parity | `src/inference/engine.py` | `tests/test_forward_numpy_jax.py`, `tests/test_kv_cache.py` | recorded tolerances | VERIFIED |
| §26 | Independent RoPE verification | `tests/test_rope.py` | `docs/audit_evidence/rope_pairing_probe_current.out.txt` (worst |Δ| 4.287e-07, norm 1.471e-07); the pre-fix 4.076 probe is kept as `.out.txt.history` | explicit NumPy rotation reference written without repo code, multiple head dims (4/6/8) and positions (7/33), batch 2×3 | VERIFIED |
| §27 | Checkpoint resume verification | `src/training/checkpoint.py` | `tests/test_checkpoint_resume.py` | uninterrupted vs resumed comparison | VERIFIED |
| §28 | Gradient-accumulation equivalence | `src/training/trainer.py` | `tests/test_grad_accumulation.py` | micro×K vs single large batch | VERIFIED |
| §29 | Documentation synchronised | `STATE.md`, `TODO.md`, `DECISIONS.md`, `FINAL_REPORT.md` | this matrix + experiment logs | statuses match executable evidence | see below |

## Defects found and fixed during this audit

| Defect | Evidence it was real | Fix | Regression test |
| :--- | :--- | :--- | :--- |
| Training stream had **no `<|assistant|>` before `<|final|>`/`<|tool_call|>`** turns, while every runtime prompt ends with it | token dump of `build_sequence` on dataset_v3 rev1 | `build_sequence` inserts the turn opener (`src/data/sequence.py`) | `test_every_model_turn_opens_with_the_assistant_marker`, `test_prompt_is_an_exact_prefix_of_the_training_sequence` |
| `InferenceEngine.decode` **stripped protocol markers** (`skip_special_tokens=True`), so the agent runtime could never parse a generated tool call | `evaluate_tools.py` returned empty/`no_final_no_tool_call` for all 25 cases | `decode(..., skip_special_tokens=False)` by default | `tests/test_inference.py`, tool-eval suite |
| Generation **did not prepend `<|bos|>`**, shifting every RoPE position relative to training | side-by-side generation with/without BOS | `InferenceEngine._with_bos` | `tests/test_inference.py` |
| `AgentRuntime.solve` **crashed (`IndexError`)** when the model produced neither a tool call nor a final answer | live eval run crashed | explicit `no_final_no_tool_call` stop reason | `tests/test_agent_loop.py` |
| `render_prompt` leaked the **reference tool result** into the prompt (context that only exists *after* the model acts) | prompt dump vs training prefix | prompt stops at the first target segment | `test_prompt_is_an_exact_prefix_of_the_training_sequence` |
| Packed shards could be **stale** relative to the sequence builder (trained silently on an old format) | `test_shard_first_loading_matches_the_packed_builder` failed after the format fix | manifest records `shards_fingerprint`; `load_split` refuses mismatched shards | `tests/test_data_pipeline.py` |
| `package_model.py` verified the artifact **before** writing `config.json`, and inventoried files **before** writing `README.md` | release manifest without README; strict-config fallback | write-order fixed; strict load + param-count check | `tests/test_release_contract.py` |
| `prepare_data_v2.py` crashed on `tokenizer_saved.version` | build log traceback | `getattr(..., "version", TOKENIZER_VERSION)` | dataset build in CI |
| Tagged citation ids (`[S99]`) were accepted without evidence | `tests/test_evidence.py` | id/url/quote resolution in `EvidenceStore.verify()` | 10 evidence tests |

## Capability results (the numbers behind the FAILED rows)

Tool-use suite, 25 independent §4 cases (`scripts/evaluate_tools.py`):

| Experiment | Data rev | Steps | Val ppl | tool_needed | syntax | name | exec | multi_step | err_recovery | grounded_final | citation_validity | **task_completion** |
| :--- | :--- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| `EXP-004-TOOL-SFT` | rev1 | 320 | 7.21 | 0.20 | 0.20 | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| `EXP-004-TOOL-SFT-V2` | rev3 | 700 | 5.41 | 0.25 | 0.76 | 0.25 | 0.35 | 0.00 | 0.00 | 0.40 | 0.00 | 0.00 |
| `EXP-004-TOOL-SFT-V3` | rev4 | 600 | 4.62 | 0.30 | 0.84 | 0.30 | 0.50 | 0.00 | 0.00 | 0.30 | 0.00 | 0.00 |

Per-domain held-out evaluation (`scripts/evaluate.py --split test`, NumPy backend,
2,158 samples, `EXP-004-TOOL-SFT-V3`), with the previous iteration for reference:

| Domain | V3 (rev4) | V2 (rev3) |
| :--- | ---: | ---: |
| logic | **0.7385** | 0.2523 |
| tool_syntax | **0.9867** | 0.2067 |
| tool_execution | 0.2067 | 0.1467 |
| code_repair | 0.0942 | 0.0000 |
| tool_selection | 0.0200 | 0.0200 |
| math | 0.0000 | 0.0025 |
| code / code_generation / grounding / instruction_following / tool_arguments | 0.0000 | 0.0000 |
| language (ppl) | 812.40 | 620.78 |
| code_explanation (ppl) | 113.90 | 117.50 |
| **mean accuracy** | **0.1860** | 0.0571 |

Challenge split (held-out templates), same model: mean accuracy **0.1000** (only
`generalization` scores). Test 0.1860 vs challenge 0.1000 — with logic at 0.7385
in-domain and 0.00 on held-out templates — is the §23 generalisation gap, stated
explicitly rather than averaged away.

**Interpretation (conservative, evidence-based).** Training loss falls
(0.99 → 0.64 → 0.70 over the three iterations) and validation perplexity
improves (7.21 → 5.41 → 4.62), yet *task completion stays at exactly zero* in all
three. The generation transcripts show why: the model emits fluent, well-formed
protocol scaffolding and computes real values inside the tool runtime, then
answers with memorised template text instead of the computed value — e.g.

```
call   {'name': 'compute', 'arguments': {'expression': '69 + 45 - 19'}}
result {'value': 95, 'exact': '95', 'method': 'python-ast-exact', 'citation': 'calc-51a69d1f'}
final  'Implement the specification and check the edge cases.'   <-- memorised, wrong
```

This is a **capability failure at the data/capacity level**, not a defect in the
runtime: the corpus supplies 718,077 active target tokens = **1.4 %** of the
Chinchilla-optimal 20N budget for a 2.56 M-parameter model
(`docs/CAPACITY_ANALYSIS.md`). Loss-based acceptance was explicitly rejected
(§22): the honest verdict is **FAILED**, recorded with the measurements above
rather than hidden behind a good loss curve.

## Defects found during the capability iteration (this stretch)

| Defect | Evidence it was real | Fix | Regression test |
| :--- | :--- | :--- | :--- |
| **A single repeated target sentence dominated the corpus**: the code-repair thought string appeared 382 times and had exactly **1 distinct value** across all 382 records (5.7 % of all 6,652 target segments) | target-segment census over `dataset_v3/train`: `thought texts distinct: 1 of 382`; the model then emitted that sentence as its answer to arithmetic prompts | `data_sources/synthetic_v2.py`: `_THOUGHTS` pool (6 phrasings per family) + `rng.choice` for code_gen / code_repair / trace / arithmetic / logic | distinct-thought probe: 6/6/6 (was 1/1/1); `dataset_v3` rev4 rebuilt, contamination PASS |
| `scripts/evaluate.py` **overwrote** `result.environment`, discarding the generation latency/throughput statistics `evaluate_model` had measured | the first `evaluation_test.json` had `generation: null` even though the run took 660 s | assign with `.update()` instead of `=` | challenge run now records `calls/tokens_per_sec/latency_mean_s/latency_p50_s/latency_p95_s` |
| `scripts/evaluate.py::quantized_variants` **crashed** for every non-fp32 variant: `'dict' object has no attribute 'astype'` (transformer `blocks` is a list of dicts, not an array) | `AttributeError` traceback on `--variants fp32,fp16,int8,int4` | nested `_f32_list()` normaliser | all four variants now evaluate end-to-end |
| `scripts/evaluate_tools.py::_apply_variant` had the **same nested-block defect** | `AttributeError` on `--variants int8` | module-level recursive `_f32()` | fp16 / int8 / int4 tool-suite rows produced |
| `release/evaluation_report.md` was rendered with **0 samples scored / empty tables** because `package_model.py` expected a schema `evaluate.py` never wrote | generated report against `EXP-004-TOOL-SFT-V3` | `_normalise_results()` understands both the evaluator and tool-suite schemas | report now lists 176 scored samples and every domain |
| The packager's `fp32_bytes` mixed **serialized file size** with **raw tensor payload** | `tests/test_package_size.py` failed: 10,230,528 ≠ 10,233,776 | `fp32_bytes` = serialized artifact; `fp32_tensor_bytes` = payload | `tests/test_package_size.py` (14 passed) |
| `package_model.py::_write_readme` raised **`KeyError: 'files'`** (README quoted an inventory that did not exist yet) | packager crash traceback | `_inventory()` helper; inventory → README → recompute inventory | release assembly runs clean |

## CI defect found and fixed by auditing a fresh clone (this stretch)

| Defect | Evidence it was real | Fix | Verified by |
| :--- | :--- | :--- | :--- |
| Four `test_dataset_api.py` tests **failed in CI** (and in any fresh clone): they load packed shards, which are git-ignored build artefacts, so the runner had nothing to load | `git clone --depth 1` of the pushed branch + the exact CI command → 4 failed / 256 passed, job exit 2 | new session fixture `packed_dataset` runs the **real** builder (`scripts/prepare_data_v2.py --version ci_unit --scale 0.05 --seq-len 64`) and the tests assert against that artifact, skipping the build only when the recorded `shards_fingerprint` still matches disk | fresh-clone run of the exact CI command → **260 passed, 5 skipped, 0 failed** |
| The CI lint gate ended in `|| true`, so it could never fail the build | `.github/workflows/ci.yml` line 37 | gate now fails: `ruff check --select E9,F63,F7,F82,F811,F841 src scripts tests`; the three real findings it exposed (two dead locals, one unused binding) were removed | `ruff check` → *All checks passed*; the same command is what CI runs |

## Quality-filter defect found by CI (this stretch)

| Defect | Evidence it was real | Fix | Regression test |
| :--- | :--- | :--- | :--- |
| A binary payload could be labelled **`malformed_code_rejected` instead of `binary_content`**, because the syntax check ran before the binary check and its exception type is interpreter-dependent (CPython raises `ValueError` for a NUL byte on the audited box, `SyntaxError` on the CI runner) | CI check annotation on commit `cbdd655`: `FAILED tests/test_quality_filters.py::test_binary_and_minified_payloads_are_rejected - assert (not False and 'binary' in 'malformed_code_rejected')` — the same test passes locally | `filter_record` now checks `is_binary_like` **before** syntax validity: whether a payload is binary is a property of its bytes, not of the parser that happens to run next | `test_binary_reason_is_stable_regardless_of_parser_behaviour` (simulates the `SyntaxError` path and asserts `binary_content`) |

## CI brought to green (remote evidence)

The pipeline had never passed. Three real defects were found by auditing a fresh
clone and by running the workflow, each fixed with a regression test or an
explicit guarantee:

| # | Defect | How it was found | Fix |
| --: | :--- | :--- | :--- |
| 1 | CI installed only `requirements.txt`, the **runtime** set (numpy/tokenizers/safetensors) — the suite needs JAX/Optax, so the unit job died with 10 collection errors before running a test | fresh clone + the exact CI command in a clean venv | new `requirements-training.txt` (`-r requirements.txt` + `jax==0.10.2` + `optax==0.2.8`) installed by every job; the lint job still imports the runtime set alone, keeping the release-only claim honest |
| 2 | Four `test_dataset_api.py` tests loaded **git-ignored packed shards**, so any fresh clone failed them | `git clone --depth 1` of the pushed branch | session fixture builds a small dataset with the *real* builder (`--version ci_unit --scale 0.05 --seq-len 64`) and asserts against it |
| 3 | The lint gate ended in `\|\| true` (could never fail) | reading the workflow | gate removed; the three real findings it exposed were fixed |
| 4 | A binary payload was labelled `malformed_code_rejected` instead of `binary_content` because the syntax check ran first and the parser exception type is interpreter-dependent (`ValueError` locally, `SyntaxError` on the runner) | CI check annotation on `cbdd655` | `filter_record` classifies binary content first; regression test simulates the `SyntaxError` path |

Final state: **CI #37020181438 — lint ✓, unit ✓, integration ✓, evaluation
smoke ✓, packaging + release smoke ✓**.
