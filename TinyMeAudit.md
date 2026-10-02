TinyMe — FULL CORRECTIVE AUDIT, MODEL REPAIR, TOOL RUNTIME, HARDENED SANDBOX & FRESH RETRAINING

You are operating on the existing TinyMe repository.

Repository:
"RNaufalR/TinyMe"

Target branch:
"arena/01a0f722-tinnyme"

PRIMARY OBJECTIVE:

Transform the existing TinyMe repository into a scientifically honest, reproducible, resource-efficient tiny language model system that:

1. has a correctly functioning Transformer training pipeline,
2. has a high-quality data pipeline,
3. has genuine held-out evaluation,
4. can learn structured reasoning/instruction behavior,
5. can learn reliable tool-use behavior,
6. has a provider-agnostic search/retrieval tool runtime,
7. has an actually hardened code-execution sandbox,
8. can be retrained from a corrected foundation,
9. remains under the strict final model-size requirement of UNDER 50 MB,
10. produces a genuinely usable release artifact,
11. contains measurable evidence for every important claim.

DO NOT merely describe how to do this.

INSPECT.
VERIFY.
IMPLEMENT.
TEST.
TRAIN.
MEASURE.
DEBUG.
RETRAIN.
COMPARE.
PACKAGE.

Do not fabricate results.

Do not simulate successful commands.

Do not claim a benchmark passed unless it actually passed.

Do not claim training completed unless it actually completed.

Do not claim a model improved unless the comparison was actually executed.

Do not silently reuse invalid historical evaluation results.

---

0. EXISTING PROJECT MUST BE TREATED AS AN IMPERFECT BASELINE

The current EXP-001 artifact is historical and must NOT automatically be treated as the trusted scientific baseline.

Known evidence already found in the repository includes:

- current nano model: ~2.56M parameters,
- historical FP32 model: ~9.76 MiB,
- dataset version contains approximately 514 training samples and 25 evaluation samples,
- tokenizer has 4096 vocabulary entries,
- historical experiment reports validation perplexity around 2.45,
- however the training script currently obtains evaluation tensors from "train.jsonl",
- therefore the historical validation number is potentially contaminated and must not be used as a genuine held-out benchmark.

Treat EXP-001 as:

"HISTORICAL / INVALID-AS-HELDOUT-BASELINE"

not as:

"TRUSTED QUALITY BASELINE".

Preserve the historical artifacts.

Do NOT overwrite them.

Create a new experiment identity for corrected training, for example:

"EXP-002-CORRECTED"

or another clearly versioned identifier.

Never silently resume EXP-001 after changing the data pipeline, tokenizer, model implementation, loss function, or evaluation system.

Corrected training MUST start from fresh model initialization unless an explicit controlled ablation proves that reusing a checkpoint is scientifically valid.

---

1. FIRST: FULL REPOSITORY AUDIT

Before changing anything:

inspect the complete repository.

Read at minimum:

- README.md
- STATE.md
- TODO.md
- DECISIONS.md
- EXECUTION_PLAN.md
- TINY_AI_TRAINING_LAB.md
- DATA_PROVENANCE.json
- docs/DATA_QUALITY_REPORT.md
- docs/TOKENIZER_METRICS.json
- configs/*
- src/model/*
- src/training/*
- src/data/*
- src/tokenizer/*
- src/evaluation/*
- src/inference/*
- src/quantization/*
- scripts/*
- data_sources/*
- datasets/*
- release/*
- experiments/*
- checkpoints/*
- tests/*

Do not blindly trust documentation.

Compare documentation against actual implementation.

Create or update:

"docs/CORRECTIVE_AUDIT.md"

Classify every discovered issue:

- P0 = correctness blocker
- P1 = quality/reproducibility blocker
- P2 = optimization
- P3 = optional research

Every issue must have:

- evidence,
- affected file,
- impact,
- proposed fix,
- verification method,
- final status.

---

2. P0 — FIX THE DATA PREPROCESSING CORRUPTION

This is one of the most critical existing defects.

The current text normalization collapses repeated whitespace globally.

That is unacceptable for source code.

Python indentation is semantic.

Do NOT perform prose-style whitespace collapsing on code.

Implement content-aware normalization.

For prose:

- normalize Unicode safely,
- normalize line endings,
- remove invalid control characters,
- optionally collapse unnecessary spaces.

For source code:

- preserve indentation,
- preserve meaningful repeated spaces,
- preserve newlines,
- preserve tabs when semantically meaningful,
- remove only unsafe/control artifacts that do not change syntax.

For Markdown containing code fences:

preserve code fence content exactly.

For structured TinyMe control tokens:

preserve:

"<|system|>"
"<|user|>"
"<|assistant|>"
"<|thought|>"
"<|answer|>"
"<|tool_call|>"
"<|tool_result|>"
"<|final|>"
"<|code|>"
"<|endcode|>"

as structural tokens.

Do not normalize away the grammar.

Add regression tests proving:

- valid Python remains valid Python,
- indentation survives preprocessing,
- code fences survive,
- whitespace-sensitive examples survive,
- Unicode survives,
- prose normalization still works.

Run AST validation after preprocessing.

For code categories:

"syntax_valid == false"

must normally be a hard rejection, not merely a small quality penalty.

Do not train the model on malformed code unless the record is intentionally a debugging example whose malformed state is part of the task.

For repair tasks, distinguish:

"BUGGY_INPUT"

from:

"TARGET_CORRECT_CODE"

and only use the correct target as desired output.

---

3. P0 — REBUILD THE DATA CONTRACT

Do not use a single opaque "text" string as the only representation internally.

Introduce a structured record representation.

A training example should conceptually contain:

record_id
category
task_type
source
source_id
license
language
template_id
split
messages/segments
target_segments
loss_mask
verification metadata
tests
answer
provenance
quality

Support explicit segments:

SYSTEM
USER
ASSISTANT
THOUGHT
TOOL_CALL
TOOL_RESULT
FINAL
CODE

The raw serialized text may still exist for portability.

However, internally the trainer must know which tokens are:

- conditioning/context tokens,
- target tokens,
- ignored tokens,
- tool-result tokens,
- user tokens,
- assistant tokens.

This is mandatory for reliable instruction/tool-use training.

---

4. P0 — FIX THE TRAINING OBJECTIVE

The current training loop applies next-token loss indiscriminately.

Replace this with task-aware loss masking.

For ordinary language-model pretraining:

loss may cover the normal causal stream.

For instruction examples:

system/user/input tokens should normally be context only.

assistant output should normally be the prediction target.

For reasoning examples:

decide explicitly whether "<|thought|>" belongs to the target.

For tool-use examples:

assistant tool-call tokens:

TARGET

tool result tokens:

CONTEXT ONLY

assistant final response:

TARGET

Example:

<|system|>
...
<|user|>
Search for ...
<|assistant|>
<|tool_call|>
{"name":"search","arguments":{"query":"..."}}
<|tool_result|>
...
<|assistant|>
Final grounded response...

The model should learn to predict:

tool_call

and:

final response

It should NOT be trained to reproduce the tool result itself.

Implement a "loss_mask" tensor aligned exactly with labels.

Masked positions must contribute zero loss.

Implement tests proving that:

- ignored tokens produce zero contribution,
- target tokens contribute normally,
- padding contributes zero loss,
- tool-result content does not become a generation target.

---

5. P0 — FIX PADDING

The current trainer pads examples and then computes cross-entropy across the padded positions.

Implement correct padding masking.

Targets equal to "pad_id" must not contribute to loss or evaluation metrics.

Never train the model to predict padding.

Report:

- active target tokens,
- masked tokens,
- padding ratio.

Do not report padded sequence length as equivalent to useful training tokens.

---

6. P0 — FIX TRAIN/EVAL/TEST SEPARATION

Create a genuine three-way split:

TRAIN
VALIDATION
TEST

The test set must remain untouched by training and model-selection decisions.

The validation set is for iteration.

The test set is for final measurement.

For synthetic data:

do NOT perform only random row-level splits.

Use:

- template-family split,
- generator split,
- parameter-range separation where appropriate,
- task-family separation where appropriate.

For example:

training may see many arithmetic templates,

validation may see new parameter combinations,

test may contain unseen template structures.

For code:

hold out entire tasks/functions/templates where possible.

For external datasets:

never accidentally mix benchmark test data into training.

Track:

- source ID,
- row ID,
- template ID,
- split.

Implement automated contamination checks:

- exact hash,
- normalized hash,
- similarity,
- code similarity,
- template overlap.

The final report must explicitly state:

TRAIN/VALIDATION/TEST CONTAMINATION:
PASS / FAIL

---

7. P0 — FIX THE TRAINING SCRIPT

The training script must load:

"train.jsonl"

for training.

It must load:

"eval.jsonl"

for validation.

It must load:

"test.jsonl"

or an equivalent independent test artifact for final evaluation.

Never use:

train.jsonl[:32]

as validation.

Create a clean dataset API:

load_split(version, split, ...)

rather than a function that always assumes "train.jsonl".

Add tests for split loading.

---

8. P0 — FIX TOKENIZER LEAKAGE

Current tokenizer training occurs before the train/eval separation.

Correct the pipeline.

Procedure:

1. ingest raw sources,
2. validate/license/filter,
3. establish train/eval/test split,
4. ensure contamination-free split,
5. train tokenizer ONLY on training text,
6. freeze tokenizer,
7. tokenize train/validation/test separately.

Never train tokenizer vocabulary on test content.

Generate:

"docs/TOKENIZER_REPORT.md"

and include:

- algorithm,
- vocab size,
- file size,
- train-only training corpus,
- tokens/sample,
- code fertility,
- prose fertility,
- math fertility,
- tool-call JSON fertility,
- special token behavior,
- tokenizer hash.

The model's vocabulary size must equal the tokenizer's vocabulary size.

Add an explicit startup assertion.

---

9. P0 — FIX SEQUENCE CONSTRUCTION

The current project has inconsistent paths:

- pipeline tokenization,
- shards,
- training-time re-tokenization,
- fixed sequence truncation.

Unify them.

There must be exactly one authoritative sequence-building mechanism.

Do NOT preprocess 4x "seq_len" and then silently truncate to "seq_len" during training without documentation.

Support:

- dynamic examples,
- fixed-length packing,
- target-aware truncation,
- optional sliding windows for long documents.

For instruction/tool records:

NEVER blindly truncate the beginning of a sample if doing so removes the answer or tool call.

Prefer:

- preserve prompt + target,
- crop irrelevant context,
- use sliding windows where appropriate,
- reject samples that cannot preserve a meaningful target.

For long-context training:

train explicitly at the context length that you intend to support.

Do not claim 512-token context competence when almost all training occurred at 128 tokens.

Primary target:

"seq_len = 256"

Secondary:

"seq_len = 512" if environment permits.

Benchmark both.

---

10. P0 — FIX GRADIENT ACCUMULATION

Current implementation is not true gradient accumulation.

Correct behavior:

microbatch 1 → gradients
microbatch 2 → gradients
microbatch 3 → gradients
...
average/sum gradients
↓
ONE optimizer update

Parameters MUST remain unchanged across microbatches inside one accumulation step.

Optimizer state MUST update only once per accumulation step.

Use JAX primitives such as:

- "jax.lax.scan"
- "vmap"
- suitable rematerialization/checkpointing

when useful for memory efficiency.

Add a regression test that verifies:

"grad_accum_steps = N"

is numerically equivalent, within tolerance, to computing the combined batch gradient and applying one optimizer update.

---

11. P0 — FIX BATCH SAMPLING

Current sampling has multiple problems.

Replace it with a deterministic epoch/sampler system.

Requirements:

- no accidental exclusion of the tail of the dataset,
- no silent replacement sampling unless explicitly intended,
- reproducible shuffling,
- explicit epochs,
- deterministic batch ordering,
- sampler state persistence.

Persist:

- epoch,
- batch position,
- RNG state or deterministic seed derivation,
- dataset fingerprint.

Never restart from the first batch after resume unless explicitly configured.

---

12. P0 — FIX TRUE CHECKPOINT/RESUME

Checkpoint MUST include:

- model parameters,
- optimizer state,
- scheduler state,
- global step,
- epoch,
- batch/sampler position,
- RNG state or deterministic RNG derivation state,
- best validation metric,
- complete training configuration,
- model configuration,
- tokenizer version/hash,
- dataset version/hash,
- git commit,
- environment information.

Do NOT use unsafe pickle for core checkpoint serialization.

Use safe serialized tensor/state representations.

Add:

"checkpoint integrity verification"

and:

"checkpoint compatibility verification".

A checkpoint from another:

- architecture,
- tokenizer,
- dataset,
- experiment,

must not be loaded silently.

Use configuration/data/model fingerprints.

Implement:

save_checkpoint()
load_checkpoint()
verify_checkpoint()

with tests.

Checkpoint writes should be atomic.

A partial/corrupt checkpoint must not destroy the previous valid checkpoint.

---

13. P0 — FIX SCHEDULER/OPTIMIZER RESUME

The learning-rate schedule must continue from the real optimizer/training step.

Do not recreate AdamW with empty state during resume.

Verify:

LR_before_checkpoint
=
LR_after_resume

at the same logical training step.

Add a test.

---

14. P0 — FIX RANDOMNESS / REPRODUCIBILITY

Synchronize and document:

- Python random,
- NumPy,
- JAX PRNG,
- data shuffling,
- synthetic generator seeds.

Where bitwise determinism is impossible because of backend differences, document the allowed tolerance.

Every experiment should have:

seed
dataset fingerprint
tokenizer fingerprint
model config
training config
git commit
dependency versions
hardware/backend

---

15. P0 — FIX RoPE

Correct the current RoPE implementation.

The current frequency construction duplicates frequency halves using concatenation while rotation is implemented over even/odd pairs.

Implement the mathematically correct pairing representation.

Then verify:

1. training forward pass,
2. NumPy inference,
3. JAX inference,
4. KV-cache inference

all use the SAME RoPE semantics.

Create tests for:

full_forward(tokens)
vs
cached_forward(tokens)

The final logits should match within a documented numerical tolerance.

Do NOT retrain before this test passes.

---

16. P0 — FIX NUMERICAL STABILITY

Add:

- finite-loss checks,
- NaN/Inf gradient detection,
- gradient norm logging,
- activation sanity checks where useful,
- stable softmax,
- stable normalization,
- overflow/underflow checks.

When a training run encounters NaN/Inf:

record the exact step/configuration.

Do not continue blindly.

---

17. P0 — ACTUALLY HONOR compute_dtype

"compute_dtype" must affect real computation.

Support what the available backend reliably supports.

Keep numerically sensitive operations in appropriate precision where necessary.

Do not advertise bfloat16/float16 support merely because YAML contains the string.

Log the actual dtype of:

- embeddings,
- linear weights,
- activations,
- logits,
- optimizer calculations.

---

18. MODEL ARCHITECTURE SELECTION

Do not automatically remain on nano.

Calculate the actual deployed footprint.

Candidate:

Nano

~2.56M parameters.

Base

~8.93M parameters.

Base is approximately 34 MiB FP32 by analytic parameter size and therefore may still fit under the 50 MB model constraint before other packaging considerations.

Medium may exceed 50 MB in FP32 and may require quantization.

Do not choose by opinion.

Run a resource-feasibility comparison.

At minimum:

- corrected nano baseline,
- base candidate when environment allows.

Compare:

- validation quality,
- test quality,
- tool-call validity,
- reasoning accuracy,
- code correctness,
- context behavior,
- inference latency,
- memory usage,
- actual artifact size.

Choose based on measured evidence.

If the environment cannot fully train Base:

run a short feasibility pilot and document the limitation.

Do not fake completion.

---

19. REBUILD THE DATA PIPELINE FOR HIGH INFORMATION DENSITY

The current dataset is extremely small.

Do not simply duplicate the same samples to inflate token counts.

Increase useful diversity.

Prioritize:

- high-quality technical prose,
- programming,
- algorithms,
- mathematics,
- logic,
- code repair,
- code generation,
- code explanation,
- instruction following,
- tool-use examples,
- evidence/citation behavior.

For every source track:

- license,
- source URL,
- retrieval date,
- source ID,
- row ID,
- language,
- category,
- verification,
- provenance.

Do not blindly ingest arbitrary internet content.

---

20. FIX CODE DATA QUALITY

Code should be handled separately from prose.

For source code:

- preserve formatting,
- language detection,
- language-specific syntax checks,
- secret scanning,
- PII scanning,
- malware/suspicious-pattern scanning,
- generated-file detection,
- vendored dependency removal,
- minified-code rejection,
- huge-file rejection.

Do not run Python AST parsing on Rust/Go/JavaScript as if it were Python.

Either:

1. implement language-aware validators,

or:

2. restrict training/evaluation to supported languages and declare it explicitly.

For TinyMe vNext, Python may be the primary execution language.

That is acceptable if explicitly documented.

---

21. FIX DEDUPLICATION

Implement real multi-stage deduplication:

1. exact hash,
2. normalized hash,
3. document similarity,
4. code similarity.

If claiming LSH/MinHash:

actually implement the index.

Do not leave placeholder loops.

Support scalable indexes rather than comparing every item against thousands of recent records.

Also report:

- duplicates before split,
- train/eval duplicates,
- train/test duplicates,
- template duplicates,
- code duplicates.

---

22. FIX SHARDING

The current shard writer repeatedly loads and rewrites the current ".npy".

Replace this with efficient writing.

Possible approaches:

- preallocation,
- memory mapping,
- temporary chunk files + concatenate,
- appendable binary format.

Do not perform repeated "np.vstack()" for every sample.

Shards must be actually consumed by training OR removed from the critical path.

Do not maintain fake infrastructure that training does not use.

---

23. CREATE A TWO-STAGE TRAINING STRATEGY

Use a scientifically useful training schedule.

STAGE A — DOMAIN/LANGUAGE PRETRAINING

Train on:

- clean prose,
- programming,
- algorithms,
- mathematics,
- logic,
- high-value technical text.

Use packed sequences where appropriate.

The objective is ordinary causal language modeling.

STAGE B — INSTRUCTION / REASONING / TOOL-USE TRAINING

Train on structured examples using loss masks.

Target:

- instruction following,
- concise reasoning,
- code generation,
- code repair,
- structured outputs,
- tool selection,
- tool argument generation,
- search workflows,
- evidence-grounded answering.

Do not blindly imitate excessively verbose hidden reasoning.

Optimize for observable task completion.

---

24. ADD EXPLICIT TOOL-USE TRAINING

This is mandatory for the requested search-engine capability.

The model itself should remain small.

The external runtime should perform search and computation.

Train the model to:

understand intent
→ decide whether a tool is needed
→ choose tool
→ construct arguments
→ inspect result
→ decide whether another tool call is needed
→ synthesize final answer

Create training trajectories such as:

USER
→ ASSISTANT TOOL CALL
→ TOOL RESULT
→ ASSISTANT FINAL

and:

USER
→ TOOL CALL
→ TOOL RESULT
→ TOOL CALL
→ TOOL RESULT
→ FINAL

also:

USER
→ TOOL CALL
→ ERROR
→ RETRY/CORRECT TOOL CALL
→ RESULT
→ FINAL

Teach cases where:

- no search is needed,
- search is required,
- calculator is required,
- code execution is required,
- source verification is required,
- multiple sources must be cross-checked.

---

25. DEFINE A STRICT TOOL PROTOCOL

Use explicit special tokens such as:

<|tool_call|>
<|tool_result|>
<|final|>
<|endtool_call|>
<|endtool_result|>

Only add the minimum required number of tokens.

Define a strict structured format.

Example:

{
  "name": "search",
  "arguments": {
    "query": "..."
  }
}

Tool arguments MUST be machine-parseable.

Do not rely on free-form natural-language tool selection.

Runtime must validate model-generated tool calls against a tool schema.

Invalid calls must not execute.

---

26. TOOL RUNTIME ARCHITECTURE

Create:

src/
├── agent/
│   ├── router.py
│   ├── planner.py
│   ├── executor.py
│   ├── tool_registry.py
│   ├── protocol.py
│   └── evidence.py
│
├── tools/
│   ├── search/
│   │   ├── web.py
│   │   ├── news.py
│   │   ├── image.py
│   │   ├── academic.py
│   │   ├── code.py
│   │   ├── docs.py
│   │   └── package.py
│   │
│   ├── retrieval/
│   │   ├── open_url.py
│   │   ├── html_extract.py
│   │   ├── pdf_extract.py
│   │   ├── document_extract.py
│   │   └── find.py
│   │
│   └── compute/
│       ├── calculator.py
│       ├── python.py
│       ├── json.py
│       ├── csv.py
│       └── statistics.py
│
└── sandbox/
    ├── runner.py
    ├── policy.py
    ├── workspace.py
    ├── limits.py
    └── isolation.py

Keep the model-facing surface small.

Prefer approximately:

search
fetch
compute
code
files

with internal routing.

Do not expose dozens of raw provider-specific tools directly to the tiny model.

---

27. SEARCH TOOL REQUIREMENTS

Implement provider-agnostic interfaces.

Search categories:

- web,
- news,
- image,
- academic,
- code/GitHub,
- documentation,
- package/Hugging Face.

The runtime should support configurable providers.

Do not hard-code one provider in the architecture.

Provider credentials, where needed, must come from environment/configuration.

Never commit API keys.

Never pretend that an unavailable provider worked.

Every result should preserve source metadata.

---

28. RETRIEVAL REQUIREMENTS

Implement:

- URL opening,
- HTML extraction,
- PDF extraction,
- document extraction,
- text search/find,
- link following,
- content truncation,
- content hashing,
- duplicate-result removal.

Retrieval output should be normalized before being passed back to the model.

Do not dump uncontrolled full web pages into the tiny model context.

Use:

search
→ rank
→ select
→ fetch
→ extract
→ compress
→ pass evidence

---

29. SEARCH RESULT QUALITY

Each source should contain metadata conceptually like:

{
  "source_id": "...",
  "url": "...",
  "title": "...",
  "source_type": "...",
  "retrieved_at": "...",
  "published_at": "...",
  "query": "...",
  "rank": 1,
  "content_hash": "...",
  "excerpt": "..."
}

Do not allow the model to fabricate source IDs.

Citation identifiers should come from runtime-owned metadata.

---

30. EVIDENCE ENGINE

Implement an evidence layer.

Every retrieved claim should be traceable to:

claim
→ evidence
→ source

Support:

- source IDs,
- excerpts,
- URLs,
- timestamps,
- content hashes,
- confidence/quality metadata.

The final answer generator should be able to cite real retrieved sources.

Do not allow invented citations.

---

31. TOOL EXECUTION CONTROL

The orchestrator must support:

- maximum number of tool steps,
- per-tool timeout,
- total agent timeout,
- retry policy,
- loop detection,
- duplicate-call suppression,
- result-size limits,
- context budgeting,
- error propagation.

A tiny model must not be allowed to call tools indefinitely.

---

32. HARDEN THE SANDBOX

The existing evaluator subprocess is NOT sufficient as a production sandbox.

"cwd="/tmp"" is not a security boundary.

Separate:

NETWORK-ENABLED RETRIEVAL

from:

NETWORK-DISABLED CODE EXECUTION.

Sandbox requirements:

task_workspace/
├── input/
├── output/
└── temp/

Apply:

- CPU limit,
- memory limit,
- wall-time limit,
- output-size limit,
- file-size limit,
- file-count limit,
- process-count limit,
- restricted filesystem,
- no access to host project files,
- no inherited secrets,
- no credentials,
- no unnecessary environment variables,
- network disabled by default,
- restricted privileges,
- restricted syscalls/capabilities where the platform supports them,
- process-tree cleanup after timeout.

Do not rely solely on:

subprocess.run(..., cwd="/tmp")

Also fix the output capture problem.

The current implementation captures subprocess output before truncating it.

Implement bounded output handling so an untrusted process cannot consume unlimited memory simply by writing huge output.

Where possible, use OS-level resource limits.

Detect the available isolation mechanism automatically.

Possible strategies may include suitable OS/container sandboxing when available.

If true strong isolation is unavailable:

- implement the strongest available restrictions,
- label the environment honestly,
- do NOT describe it as fully secure.

Create sandbox escape regression tests.

---

33. SANDBOX TEST MATRIX

Test at minimum:

1. normal valid Python,
2. infinite loop,
3. huge output,
4. excessive memory allocation,
5. large file creation,
6. many file creation attempts,
7. process spawning,
8. network access,
9. environment-variable access,
10. host filesystem access,
11. parent/project directory access,
12. secret discovery attempts,
13. malformed code,
14. timeout cleanup.

Every test must record actual result.

---

34. FIX INFERENCE

The inference engine currently has multiple issues.

Fix:

- RNG initialization when seed is omitted,
- top-p threshold handling,
- EOS termination,
- deterministic generation,
- prompt truncation,
- sequence window handling,
- repetition penalty behavior,
- model/tokenizer compatibility.

Use the existing KV-cache implementation properly.

Do not repeatedly run the entire context for every generated token when KV caching is available.

Implement:

prefill
→ cached decode
→ cached decode
→ ...

Verify:

full forward logits
≈
cached forward logits

within documented tolerance.

---

35. JAX / NUMPY PARITY

Test:

JAX forward
≈
NumPy forward

for the same model/tokens.

Also test:

JAX forward
≈
cached JAX forward

and:

NumPy forward
≈
cached NumPy/reference implementation

at least on representative cases.

Do not release a model before these pass.

---

36. IMPROVE EVALUATION

Create independent datasets for:

Language

Perplexity on held-out text.

Logic

Accuracy.

Mathematics

Accuracy with robust answer extraction.

Algorithms

Accuracy.

Code generation

Executable test pass rate.

Code repair

Executable test pass rate.

Code explanation

Semantic/reference evaluation where possible.

Instruction following

Task completion rate.

Tool selection

Correct tool selected.

Tool arguments

Valid and correct JSON arguments.

Tool-call syntax

Valid protocol rate.

Search grounding

Percentage of claims linked to retrieved evidence.

Citation validity

Citations must correspond to actual retrieved sources.

Tool error recovery

Successful recovery after failed tool call.

Generalization

Unseen task templates.

Do NOT collapse all of this into one simplistic overall score.

Report domain-specific metrics.

---

37. EVALUATION MUST BE LARGER

The current 25 evaluation samples are insufficient for serious conclusions.

Expand independent evaluation sets.

Use:

- generated validation sets,
- independently generated test sets,
- held-out task templates,
- small manually curated challenge sets,
- tool-runtime synthetic tests.

Do not inflate test results by tuning directly on the test set.

Maintain:

train
validation
test
challenge

where appropriate.

---

38. EVALUATE QUANTIZATION FUNCTIONALLY

Do not select INT4 simply because it is smaller.

For every release variant:

- FP32,
- FP16,
- INT8,
- INT4,

measure:

- actual file size,
- model load success,
- generation success,
- validation quality,
- test quality,
- code correctness,
- reasoning accuracy,
- tool-call validity,
- latency,
- memory.

Select the smallest variant that does not produce an unacceptable functional regression.

---

39. ACTUAL PACKAGE SIZE

Define:

MODEL_WEIGHT_LIMIT = <50 MB

Measure actual bytes.

Also separately report:

weights
tokenizer
config
metadata
runtime package

Do not claim “under 50 MB” using only an analytical parameter estimate.

Use actual serialized artifact size.

The final release must contain its own size report.

---

40. RELEASE PACKAGE

Create a reproducible release directory containing at minimum:

release/
├── model.*
├── tokenizer.json
├── config.json
├── manifest.json
├── checksums.txt
├── evaluation_report.md
├── model_comparison.md
├── provenance.md
├── README.md
└── inference entrypoint

The model must be loadable without the training environment.

---

41. CREATE REAL PACKAGING TOOLING

The project documentation references packaging behavior that is incomplete.

Create:

scripts/evaluate.py
scripts/package_model.py

or the equivalent architecture.

Also create:

- dependency file,
- pinned versions,
- reproducible environment specification.

Prefer:

pyproject.toml
requirements.txt

with versions pinned appropriately for the tested environment.

Do not claim reproducibility without dependency information.

---

42. AUTOMATED TEST SUITE

Populate "tests/".

At minimum:

test_tokenizer.py
test_data_pipeline.py
test_preprocessing.py
test_dedup.py
test_split_contamination.py
test_model_shapes.py
test_rope.py
test_forward_numpy_jax.py
test_kv_cache.py
test_loss_mask.py
test_padding_mask.py
test_grad_accumulation.py
test_checkpoint_resume.py
test_sampler.py
test_inference.py
test_quantization.py
test_tool_protocol.py
test_tool_registry.py
test_evidence.py
test_sandbox.py
test_package_size.py

Before training:

run all correctness tests.

Before release:

run all regression tests again.

---

43. TOOL-USE DATA GENERATION

Create deterministic synthetic tool-use training data.

Examples:

SEARCH

USER:
What is the current population of X?

ASSISTANT:
TOOL_CALL search(...)

Tool result is deterministic mock evidence during training.

Then:

ASSISTANT FINAL:
...
[citation]

MULTI-STEP SEARCH

search
→ inspect result
→ fetch source
→ extract evidence
→ final

CALCULATION

user asks multi-step arithmetic
→ compute tool
→ result
→ final

CODE

user asks for code
→ code generation
→ sandbox execution
→ test result
→ repair if needed
→ final

ERROR RECOVERY

invalid search query
→ error
→ corrected query
→ result
→ final

NO-TOOL CASE

Teach the model that not every question requires a tool.

---

44. TRAINING TOOL CALLS WITH LOSS MASKS

For example:

SYSTEM             MASK
USER               MASK
ASSISTANT TOOLCALL TARGET
TOOL RESULT        MASK
ASSISTANT FINAL     TARGET

The model must learn:

when to call
what to call
how to call
how to read
when to stop
how to answer

Do not train tool result text as something that the model should reproduce verbatim.

---

45. TOOL-USE EVALUATION

Create measured metrics:

tool_needed_accuracy
tool_not_needed_accuracy
tool_name_accuracy
argument_validity
argument_accuracy
protocol_validity
successful_tool_execution
multi_step_success
error_recovery_success
grounded_final_answer
citation_validity

Do not say that the tiny model “has search” merely because the runtime has search.

The model must demonstrably learn the tool protocol.

---

46. SEARCH SHOULD NOT BE STORED INSIDE THE TINY MODEL

Architecturally:

TinyMe
+
Tool Runtime
+
Search/Retrieval
+
Evidence Engine
+
Sandbox

The tiny model is the controller/reasoner.

The search engine provides current external information.

Do NOT attempt to embed the entire web into a <50 MB model.

---

47. TRAINING DATA SCALE

The current dataset is tiny.

Do not solve this by repeating identical examples endlessly.

Instead maximize:

- uniqueness,
- verified correctness,
- task diversity,
- language diversity when supported,
- code diversity,
- template diversity,
- tool workflow diversity.

Target a meaningfully larger effective token corpus than the previous ~55k-token dataset when resources allow.

A practical goal may be:

hundreds of thousands to several million useful tokens

depending on available resources.

Do not fabricate corpus size.

Report exact measured token counts.

---

48. CURRICULUM

Experiment with:

Curriculum A

language
→ code
→ math
→ logic
→ instruction
→ tools

Curriculum B

mixed training

Do not run expensive experiments blindly.

Start with small pilots.

Keep the configuration that demonstrates measurable benefit.

---

49. TARGETED ITERATIVE LOOP

Implement:

TRAIN
↓
EVALUATE
↓
IDENTIFY WEAKNESS
↓
GENERATE TARGETED DATA
↓
TRAIN / FINETUNE
↓
EVALUATE
↓
COMPARE
↓
ACCEPT OR REJECT

Every iteration must document:

1. observed failure,
2. hypothesis,
3. intervention,
4. measured result,
5. regression check.

Do NOT accept a change merely because training loss became smaller.

---

50. PREVENT OVERFITTING

Monitor:

- train loss,
- validation loss,
- test metrics,
- domain metrics,
- generalization.

Watch for:

train ↓
validation ↓
test unchanged

or:

train ↓
validation ↑

which may indicate overfitting.

Do not endlessly train on the tiny corpus.

---

51. SYNTHETIC DATA QUALITY

Every deterministic synthetic example must have:

problem generator
solution generator
independent verifier

Prefer:

generate
→ solve independently
→ execute/check
→ accept only verified examples

Never blindly trust a generated answer.

For code:

buggy implementation
→ test failure
→ proposed repair
→ tests pass

The verifier must be separate enough to catch trivial generator mistakes.

---

52. DATA PROVENANCE

Create/update:

"docs/DATA_PROVENANCE.json"

and:

"docs/LICENSE_POLICY.md"

Track:

- source,
- source URL,
- source ID,
- dataset version,
- license,
- license URL,
- retrieval date,
- preprocessing,
- filtering,
- verification,
- inclusion/exclusion reason.

Do not make legal conclusions beyond documented project policy.

---

53. QUALITY REPORT

Generate:

"docs/DATA_QUALITY_REPORT.md"

Include:

- raw samples,
- filtered samples,
- train samples,
- validation samples,
- test samples,
- token counts,
- duplicate counts,
- language distribution,
- code distribution,
- reasoning distribution,
- verification rate,
- malformed-code rate,
- discarded samples by reason,
- contamination statistics,
- source distribution,
- license distribution.

The malformed-code rate should be effectively zero for samples intended to be valid source code.

---

54. TRAINING REPORT

Generate:

"docs/TRAINING_REPORT.md"

Include:

- model architecture,
- parameter count,
- tokenizer,
- sequence length,
- effective target tokens,
- batch size,
- gradient accumulation,
- optimizer,
- scheduler,
- learning rate,
- precision,
- seed,
- dataset fingerprint,
- environment,
- training time,
- throughput,
- checkpoints,
- final metrics,
- failures,
- fixes.

---

55. MODEL COMPARISON

Create:

"docs/MODEL_COMPARISON.md"

Compare:

EXP-001 historical
EXP-002 corrected
Nano candidate
Base candidate
FP32
FP16
INT8
INT4

But do NOT compare invalid historical metrics against corrected evaluation as if they were equivalent.

Explicitly mark:

NOT COMPARABLE

where the evaluation protocol changed.

---

56. EXPERIMENT VERSIONING

Every new experiment must have:

- unique experiment ID,
- immutable config,
- dataset version,
- tokenizer version,
- model signature,
- git commit,
- environment report.

Never overwrite an experiment.

Suggested:

EXP-002-CORRECTED-NANO
EXP-003-CORRECTED-BASE
EXP-004-TOOL-SFT
EXP-005-QUANT

Use meaningful names.

---

57. RESOURCE-AWARE EXECUTION

Inspect the actual environment.

Measure:

- CPU,
- RAM,
- disk,
- GPU,
- backend,
- Python,
- JAX,
- tokenizers,
- Optax,
- safetensors.

Do not blindly execute the GPU config on a CPU-only environment.

Choose the largest experiment that the environment can genuinely execute.

If full training is impossible:

run the largest valid subset and document the blocker.

Do not fabricate full-scale completion.

---

58. REQUIRED FRESH RETRAINING ORDER

After P0 fixes pass:

Step 1

Prepare corrected dataset.

Step 2

Create train/validation/test split.

Step 3

Train tokenizer only on training split.

Step 4

Run preprocessing/data-quality tests.

Step 5

Run model correctness tests.

Step 6

Run training smoke test.

Step 7

Train fresh corrected nano baseline.

Step 8

Evaluate on true validation.

Step 9

Evaluate on independent test.

Step 10

Run tool-use evaluation.

Step 11

If feasible, train/evaluate Base candidate.

Step 12

Select based on measurements.

Step 13

Run quantization.

Step 14

Functionally evaluate each quantized variant.

Step 15

Build hardened sandbox.

Step 16

Run sandbox security tests.

Step 17

Build complete release.

Step 18

Run final end-to-end regression suite.

---

59. DO NOT RESUME EXP-001

After the correctness repairs:

DO NOT do:

resume EXP-001

unless explicitly performing a controlled historical ablation.

The corrected model must normally be trained from new initialization because:

- preprocessing changes,
- loss masking changes,
- data splitting changes,
- tokenizer may change,
- RoPE may change,
- optimizer/resume logic changes,
- context length may change.

Create a new experiment.

---

60. FINAL ACCEPTANCE GATES

Do not declare completion unless all feasible gates have evidence.

CORE CORRECTNESS

[ ] model forward pass works
[ ] backward pass works
[ ] RoPE is correct
[ ] JAX/NumPy parity passes
[ ] cache/full-forward parity passes
[ ] padding is masked
[ ] loss masking works
[ ] gradient accumulation is correct
[ ] checkpoint resume is correct
[ ] RNG/sampler persistence works
[ ] compute dtype actually works

DATA

[ ] code indentation preserved
[ ] malformed valid-code records rejected
[ ] train/validation/test split exists
[ ] no contamination
[ ] tokenizer trained on training split only
[ ] deduplication works
[ ] provenance works
[ ] license policy works

TRAINING

[ ] fresh corrected training executed
[ ] actual train token count recorded
[ ] actual validation token count recorded
[ ] actual test metrics recorded
[ ] experiment reproducibility metadata recorded

MODEL

[ ] parameter count verified
[ ] tokenizer/model vocab match
[ ] model loads independently
[ ] generation works
[ ] context length behavior tested

TOOLS

[ ] tool protocol exists
[ ] tool registry works
[ ] search tool exists
[ ] retrieval works
[ ] calculator works
[ ] code execution works
[ ] evidence engine works
[ ] tool-use training data exists
[ ] tool-use evaluation exists

SANDBOX

[ ] timeout protection
[ ] memory protection
[ ] CPU protection
[ ] output limit
[ ] file limit
[ ] process limit
[ ] network restriction
[ ] host FS restriction
[ ] environment restriction
[ ] process cleanup
[ ] security tests executed

RELEASE

[ ] model serialized
[ ] tokenizer included
[ ] config included
[ ] checksums included
[ ] inference entrypoint exists
[ ] evaluation report exists
[ ] provenance exists
[ ] actual model artifact <50 MB
[ ] functional quantization verified
[ ] release loads successfully

---

61. NO-HALLUCINATION RULE

This is mandatory.

For every major claim use one of:

PLANNED
EXECUTED
VERIFIED
BLOCKED

Example:

GOOD:

Status: VERIFIED
Command:
...
Observed:
...
Artifact:
...

BAD:

The model was successfully trained to 5 million tokens.

when that was not actually executed.

Never fabricate:

- benchmark results,
- training duration,
- token counts,
- search results,
- dataset sizes,
- sandbox security,
- file sizes,
- model quality.

---

62. FAILURE RECOVERY

When something fails:

1. inspect the actual error,
2. identify root cause,
3. inspect affected source,
4. make smallest defensible fix,
5. rerun the failing test,
6. rerun dependent tests,
7. continue.

Do not repeatedly make speculative changes.

Do not hide failed experiments.

Record failures in:

"EXPERIMENT_LOG.md"

and machine-readable form:

"EXPERIMENT_LOG.jsonl"

---

63. DOCUMENTATION STATE

Keep these synchronized with actual repository state:

- STATE.md
- TODO.md
- DECISIONS.md
- EXECUTION_PLAN.md
- EXPERIMENT_LOG.md

Do not leave documents claiming requirements are pending when they are verified.

Do not mark requirements complete merely because a file exists.

Completion means:

"implemented + executed + verified".

---

64. FINAL REPORT

Produce:

"FINAL_REPORT.md"

Structure:

1. Objective

2. Initial Audit

3. Critical Problems Found

4. Corrective Changes

5. Dataset

6. Tokenizer

7. Architecture

8. Training

9. Evaluation

10. Tool Runtime

11. Evidence Engine

12. Sandbox

13. Quantization

14. Final Model Size

15. Inference Measurements

16. Regression Results

17. Failures and Fixes

18. Limitations

19. Reproduction Instructions

20. Release Artifacts

Every number must come from actual measurement.

---

65. FINAL README

Update README.md to explain:

- TinyMe purpose,
- tiny-model constraint,
- architecture,
- model size,
- parameter count,
- tokenizer,
- data strategy,
- training,
- evaluation,
- tool runtime,
- search system,
- evidence system,
- sandbox,
- limitations,
- installation,
- training,
- evaluation,
- inference,
- tool usage,
- reproduction.

Do not exaggerate.

Do not call TinyMe a frontier model.

Describe it as a tiny model paired with an external tool runtime.

---

66. IMPLEMENTATION PRIORITY

Use this order.

P0 — ABSOLUTE

1. destructive preprocessing
2. train/eval/test separation
3. loss masking
4. padding masking
5. sequence construction
6. true gradient accumulation
7. checkpoint/resume
8. RNG/sampler
9. RoPE
10. numerical stability
11. tokenizer split
12. model/tokenizer compatibility
13. actual tests

Do not proceed to expensive retraining until P0 is green.

P1

1. KV cache
2. context-length improvements
3. better dedup
4. better dataset diversity
5. evaluation framework
6. architecture comparison
7. tool protocol
8. tool runtime
9. evidence engine
10. hardened sandbox

P2

1. curriculum optimization
2. distillation where available
3. quantization improvements
4. inference optimization
5. provider optimization

P3

Optional research experiments.

Do NOT let P2/P3 delay P0.

---

67. CORE ARCHITECTURE TO ACHIEVE

The final system should conceptually become:

                    USER
                      │
                      ▼
                ┌───────────┐
                │  TinyMe   │
                │  < 50 MB  │
                └─────┬─────┘
                      │
            intent / planning
                      │
                      ▼
              ┌───────────────┐
              │ Tool Runtime  │
              └───────┬───────┘
                      │
      ┌───────────────┼────────────────┐
      │               │                │
      ▼               ▼                ▼
 SEARCH           RETRIEVAL        COMPUTE/CODE
 web              URL              calculator
 news             HTML             JSON
 image            PDF              CSV
 academic         documents        statistics
 code             find             sandbox
 docs
 package
      │               │                │
      └───────────────┼────────────────┘
                      ▼
               Evidence Engine
                      │
                      ▼
                  TinyMe
                      │
                      ▼
              grounded final answer

The tiny model remains tiny.

The external tools provide:

- current information,
- search,
- retrieval,
- calculation,
- execution.

---

68. FINAL DIRECTIVE

Do not optimize for making the repository look complete.

Optimize for making the repository ACTUALLY correct.

Do not optimize for the lowest training loss.

Optimize for reliable observable capability.

Do not optimize for the smallest model merely because it is small.

Optimize for the strongest model that satisfies:

"MODEL_ARTIFACT < 50 MB"

and the available compute budget.

Do not optimize for the most features.

Optimize for a coherent end-to-end system.

The final objective is:

CORRECT FOUNDATION
        ↓
HIGH-QUALITY DATA
        ↓
FRESH VERIFIED TRAINING
        ↓
INDEPENDENT EVALUATION
        ↓
TOOL-USE LEARNING
        ↓
SEARCH / RETRIEVAL
        ↓
EVIDENCE GROUNDING
        ↓
HARDENED SANDBOX
        ↓
QUANTIZATION
        ↓
< 50 MB MODEL
        ↓
PORTABLE VERIFIED RELEASE

WORK FIRST.

Inspect.

Fix.

Test.

Train from scratch.

Evaluate.

Diagnose.

Improve.

Retest.

Quantize.

Package.

Verify.

Document.

Never simulate success.

Never fabricate evidence.