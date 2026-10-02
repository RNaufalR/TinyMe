#!/usr/bin/env python3
"""Independent tool-use evaluation of a checkpoint (audit §2, §4, §5).

The model drives the *real* runtime: BM25 retrieval over the shipped index, the
exact-arithmetic tool, the sandboxed code runner and the evidence engine.  Cases
are authored independently of the training generator (``src/evaluation/tool_cases.py``),
so a passing score reflects runtime behaviour rather than a memorised template.

Usage::

    python scripts/evaluate_tools.py --experiment EXP-004-TOOL-SFT --checkpoint best
    python scripts/evaluate_tools.py --experiment EXP-004-TOOL-SFT --variants fp32,int8
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.agent.executor import AgentRuntime  # noqa: E402
from src.agent.protocol import render_tool_result  # noqa: E402
from src.data.dataset_api import load_tokenizer  # noqa: E402
from src.evaluation.tool_cases import (build_tool_cases, citation_validity,  # noqa: E402
                                       summarise)
from src.inference.engine import InferenceEngine  # noqa: E402
from src.model import TinyMeConfig, count_parameters  # noqa: E402
from src.quantization.quantize import (dequantize_int4, dequantize_int8, flatten_params,  # noqa: E402
                                       quantize_int4, quantize_int8, unflatten_params)
from src.training.checkpoint import verify_checkpoint  # noqa: E402
from src.utils.io_utils import REPO_ROOT, read_json, write_json  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
log = logging.getLogger("tinyme.eval.tools")

SYSTEM_PROMPT = ("You are TinyMe, a controller for external tools. Use a tool only when needed, "
                 "then answer from the evidence.")

ROLE_MARKERS = ("<|system|>", "<|user|>", "<|assistant|>", "<|tool_call|>", "<|end_tool_call|>",
                "<|tool_result|>", "<|end_tool_result|>", "<|final|>", "<|thought|>", "<|code|>",
                "<|endcode|>", "<|answer|>")


def render_model_prompt(system: str, history: list[str]) -> str:
    """Render the conversation exactly the way the training sequences are built.

    ``encode_segment`` writes ``<|role|>\\n<body>``; the runtime history omits that
    newline, so it is normalised here.  Getting this wrong is what makes a model
    look incapable: the prompt must match the training distribution byte for byte.
    """
    parts = [f"<|system|>\n{system}"]
    for chunk in history:
        text = chunk
        for marker in ROLE_MARKERS:
            if text.startswith(marker) and not text.startswith(marker + "\n"):
                text = text[:len(marker)] + "\n" + text[len(marker):]
        parts.append(text)
    return "\n".join(parts) + "\n<|assistant|>\n"


def make_policy(engine: InferenceEngine, *, max_new_tokens: int = 96, temperature: float = 0.0,
                system: str = SYSTEM_PROMPT):
    calls = {"n": 0}

    def policy(request: str, history: list[str], step: int, tools: list[dict]) -> str:
        prompt = render_model_prompt(system, history)
        calls["n"] += 1
        text = engine.generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature,
                               top_p=1.0, repetition_penalty=1.0,
                               stop=["<|end_tool_call|>", "<|end_tool_result|>", "<|endcode|>"])
        return text.strip("\n")

    policy.calls = calls  # type: ignore[attr-defined]
    return policy


def _f32(tree):
    """Cast every array in a nested param tree to float32 (lists included)."""
    if isinstance(tree, dict):
        return {k: _f32(v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_f32(v) for v in tree]
    return tree.astype(np.float32) if hasattr(tree, "astype") else tree


def _apply_variant(engine: InferenceEngine, variant: str, base_params) -> dict:
    if variant == "fp32":
        return base_params
    if variant == "fp16":
        return _f32({k: (np.asarray(v).astype(np.float16) if hasattr(v, "shape") else v)
                     for k, v in base_params.items()})
    flat = flatten_params(base_params)
    if variant == "int8":
        q, scale = quantize_int8(flat)
        out = unflatten_params(dequantize_int8(q, scale))
    elif variant == "int4":
        q, scale = quantize_int4(flat)
        out = unflatten_params(dequantize_int4(q, scale, shape_hint={k: v.shape for k, v in flat.items()}))
    else:
        raise ValueError(f"unknown variant {variant!r}")
    return _f32(out)


def run_cases(engine: InferenceEngine, *, max_new_tokens: int, temperature: float,
              verbose: bool) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    transcripts: list[dict] = []
    for case in build_tool_cases():
        runtime = AgentRuntime(max_steps=5, max_seconds=60.0, system_prompt=SYSTEM_PROMPT)
        try:
            traj = runtime.solve(case.request, make_policy(engine, max_new_tokens=max_new_tokens,
                                                           temperature=temperature))
        finally:
            runtime.close()
        graded = case.grade(traj, case) if case.grade else {}
        row = {"case_id": case.case_id, "family": case.family, "expectation": case.expectation,
               "request": case.request, "final": traj.final, "stop_reason": traj.stop_reason,
               "tool_needed": case.tool_required, "notes": case.notes, **graded}
        row.update(citation_validity(traj))
        row["citation_valid"] = row["answers_with_citations"] == 0 or row["invalid"] == 0
        rows.append(row)
        transcripts.append({"case_id": case.case_id, "family": case.family,
                            "trajectory": traj.to_dict()})
        if verbose:
            log.info("%s (%s): stop=%s tools=%s final=%r", case.case_id, case.family,
                     traj.stop_reason, row.get("tools_used"), (traj.final or "")[:120])
    return rows, transcripts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--checkpoint", default="best", choices=["best", "latest", "final"])
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--variants", default="fp32", help="comma separated: fp32,fp16,int8,int4")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--backend", default="numpy", choices=["numpy", "jax"])
    ap.add_argument("--output", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    ckpt_dir = REPO_ROOT / "checkpoints" / args.experiment
    report = verify_checkpoint(ckpt_dir, args.checkpoint)
    if not report.get("ok"):
        raise SystemExit(f"checkpoint {args.experiment}/{args.checkpoint} failed verification: {report['errors']}")

    run_cfg_path = REPO_ROOT / "experiments" / args.experiment / "run_config.json"
    run_cfg = read_json(run_cfg_path) if run_cfg_path.exists() else {}
    dataset_version = args.dataset or run_cfg.get("dataset_version") or "dataset_v3"
    tokenizer = load_tokenizer(dataset_version)
    model_cfg_dict = run_cfg.get("model_config") or {}
    cfg = TinyMeConfig(**{k: v for k, v in model_cfg_dict.items()
                          if k in TinyMeConfig.__dataclass_fields__}) if model_cfg_dict \
        else TinyMeConfig.from_name("nano")
    cfg.vocab_size = int(tokenizer.vocab_size)

    tmp_cfg = REPO_ROOT / "experiments" / args.experiment / f"_eval_config_{args.checkpoint}.json"
    tmp_cfg.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")

    started = time.time()
    variants_out: dict[str, dict] = {}
    all_transcripts: dict[str, list[dict]] = {}
    base_params = None
    engine = None
    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        if engine is None:
            engine = InferenceEngine(ckpt_dir / f"{args.checkpoint}.safetensors",
                                     REPO_ROOT / "datasets" / "versions" / dataset_version / "tokenizer.json",
                                     tmp_cfg, backend=args.backend, strict_config=True)
            log.info("loaded %s/%s: %d params (config=%s)", args.experiment, args.checkpoint,
                     count_parameters(engine.params), engine.config_source)
            base_params = engine.params
        engine.set_params(_apply_variant(engine, variant, base_params))
        log.info("running tool-use cases for variant %s ...", variant)
        rows, transcripts = run_cases(engine, max_new_tokens=args.max_new_tokens,
                                      temperature=args.temperature, verbose=args.verbose)
        summary = summarise(rows)
        variants_out[variant] = {"summary": summary, "cases": rows}
        all_transcripts[variant] = transcripts
        log.info("variant %s: %s", variant, json.dumps({k: v for k, v in summary.items()
                                                        if k not in ("by_family",)}))

    # merging writer: running a subset of variants must never silently drop the
    # rows a previous run produced (that is how fp32 vanished from this file)
    out = Path(args.output) if args.output else (REPO_ROOT / "experiments" / args.experiment /
                                                 f"evaluation_tools_{args.checkpoint}.json")
    merged_variants = dict(variants_out)
    merged_transcripts = dict(all_transcripts)
    if out.exists():
        try:
            previous = json.loads(out.read_text(encoding="utf-8"))
            for name, stats in (previous.get("variants") or {}).items():
                merged_variants.setdefault(name, stats)
            for name, tr in (previous.get("transcripts") or {}).items():
                merged_transcripts.setdefault(name, tr)
        except (OSError, ValueError):
            log.warning("could not read previous %s; writing fresh results", out)

    payload = {"experiment": args.experiment, "checkpoint": args.checkpoint,
               "dataset_version": dataset_version, "backend": args.backend,
               "temperature": args.temperature, "max_new_tokens": args.max_new_tokens,
               "suite": "independent tool-use cases A-H (src/evaluation/tool_cases.py)",
               "variants": merged_variants, "transcripts": merged_transcripts,
               "duration_s": round(time.time() - started, 2),
               "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    write_json(payload, out)
    log.info("wrote %s", out)

    md = [f"# Tool-use evaluation — {args.experiment} ({args.checkpoint})",
          "",
          f"- Suite: independent cases A-H, executed through the real agent runtime + sandbox",
          f"- Dataset: `{dataset_version}` · temperature {args.temperature} · "
          f"{'greedy' if args.temperature <= 0 else 'sampled'} decoding",
          f"- Generated: {payload['generated_at']} ({payload['duration_s']} s)",
          "",
          "| Metric | " + " | ".join(variants_out) + " |",
          "| :--- | " + " | ".join("---:" for _ in variants_out) + " |"]
    for metric in ("tool_needed_accuracy", "tool_not_needed_accuracy", "tool_syntax_validity",
                   "tool_name_accuracy", "argument_validity", "argument_accuracy",
                   "execution_success", "multi_step_success", "error_recovery_success",
                   "grounded_final_answer", "citation_validity", "task_completion"):
        cells = []
        for v in variants_out:
            value = variants_out[v]["summary"].get(metric, 0.0)
            cells.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        md.append(f"| {metric} | " + " | ".join(cells) + " |")
    md += ["", "## Per-family task completion", "",
           "| Family | " + " | ".join(variants_out) + " |",
           "| :--- | " + " | ".join("---:" for _ in variants_out) + " |"]
    for fam in sorted({r["family"] for r in next(iter(variants_out.values()))["cases"]}):
        cells = [f"{variants_out[v]['summary']['by_family'].get(fam, 0.0):.4f}" for v in variants_out]
        md.append(f"| {fam} | " + " | ".join(cells) + " |")
    md += ["", "## Example trajectories (first case per family, fp32)", ""]
    for row, tr in zip(next(iter(variants_out.values()))["cases"], all_transcripts[next(iter(variants_out))]):
        if row["case_id"].endswith("0"):
            md.append(f"**{row['case_id']}** — {row['request'][:110]!r}  ")
            md.append(f"stop=`{row['stop_reason']}` tools={row.get('tools_used')} "
                      f"final=`{(row.get('final') or '')[:160]}`")
            md.append("")
    md_path = out.with_suffix(".md")
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    log.info("wrote %s", md_path)
    tmp_cfg.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
