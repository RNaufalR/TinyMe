"""Probe: measured state of dataset_v1 and EXP-001 validation provenance."""
import json, collections
tr = [json.loads(l) for l in open("datasets/versions/dataset_v1/train.jsonl")]
ev = [json.loads(l) for l in open("datasets/versions/dataset_v1/eval.jsonl")]
print("train rows", len(tr), "eval rows", len(ev))
print("train chars", sum(len(r["text"]) for r in tr), "eval chars", sum(len(r["text"]) for r in ev))
print("eval rows carrying token_ids:", sum(1 for r in ev if "token_ids" in r))
s = json.load(open("experiments/EXP-001/summary.json"))
print("EXP-001 final_val_perplexity =", s["final_val_perplexity"], "-> produced from train.jsonl[:32] (see scripts/train.py line 63)")
