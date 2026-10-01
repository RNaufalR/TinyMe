"""Independent evaluation engine (spec §19, §20)."""
from .evaluator import (  # noqa: F401
    EVAL_DOMAINS, EvaluationResult, evaluate_model, run_sandboxed_python,
)
