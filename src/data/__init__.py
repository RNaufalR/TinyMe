"""TinnyMe data-engineering subsystem."""
from .records import (  # noqa: F401
    ALLOWED_LICENSES, CATEGORIES, SOURCE_TYPES, TASK_TYPES, TrainingRecord,
    classify_license, record_from_hf, validate_record,
)
