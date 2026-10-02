"""TinyMe data-engineering subsystem (corrected contract)."""
from .records import (  # noqa: F401
    ALLOWED_LICENSES, CATEGORIES, CONTEXT_ROLES, ROLES, SOURCE_TYPES, TARGET_ROLES,
    TASK_TYPES, Segment, TrainingRecord, classify_license, make_segment_record,
    parse_segments, record_from_hf, serialize_segments, validate_record,
)
