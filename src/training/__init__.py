"""Training engine (JAX/XLA CPU) with checkpointing, resume and crash recovery."""
from .trainer import TrainConfig, Trainer, load_dataset  # noqa: F401
