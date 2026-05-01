"""Memory subsystem: MemoryProvider protocol and reference backends."""

from t0_agents.core.types import MemoryProvider
from t0_agents.memory.base import MemoryRecord, render_memory_summary
from t0_agents.memory.in_memory import InMemoryMemory
from t0_agents.memory.tensorzero import TensorZeroMemory

try:
    from t0_agents.memory.sqlite import SqliteMemory
except ImportError:
    SqliteMemory = None  # type: ignore[assignment,misc]

__all__ = [
    "InMemoryMemory",
    "MemoryProvider",
    "MemoryRecord",
    "SqliteMemory",
    "TensorZeroMemory",
    "render_memory_summary",
]
