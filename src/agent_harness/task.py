"""Base task protocol."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Task(Protocol):
    """Protocol for tasks to evaluate agents on."""

    @property
    def id(self) -> str:
        """Return the task identifier."""
        ...

    def evaluate(self, output: object) -> float:
        """Evaluate agent output and return a score between 0 and 1."""
        ...
