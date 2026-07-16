"""Base agent protocol."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class Agent(Protocol):
    """Protocol for agents to be evaluated."""

    def run(self, task: object) -> object:
        """Run the agent on a task and return the output."""
        ...
