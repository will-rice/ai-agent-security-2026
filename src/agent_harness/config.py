"""Configuration for the agent harness."""

from pydantic import BaseModel


class HarnessConfig(BaseModel):
    """Configuration for the agent harness."""

    max_workers: int = 1
    timeout: float = 60.0
    seed: int = 42
    debug: bool = False
