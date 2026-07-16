"""Main harness module."""

from agent_harness.agent import Agent
from agent_harness.config import HarnessConfig
from agent_harness.result import Result
from agent_harness.task import Task


class Harness:
    """Harness for running and evaluating agents against tasks."""

    def __init__(self, config: HarnessConfig | None = None) -> None:
        """Initialize the harness with an optional config."""
        self.config = config or HarnessConfig()
        self.results: list[Result] = []

    def run(self, agent: Agent, tasks: list[Task]) -> list[Result]:
        """Run an agent against a list of tasks and return results."""
        results = []
        for task in tasks:
            try:
                output = agent.run(task)
                score = task.evaluate(output)
                result = Result(
                    task_id=task.id,
                    agent_id=repr(agent),
                    score=score,
                    output=str(output),
                )
            except Exception as e:
                result = Result(
                    task_id=task.id,
                    agent_id=repr(agent),
                    score=0.0,
                    error=str(e),
                )
            results.append(result)
        self.results.extend(results)
        return results
