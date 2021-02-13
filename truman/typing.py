"""Truman (gymenv) types."""
from typing import Any, Tuple
from typing_extensions import Protocol


StepReturn = Tuple[Any, float, bool, dict]


class Agent(Protocol):
    """Protocol that agents should implement."""

    def act(self, previous_observation) -> Tuple[Any, dict]:
        """Choose an action given the previous observation.

        Returns:
            tuple of (chosen action, dict of any extra information)
        """
        ...
