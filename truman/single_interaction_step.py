"""Contains envs which have an interaction with a single entity (e.g. bandit) for each step."""

from typing import Callable, Dict, List
from typing_extensions import Protocol
from truman.typing import StepReturn

import gym
import numpy as np


class Bandit:
    """A bandit implementation - like a slot machine with a fixed conversion rate."""

    def __init__(self, conversion_rate: float):
        self.conversion_rate = conversion_rate

    def action(self, multiplier: float = 1.0) -> bool:
        """Use rand to see if success based on bandit's conversion rate."""
        conversion_rate = min(self.conversion_rate * multiplier, 1)
        # TODO set random seeding etc
        return np.random.random() < conversion_rate


class BasicDiscreteBernoulliBandits(gym.Env):
    """Env of N bandits who each have a static conversion rate."""

    def __init__(self, bandits: List[Bandit]):
        self.bandits = bandits

        self.action_space = gym.spaces.Discrete(len(bandits))
        self.observation_space = gym.spaces.Discrete(1)

    def step(self, selected_bandit: int) -> StepReturn:
        """Select bandit and receive response."""
        assert self.action_space.contains(selected_bandit)
        observation = self.bandits[selected_bandit].action()
        reward = float(observation)
        return observation, reward, False, {}


class HeirarchicalStaticBernoulliBandits(gym.Env):
    """Env of N bandits whose conversion rates are modified by a context."""

    def __init__(self, bandits: List[Bandit], context: Dict[str, Dict[str, float]]):
        self.bandits = bandits
        self.context_keys = {key: list(value.keys()) for key, value in context.items()}
        self.context = [list(value.values()) for value in context.values()]

        self.action_space = gym.spaces.MultiDiscrete(
            [len(bandits)] + [len(c) for c in self.context]
        )
        self.observation_space = gym.spaces.Discrete(1)

    def step(self, action: List[int]) -> StepReturn:
        """Select bandit and receive response."""
        assert self.action_space.contains(action)
        selected_bandit = action[0]
        context_multiplier = 1.0
        for dim_action, dimension in zip(action[1:], self.context):
            context_multiplier *= dimension[dim_action]
        observation = self.bandits[selected_bandit].action(multiplier=context_multiplier)
        reward = float(observation)
        return observation, reward, False, {}


class StepperModifier(Protocol):
    """Protocol defining context modifier that can change per step."""

    def step(self) -> float:
        """Return next context modifier."""
        ...


def weekly_periodicity(modifiers: List[float]) -> Callable[[int], float]:
    """A utility wrapper for creating a weekly periodicity.

    This also serves as an example of how to write a periodicity function.
    """
    if len(modifiers) != 7:
        raise ValueError("There's 7 days in a week, so `modifier` must be of length 7.")

    def _periodicity(timestep: int) -> float:
        day_of_week = timestep % 7
        return float(modifiers[day_of_week])

    return _periodicity


class Periodicity:
    """A periodic modifier."""

    def __init__(self, periodicity: Callable[[int], float]):
        self.timestep = 0
        self.periodicity = periodicity

    def step(self) -> float:
        """Return next modifier."""
        multiplier = self.periodicity(self.timestep)
        self.timestep += 1
        return multiplier


class RandomWalkTrend:
    """A modifier which random walks within bounds."""

    def __init__(self, lower: float, upper: float, step_size: float):
        self.modifier = 1.0
        self.lower = lower
        self.upper = upper
        self.step_size = step_size

    def step(self) -> float:
        """Return next modifier."""
        direction = -1.0 if np.random.random() < 0.5 else +1.0
        self.modifier += direction * self.step_size
        self.modifier = min(self.upper, self.modifier)
        self.modifier = max(self.lower, self.modifier)
        return self.modifier


class TimestepContextualBernoulliBandits:
    """Env of bandits with contexts that are based on the timestep."""

    def __init__(self, bandits: List[Bandit], step_contexts: List[StepperModifier]):
        self.bandits = bandits
        self.step_contexts = step_contexts
        self.timestep = 0

        self.action_space = gym.spaces.Discrete(len(bandits))
        self.observation_space = gym.spaces.Discrete(1)

    def step(self, selected_bandit: int) -> StepReturn:
        """Select bandit and receive response."""
        assert self.action_space.contains(selected_bandit)

        context_multiplier = 1.0
        for context in self.step_contexts:
            context_multiplier *= context.step()
        self.timestep += 1

        observation = self.bandits[selected_bandit].action(multiplier=context_multiplier)
        reward = float(observation)
        return observation, reward, False, {}


weekly_with_trend = TimestepContextualBernoulliBandits(
    [Bandit(0.01), Bandit(0.02)],
    [RandomWalkTrend(0.8, 1.2, 0.01), Periodicity(weekly_periodicity([1.0] * 5 + [1.2] * 2))],
)
