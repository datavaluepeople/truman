from typing import Callable, Dict, List, Tuple
from typing_extensions import Protocol

import numpy as np
import gym

# basic multi armed bandit first


class Bandit:
    def __init__(self, conversion_rate: float):
        self.conversion_rate = conversion_rate

    def action(self, multiplier: float = 1.0) -> bool:
        """Use rand to see if success based on self.conversion_rate"""
        conversion_rate = min(self.conversion_rate * multiplier, 1)
        # TODO set random seeding etc
        return np.random.random() < conversion_rate


class BasicDiscreteBernoulliBandits(gym.Env):
    """N bandits, each with static success rate."""

    def __init__(self, bandits: List[Bandit]):
        self.bandits = bandits

        self.action_space = gym.spaces.Discrete(len(bandits))
        self.observation_space = gym.spaces.Discrete(1)

    def step(self, selected_bandit: int) -> Tuple[bool, float, bool, dict]:
        """Use selected_bandit to do bandit.action()"""
        assert self.action_space.contains(selected_bandit)
        observation = self.bandits[selected_bandit].action()
        reward = float(observation)
        return observation, reward, False, {}


class HeirarchicalStaticBernoulliBandits:
    def __init__(self, bandits: List[Bandit], context: Dict[str, Dict[str, float]]):
        self.bandits = bandits
        self.context = context

    def step(self, selected_bandit: int, selected_context: Dict[str, str]) -> bool:
        """Use selected_bandit to do bandit.action()"""
        context_multiplier = 1.0
        for dimension, selected in selected_context.items():
            context_multiplier *= self.context[dimension][selected]
        return self.bandits[selected_bandit].action(multiplier=context_multiplier)


class StepperModifier(Protocol):
    def step(self) -> float:
        ...


def eg_weekly_periodicity(timestep: int) -> float:
    # Weekends are slightly better for conversions
    weekly_periodicity = [1.0] * 5 + [1.1] * 2
    day_of_week = timestep % 7
    return float(weekly_periodicity[day_of_week])


class Periodicity:
    def __init__(self, periodicity: Callable[[int], float]):
        self.timestep = 0
        self.periodicity = periodicity

    def step(self) -> float:
        multiplier = self.periodicity(self.timestep)
        self.timestep += 1
        return multiplier


class RandomWalkTrend:
    def __init__(self, lower: float, upper: float, step_size: float):
        self.modifier = 1.0
        self.lower = lower
        self.upper = upper
        self.step_size = step_size

    def step(self) -> float:
        direction = +1.0 if np.random.random() < 0.5 else -1.0
        self.modifier += direction * self.step_size
        self.modifier
        self.modifier = min(self.upper, self.modifier)
        self.modifier = max(self.lower, self.modifier)
        return self.modifier


class TimestepContextualBernoulliBandits:
    def __init__(self, bandits: List[Bandit], step_contexts: List[StepperModifier]):
        self.bandits = bandits
        self.step_contexts = step_contexts
        self.timestep = 0

    def step(self, selected_bandit: int) -> bool:
        context_multiplier = 1.0
        for context in self.step_contexts:
            context_multiplier *= context.step()
        self.timestep += 1

        return self.bandits[selected_bandit].action(multiplier=context_multiplier)


weekly_with_trend = TimestepContextualBernoulliBandits(
    [Bandit(0.01), Bandit(0.02)],
    [RandomWalkTrend(0.8, 1.2, 0.01), Periodicity(eg_weekly_periodicity)],
)
