from typing import Callable, Dict, List

import numpy as np

# basic multi armed bandit first


class Bandit():
    def __init__(self, conversion_rate: float):
        self.conversion_rate = 0
        pass

    def action(self, multiplier: float = 1.0) -> bool:
        """Use rand to see if success based on self.conversion_rate"""
        conversion_rate = min(self.conversion_rate * multiplier, 1)
        # TODO set random seeding etc
        return np.random.random() < conversion_rate


class BasicDiscreteBernoulliBandits():
    """N bandits, each with static success rate."""

    def __init__(self, bandits: List[Bandit]):
        self.bandits = bandits

    def step(self, selected_bandit: int) -> bool:
        """Use selected_bandit to do bandit.action()"""
        return self.bandits[selected_bandit].action()


class HeirarchicalStaticBernoulliBandits():
    def __init__(self, bandits: List[Bandit], context: Dict[str, Dict[str, float]]):
        self.bandits = bandits
        self.context = context

    def step(self, selected_bandit: int, selected_context: Dict[str, str]) -> bool:
        """Use selected_bandit to do bandit.action()"""
        context_multiplier = 1.0
        for dimension, selected in selected_context.items():
            context_multiplier *= self.context[dimension][selected]
        return self.bandits[selected_bandit].action(multiplier=context_multiplier)


class PseudoStaticBernoulliBandits():
    def __init__(self, bandits: List[Bandit], periodicity: Callable[[int], float]):
        self.bandits = bandits
        self.periodicity = periodicity
        self.timestep = 0

    def step(self, selected_bandit: int) -> bool:
        """Use selected_bandit to do bandit.action()"""
        periodicity_multiplier = self.periodicity(self.timestep)
        self.timestep += 1
        return self.bandits[selected_bandit].action(multiplier=periodicity_multiplier)


def eg_weekly_periodicity(timestep: int) -> float:
    # Weekends are slightly better for conversions
    weekly_periodicity = [1.0] * 5 + [1.1] * 2
    day_of_week = timestep % 7
    return float(weekly_periodicity[day_of_week])
