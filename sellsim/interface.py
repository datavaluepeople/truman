from typing import Any, Dict, List, Tuple

import numpy as np

# basic multi armed bandit first


class Bandit():
    def __init__(self, conversion_rate: float):
        self.conversion_rate = 0
        pass

    def action(self, context_multiplier: float = 1.0) -> bool:
        """Use rand to see if success based on self.conversion_rate"""
        conversion_rate = min(self.conversion_rate * context_multiplier, 1)
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
        return self.bandits[selected_bandit].action(context_multiplier=context_multiplier)
