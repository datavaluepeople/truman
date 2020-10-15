"""Env suite of discrete action space env with periodicities in system."""
from typing import Tuple
import math

from truman import time_period_step as tps


def matching_sin7_interaction(strat: int, timestep: int) -> Tuple[float, float]:
    strat_behavior = {
        0: (0.5, 0.3),
        1: (0.5, 0.2),
    }
    day_of_week = timestep % 7
    modifier = math.sin((day_of_week / 7) * 2 * math.pi)
    return tuple([x * modifier for x in strat_behavior[strat]])


matching_sin7_conversions_03_02_cohort_size_10k = tps.DiscreteStrategyBinomial(
    cohort_size=10000, strategies=["a", "b"], interaction_params=matching_sin7_interaction
)
