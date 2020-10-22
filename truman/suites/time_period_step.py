"""Env suite of discrete action space env with periodicities in system."""
import functools
from typing import Dict, Tuple
import math

from truman import time_period_step as tps


def matching_sin7_interaction(
    strat: int, timestep: int, behaviour_params: Dict[int, Tuple[float, float]],
) -> Tuple[float, float]:
    day_of_week = timestep % 7
    modifier = math.sin((day_of_week / 7) * 2 * math.pi)
    return tuple([x * modifier for x in behaviour_params[strat]])  # type: ignore


matching_sin7_conversions = {
    f"strat_1_conversion:{i}__strat_2_conversion:{j}": tps.DiscreteStrategyBinomial(
        cohort_size=10000,
        strategy_keys=["a", "b"],
        behaviour_func=functools.partial(
            matching_sin7_interaction, strat_behaviour={0: (0.5, i), 1: (0.5, j)}
        ),
    )
    for (i, j) in [(0.2, 0.3), (0.02, 0.03), (0.002, 0.003)]
}


def non_stationary_trend_interaction(
    strat: int, timestep: int, behaviour_params: Dict[int, Tuple[float, float]],
) -> Tuple[float, float]:
    modifier = 0.5 + min(timestep * 0.01, 1)
    return tuple([x * modifier for x in behaviour_params[strat]])  # type: ignore


non_stationary_trend_conversions = {
    f"strat_1_conversion:{i}__strat_2_conversion:{j}": tps.DiscreteStrategyBinomial(
        cohort_size=10000,
        strategy_keys=["a", "b"],
        behaviour_func=functools.partial(
            non_stationary_trend_interaction, strat_behaviour={0: (0.5, i), 1: (0.5, j)}
        ),
    )
    for (i, j) in [(0.2, 0.3), (0.02, 0.03), (0.002, 0.003)]
}
