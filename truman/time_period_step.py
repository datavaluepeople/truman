"""Contains envs which are interacting with cohorts of bandits in each time period."""
from typing import Dict, Callable, List, Tuple
from truman.typing import StepReturn

import functools
import math

import gym
import numpy as np
from scipy import stats

from truman import registry


class DiscreteStrategyBinomial(gym.Env):
    def __init__(
        self,
        cohort_size: int,
        strategy_keys: List[str],
        behaviour_func: Callable[[int, int], Tuple[float, float]],
    ):
        self.cohort_size = cohort_size
        self.strategies = {strategy_key: i for i, strategy_key in enumerate(strategy_keys)}
        self.behaviour_func = behaviour_func

        self.action_space = gym.spaces.Discrete(len(strategy_keys))
        self.observation_space = gym.spaces.Box(low=0, high=999999, shape=(2,), dtype=np.int)

        self.timestep = 0
        self.seed()

    def step(self, selected_strategy: int) -> StepReturn:
        assert self.action_space.contains(selected_strategy)

        interaction_prb, conversion_prb = self.behaviour_func(selected_strategy, self.timestep)
        num_interactions = stats.binom.rvs(self.cohort_size, interaction_prb)
        num_conversions = stats.binom.rvs(num_interactions, conversion_prb)

        self.timestep += 1

        observation = np.array([num_interactions, num_conversions])
        reward = float(num_conversions)
        return (
            observation,
            reward,
            False,
            {"interaction_prb": interaction_prb, "conversion_prb": conversion_prb},
        )

    def reset(self):
        self.timestep = 0
        return np.array([0, 0])

    def seed(self, seed=None):
        np.random.seed(seed)


# Register specific envs
# ----------------------------------------


def matching_sin7_interaction(
    strat: int, timestep: int, behaviour_params: Dict[int, Tuple[float, float]],
) -> Tuple[float, float]:
    day_of_week = timestep % 7
    modifier = math.sin((day_of_week / 7) * 2 * math.pi) + 1
    return tuple([x * modifier for x in behaviour_params[strat]])  # type: ignore


for strat_1_conv, strat_2_conv in [(0.2, 0.3), (0.02, 0.03), (0.002, 0.003)]:
    registry.register(
        id=f"TimePeriodStep:Matching_sin7:conv_1:{strat_1_conv}:conv_2:{strat_2_conv}-v0",
        entry_point="truman.time_period_step:DiscreteStrategyBinomial",
        kwargs={
            "cohort_size": 10000,
            "strategy_keys": ["a", "b"],
            "behaviour_func": functools.partial(
                matching_sin7_interaction,
                behaviour_params={0: (0.5, strat_1_conv), 1: (0.5, strat_2_conv)},
            ),
        },
    )


def non_stationary_trend_interaction(
    strat: int, timestep: int, behaviour_params: Dict[int, Tuple[float, float]],
) -> Tuple[float, float]:
    modifier = 0.5 + min(timestep * 0.01, 1)
    return tuple([x * modifier for x in behaviour_params[strat]])  # type: ignore


for strat_1_conv, strat_2_conv in [(0.2, 0.3), (0.02, 0.03), (0.002, 0.003)]:
    registry.register(
        id=f"TimePeriodStep:NonStationaryTrend:conv_1:{strat_1_conv}:conv_2:{strat_2_conv}-v0",
        entry_point="truman.time_period_step:DiscreteStrategyBinomial",
        kwargs={
            "cohort_size": 10000,
            "strategy_keys": ["a", "b"],
            "behaviour_func": functools.partial(
                non_stationary_trend_interaction,
                behaviour_params={0: (0.5, strat_1_conv), 1: (0.5, strat_2_conv)},
            ),
        },
    )
