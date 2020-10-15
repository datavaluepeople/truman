"""Module needs renaming as restricting and refactoring goes on.

For now this contains envs which are interacting with cohorts of bandits in period blocks.


cohort = 10k; dist1(strategy, t) for interaction; dist2(strategy, t) for conversion

strat = {0, 1}

cohort_strat_0 = [dist1(0, t) ]
cohort_strat_1 = [dist1(1, t)  ]


"""
from typing import Callable, List, Tuple

import gym
import numpy as np
from scipy import stats


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

    def step(self, selected_strategy: int):
        assert self.action_space.contains(selected_strategy)

        interaction_prb, conversion_prb = self.behaviour_func(selected_strategy, self.timestep)
        num_interactions = stats.binom.rvs(self.cohort_size, interaction_prb)
        num_conversions = stats.binom.rvs(num_interactions, conversion_prb)

        self.timestep += 1

        observation = (num_interactions, num_conversions)
        reward = float(num_conversions)
        return observation, reward, False, {}
