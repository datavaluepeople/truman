"""Contains envs which are interacting with cohorts of bandits in each time period."""

from typing import Callable, Dict, List, Tuple
from typing_extensions import Protocol
from truman.typing import StepReturn

import functools
import math

import gym
import numpy as np
from scipy import stats

from truman import registry


class DiscreteStrategyBinomial(gym.Env):
    """An env of a discrete set of cohorts of bandits."""

    def __init__(
        self,
        cohort_size: int,
        episode_length: int,
        strategy_keys: List[str],
        behaviour_func: Callable[[int, int], Tuple[float, float]],
    ):
        self.cohort_size = cohort_size
        self.episode_length = episode_length
        self.strategies = {strategy_key: i for i, strategy_key in enumerate(strategy_keys)}
        self.behaviour_func = behaviour_func

        self.action_space = gym.spaces.Discrete(len(strategy_keys))
        self.observation_space = gym.spaces.Box(low=0, high=999999, shape=(2,), dtype=int)

        self.timestep = 0
        self.seed()

    def step(self, selected_strategy: int) -> StepReturn:
        """Select strategy (cohort of bandits) and receive response."""
        if self.timestep >= self.episode_length:
            raise gym.error.ResetNeeded("Environment needs resetting before use.")

        assert self.action_space.contains(selected_strategy)

        interaction_prb, conversion_prb = self.behaviour_func(selected_strategy, self.timestep)
        num_interactions = stats.binom.rvs(self.cohort_size, interaction_prb)
        num_conversions = stats.binom.rvs(num_interactions, conversion_prb)

        self.timestep += 1

        observation = np.array([num_interactions, num_conversions])
        reward = float(num_conversions)
        done = self.timestep >= self.episode_length
        return (
            observation,
            reward,
            done,
            {"interaction_prb": interaction_prb, "conversion_prb": conversion_prb},
        )

    def reset(self):
        """Reset env."""
        self.timestep = 0
        return np.array([0, 0])

    def seed(self, seed=None):
        """Seed env."""
        np.random.seed(seed)


class DiscreteStrategyBinomialAgent(Protocol):
    """Protocol that agents applied to this class of envs should conform to."""

    def act(self, previous_observation: np.ndarray) -> Tuple[int, dict]:
        """Choose an action given the previous observation.

        Args:
            previous_observation: numpy array of length 2 representing
                [n_interactions, n_conversions]

        Returns:
            tuple of (integer index of the chosen action, dict of any extra information)
        """
        pass


# Register specific envs
# ----------------------------------------


def static_interaction(
    strat: int,
    timestep: int,
    behaviour_params: Dict[int, Tuple[float, float]],
) -> Tuple[float, float]:
    """Static behaviour."""
    return behaviour_params[strat]


for strat_1_conv, strat_2_conv in [(0.2, 0.3), (0.02, 0.03), (0.002, 0.003)]:
    registry.register(
        id=f"TimePeriodStep:Static:conv_1:{strat_1_conv}:conv_2:{strat_2_conv}-v0",
        entry_point="truman.time_period_step:DiscreteStrategyBinomial",
        kwargs={
            "cohort_size": 10000,
            "episode_length": 365,
            "strategy_keys": ["a", "b"],
            "behaviour_func": functools.partial(
                static_interaction,
                behaviour_params={0: (0.5, strat_1_conv), 1: (0.5, strat_2_conv)},
            ),
        },
    )


def matching_sin7_interaction(
    strat: int,
    timestep: int,
    behaviour_params: Dict[int, Tuple[float, float]],
) -> Tuple[float, float]:
    """A weekly periodicity behaviour."""
    day_of_week = timestep % 7
    modifier = math.sin((day_of_week / 7) * 2 * math.pi) + 1
    return tuple([x * modifier for x in behaviour_params[strat]])  # type: ignore


for strat_1_conv, strat_2_conv in [(0.2, 0.3), (0.02, 0.03), (0.002, 0.003)]:
    registry.register(
        id=f"TimePeriodStep:Matching_sin7:conv_1:{strat_1_conv}:conv_2:{strat_2_conv}-v0",
        entry_point="truman.time_period_step:DiscreteStrategyBinomial",
        kwargs={
            "cohort_size": 10000,
            "episode_length": 365,
            "strategy_keys": ["a", "b"],
            "behaviour_func": functools.partial(
                matching_sin7_interaction,
                behaviour_params={0: (0.5, strat_1_conv), 1: (0.5, strat_2_conv)},
            ),
        },
    )


def non_stationary_trend_interaction(
    strat: int,
    timestep: int,
    behaviour_params: Dict[int, Tuple[float, float]],
) -> Tuple[float, float]:
    """A linear increasing up to limit trend behaviour."""
    modifier = 0.5 + min(timestep * 0.0025, 1)
    return tuple([x * modifier for x in behaviour_params[strat]])  # type: ignore


for strat_1_conv, strat_2_conv in [(0.2, 0.3), (0.02, 0.03), (0.002, 0.003)]:
    registry.register(
        id=f"TimePeriodStep:NonStationaryTrend:conv_1:{strat_1_conv}:conv_2:{strat_2_conv}-v0",
        entry_point="truman.time_period_step:DiscreteStrategyBinomial",
        kwargs={
            "cohort_size": 10000,
            "episode_length": 365,
            "strategy_keys": ["a", "b"],
            "behaviour_func": functools.partial(
                non_stationary_trend_interaction,
                behaviour_params={0: (0.5, strat_1_conv), 1: (0.5, strat_2_conv)},
            ),
        },
    )
