import pytest
from gym.error import ResetNeeded

from truman import time_period_step


def test_discrete_strategy_binomial():
    def always_never_params(strategy, timestep):
        if strategy == 0:
            return 0.0, 0.0
        if strategy == 1:
            return 0.0, 1.0
        if strategy == 2:
            return 1.0, 0.0
        if strategy == 3:
            return 1.0, 1.0

    env = time_period_step.DiscreteStrategyBinomial(
        cohort_size=10,
        episode_length=5,
        strategy_keys=["never_never", "never_always", "always_never", "always_always"],
        behaviour_func=always_never_params,
    )

    obvs, reward, done, info = env.step(0)
    assert tuple(obvs) == (0, 0)
    assert reward == 0
    assert info["interaction_prb"] == 0.0
    assert info["conversion_prb"] == 0.0
    obvs, reward, done, info = env.step(1)
    assert tuple(obvs) == (0, 0)
    assert reward == 0
    assert info["interaction_prb"] == 0.0
    assert info["conversion_prb"] == 1.0
    obvs, reward, done, info = env.step(2)
    assert tuple(obvs) == (10, 0)
    assert reward == 0
    assert info["interaction_prb"] == 1.0
    assert info["conversion_prb"] == 0.0
    obvs, reward, done, info = env.step(3)
    assert tuple(obvs) == (10, 10)
    assert reward == 10
    assert info["interaction_prb"] == 1.0
    assert info["conversion_prb"] == 1.0


def test_discrete_strategy_binomial_correct_episode_length():
    env = time_period_step.DiscreteStrategyBinomial(
        cohort_size=0,
        episode_length=3,
        strategy_keys=["dummy"],
        behaviour_func=lambda strat, timestep: (1, 1),
    )

    env.step(0)
    env.step(0)
    env.step(0)
    with pytest.raises(ResetNeeded):
        env.step(0)

    env.reset()
    env.step(0)
    env.step(0)
    env.step(0)
    with pytest.raises(ResetNeeded):
        env.step(0)


def test_matching_sin7_interaction():
    behaviour_params = {0: (0.5, 0.2), 1: (0.5, 0.3)}
    for i in range(10):
        modifiers = time_period_step.matching_sin7_interaction(0, i, behaviour_params)
        assert 0 <= modifiers[0] <= 2
        assert 0 <= modifiers[1] <= 2
