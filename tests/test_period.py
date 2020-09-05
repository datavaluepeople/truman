from truman import period


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

    env = period.DiscreteStrategyBinomial(
        10, ["never_never", "never_always", "always_never", "always_always"], always_never_params
    )

    obvs, reward, done, info = env.step(0)
    assert obvs == (0, 0)
    assert reward == 0
    obvs, reward, done, info = env.step(1)
    assert obvs == (0, 0)
    assert reward == 0
    obvs, reward, done, info = env.step(2)
    assert obvs == (10, 0)
    assert reward == 0
    obvs, reward, done, info = env.step(3)
    assert obvs == (10, 10)
    assert reward == 10
