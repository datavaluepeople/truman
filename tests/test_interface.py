from sellsim import interface

import pytest


@pytest.fixture()
def fix_random(mocker):
    mocker.patch("numpy.random.random", return_value=0.5)


def test_bandit(fix_random):
    low_bandit = interface.Bandit(conversion_rate=0.1)
    high_bandit = interface.Bandit(conversion_rate=0.9)

    assert not low_bandit.action()
    assert high_bandit.action()


def test_bandit_multiplier(fix_random):
    low_bandit = interface.Bandit(conversion_rate=0.1)

    assert low_bandit.action(multiplier=6)


def test_basic_discrete_bernoulli_bandit():
    always_bandit = interface.Bandit(conversion_rate=1)
    never_bandit = interface.Bandit(conversion_rate=0)

    env = interface.BasicDiscreteBernoulliBandits(bandits=[always_bandit, never_bandit])

    for _ in range(5):
        observation, reward, done, info = env.step(selected_bandit=0)
        assert observation
        assert reward == 1.0
    for _ in range(5):
        observation, reward, done, info = env.step(selected_bandit=1)
        assert not observation
        assert reward == 0.0


def test_basic_discrete_bernoulli_bandit_raises_too_few_bandits():
    env = interface.BasicDiscreteBernoulliBandits(bandits=[])

    with pytest.raises(AssertionError):
        env.step(1)
