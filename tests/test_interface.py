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


@pytest.mark.parametrize(
    "action, expected_observation, expected_reward",
    [
        ([0, 0], True, 1.0),  # always bandit in always country
        ([0, 1], False, 0.0),  # always bandit in never country
        ([1, 0], False, 0.0),  # never bandit in always country
        ([1, 1], False, 0.0),  # never bandit in never country
    ],
)
def test_heirarchical_static_bernoulli_bandits(
    action, expected_observation, expected_reward
):
    always_bandit = interface.Bandit(conversion_rate=1)
    never_bandit = interface.Bandit(conversion_rate=0)

    context = {"country": {"always": 1.0, "never": 0.0}}

    env = interface.HeirarchicalStaticBernoulliBandits(
        bandits=[always_bandit, never_bandit], context=context
    )

    for _ in range(5):
        observation, reward, done, info = env.step(action=action)
        assert observation == expected_observation
        assert reward == expected_reward


@pytest.mark.parametrize(
    "action",
    [
        [1, 0],  # not enough bandits (we only have one, and it's 0-indexed)
        [0],  # forgetting the context
        [0, 0, 0],  # too many contexts
        [0, 2],  # only two possible choices for the context and its 0-indexed
    ],
)
def test_heirarchical_static_bernoulli_bandits_raises_too_few_bandits(action):
    bandit = interface.Bandit(conversion_rate=0)
    env = interface.HeirarchicalStaticBernoulliBandits(
        bandits=[bandit], context={"country": {"always": 1, "never": 0}}
    )

    with pytest.raises(AssertionError):
        env.step(action)


def test_heirarchical_static_bernoulli_bandits_context_keys():
    bandit = interface.Bandit(conversion_rate=0)
    env = interface.HeirarchicalStaticBernoulliBandits(
        bandits=[bandit], context={"country": {"always": 1, "never": 0}}
    )
    assert env.context_keys == {"country": ["always", "never"]}
