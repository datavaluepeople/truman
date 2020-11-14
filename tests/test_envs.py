"""Generic tests ran over all envs in truman's registry.

Taken from: https://github.com/openai/gym/blob/master/gym/envs/tests/test_envs.py
"""
import pytest
import numpy as np

import truman


# This runs a smoketest on each env in the registry.
@pytest.mark.parametrize("env_spec", truman.registry.all())
def test_env(env_spec):
    """Smoketest env."""
    env = env_spec.make()
    ob_space = env.observation_space
    act_space = env.action_space
    ob = env.reset()
    assert ob_space.contains(ob), "Reset observation: {!r} not in space".format(ob)
    a = act_space.sample()
    observation, reward, done, _info = env.step(a)
    assert ob_space.contains(observation), "Step observation: {!r} not in space".format(
        observation
    )
    assert np.isscalar(reward), "{} is not a scalar for {}".format(reward, env)
    assert isinstance(done, bool), "Expected {} to be a boolean".format(done)

    for mode in env.metadata.get("render.modes", []):
        env.render(mode=mode)

    # Make sure we can render the environment after close.
    for mode in env.metadata.get("render.modes", []):
        env.render(mode=mode)

    # Make sure that seeding the environment leads to reproducible results
    env_1, env_2 = env_spec.make(), env_spec.make()
    actions = [env_1.action_space.sample() for _ in range(10)]
    # Collect results after seeding for each environments, then check equality
    env_1.seed(2020)
    results_1 = [env_1.step(action) for action in actions]
    env_2.seed(2020)
    results_2 = [env_2.step(action) for action in actions]
    for r_1, r_2 in zip(results_1, results_2):
        observation_1, reward_1, done_1, info_1 = r_1
        observation_2, reward_2, done_2, info_2 = r_2
        assert (observation_1 == observation_2).all()
        assert (reward_1, done_1, info_1) == (reward_2, done_2, info_2)

    env.close()
