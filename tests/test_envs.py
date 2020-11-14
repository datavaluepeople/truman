"""Generic tests ran over all envs in truman's registry.

Taken from: https://github.com/openai/gym/blob/master/gym/envs/tests/test_envs.py
"""
import pytest
import numpy as np

import truman


# This runs a smoketest on each env in the registry.
@pytest.mark.skip(reason="Envs need full implementations before they will pass.")
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

    env.close()
