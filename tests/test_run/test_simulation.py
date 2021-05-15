import numpy as np
import pytest

from truman.errors import StoppedEarly
from truman.run import simulation


class FakeEnv:
    def __init__(self, num_steps):
        self.timestep = 0
        self.num_steps = num_steps

    def reset(self):
        return np.array([1])

    def step(self, _):
        self.timestep += 1
        return np.array([1]), 1, self.timestep == self.num_steps, {}


class FakeAgent:
    def act(self, _):
        return 1, {}


@pytest.mark.parametrize("num_steps", [5, 2])
def test_run(num_steps, mocker):
    mocker.patch.object(simulation.time, "time", side_effect=[1, 3])

    env = FakeEnv(num_steps)
    agent = FakeAgent()

    history, elapsed = simulation.run(agent=agent, env=env, run_params={"max_iters": 100})

    assert elapsed == 2
    assert len(history) == num_steps + 1


def test_run_raises_hits_max_iters():
    env = FakeEnv(10)
    agent = FakeAgent()

    with pytest.raises(StoppedEarly):
        simulation.run(agent=agent, env=env, run_params={"max_iters": 9})
