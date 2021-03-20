from unittest import mock

import pytest

from truman.run import interface


def test_missing_params():
    with pytest.raises(ValueError, match=r"Missing.*output_directory.*"):
        interface.run(None, None, run_params={})


class FakeRegistry:
    def __init__(self, num_envs):
        self.num_envs = num_envs

    def all(self):
        return [mock.Mock() for _ in range(self.num_envs)]


def fake_agent_factory(_):
    return "fake"


def test_run(mocker):
    patched_store = mocker.patch.object(interface, "store")
    patched_simulation = mocker.patch.object(interface, "simulation")
    patched_simulation.run.return_value = (None, None)

    suites = [FakeRegistry(4)]

    interface.run(fake_agent_factory, suites, run_params={"output_directory": "test"})

    assert patched_simulation.run.call_count == 4
    assert patched_store.summarise.call_count == 4
    assert patched_store.write.call_count == 4

    # Check that the default params were filled in correctly
    filled_params = patched_simulation.run.call_args_list[0][0][-1]
    assert "max_iters" in filled_params
