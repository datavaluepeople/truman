from unittest import mock

import pytest

from truman.run import interface


def test_missing_params():
    with pytest.raises(ValueError, match=r"Missing.*output_directory.*"):
        interface.run(None, None, run_params={})


class FakeRegistry:
    def __init__(self, ids):
        self.ids = ids

    def all(self):
        envs = []
        for id_ in self.ids:
            env = mock.Mock()
            env.id = id_
            envs.append(env)
        return envs


def test_run(mocker):
    patched_store = mocker.patch.object(interface, "store")
    patched_simulation = mocker.patch.object(interface, "simulation")
    patched_simulation.run.return_value = (None, None)

    env_suites = [FakeRegistry(["env_1", "env_2", "env_3", "env_4"])]
    agent_suite = FakeRegistry(["agent_1", "agent_2"])

    interface.run(agent_suite, env_suites, run_params={"output_directory": "test"})

    assert patched_simulation.run.call_count == 8
    assert patched_store.summarise.call_count == 8
    assert patched_store.write.call_count == 8

    # Check that the default params were filled in correctly
    filled_params = patched_simulation.run.call_args_list[0][0][-1]
    assert "max_iters" in filled_params
