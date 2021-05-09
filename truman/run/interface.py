"""Interface for running an agent on an env suites."""
from typing import List
from truman.typing import Agent

from gym import Env
from gym.envs.registration import EnvRegistry

from truman.agent_registration import AgentRegistry
from truman.run import simulation, store


DEFAULT_PARAMS = {
    "output_directory": "",
    "max_iters": 100_000,
}
REQUIRED_KEYS = ["output_directory"]


def run(agent_suite: AgentRegistry, env_suites: List[EnvRegistry], run_params: dict):
    """Run an agent on a list of environment suites.

    Args:
      agent_suite: an AgentRegistry (from truman.agent_registration) containing the suite of agents
        to be run on the provided environments
      env_suites: a list of env suites (as gym EnvRegistries) to run the agents on
      run_params: a dictionary of run parameters.
        Required parameters
          - output_directory: directory to store the history and summary of each agent/environment
        Optional parameters
          - max_iters: int maximum iterations to run on an environment, default 100_000
    """
    params = _parse_params(run_params)
    _check_no_clashing_ids(env_suites)
    for env_suite in env_suites:
        for env_spec in env_suite.all():
            for agent_spec in agent_suite.all():
                env = env_spec.make()
                agent = agent_spec.make(env)
                _run_agent_env(agent, env, agent_spec.id, env_spec.id, params)


def _run_agent_env(agent: Agent, env: Env, agent_id: str, env_id: str, run_params: dict):
    history, elapsed_time = simulation.run(agent, env, run_params)
    summary = store.summarise(history, elapsed_time, agent_id, env_id, run_params)
    store.write(history, summary, agent_id, env_id, run_params)


def _parse_params(run_params: dict) -> dict:
    missing_keys = set(REQUIRED_KEYS) - set(run_params.keys())

    if len(missing_keys) > 0:
        raise ValueError(f"Missing run parameters: {tuple(missing_keys)}")

    parsed = DEFAULT_PARAMS.copy()
    parsed.update(run_params)
    return parsed


def _check_no_clashing_ids(env_suites: List[EnvRegistry]):
    ids = []
    for env_suite in env_suites:
        ids += [spec.id for spec in env_suite.all()]

    assert len(set(ids)) == len(ids), "Repeated environment ID in env suites"
