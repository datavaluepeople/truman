"""Interface for running an agent on an env suites."""
from typing import Callable, List
from truman.typing import Agent

from gym import Env
from gym.envs.registration import EnvRegistry

from truman.run import simulation, store


DEFAULT_PARAMS = {
    "output_directory": "",
    "max_iters": 100_000,
}
REQUIRED_KEYS = ["output_directory"]


AgentFactory = Callable[[Env], Agent]


def run(agent_factory: AgentFactory, env_suites: List[EnvRegistry], run_params: dict):
    """Run an agent on a list of environment suites.

    Args:
      agent_factory: a function that takes a given environment and returns a compatible agent to
        run on that environment
      env_suites: a list of env suites (as gym EnvRegistries) to run the agents on
      run_params: a dictionary of run parameters.
        Required parameters
          - output_directory: directory to store the history and summary of each agent/environment
        Optional parameters
          - max_iters: int maximum iterations to run on an environment, default 100_000
    """
    params = _parse_params(run_params)
    for env_suite in env_suites:
        for spec in env_suite.all():
            env = spec.make()
            agent = agent_factory(env)
            _run_agent_env(agent, env, spec.id, params)


def _run_agent_env(agent: Agent, env: Env, env_id: str, run_params: dict):
    history, elapsed_time = simulation.run(agent, env, run_params)
    summary = store.summarise(history, elapsed_time, env_id, run_params)
    store.write(history, summary, env_id, run_params)


def _parse_params(run_params: dict) -> dict:
    missing_keys = set(REQUIRED_KEYS) - set(run_params.keys())

    if len(missing_keys) > 0:
        raise ValueError(f"Missing run parameters: {tuple(missing_keys)}")

    parsed = DEFAULT_PARAMS.copy()
    parsed.update(run_params)
    return parsed
