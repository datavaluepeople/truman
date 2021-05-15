"""Core loop to run a single agent on a single environment."""
from typing import Tuple
from truman.typing import Agent

import time

import pandas as pd
from gym import Env

from truman import errors
from truman import history as history_module


def run(agent: Agent, env: Env, run_params: dict) -> Tuple[pd.DataFrame, float]:
    """Run an agent on an environment for a single episode.

    Returns:
      a tuple (dataframe of the full history, elapsed time in seconds)
    """
    obs = env.reset()
    history = history_module.History()
    history.append(None, obs, None, None, None, None)

    start_time = time.time()

    # Run the environment for a single "episode"
    for _ in range(run_params["max_iters"]):
        action, agent_info = agent.act(obs)
        obs, reward, done, env_info = env.step(action)
        history.append(action, obs, reward, done, env_info, agent_info)
        if done:
            elapsed_secs = time.time() - start_time
            return history.to_df(), elapsed_secs

    raise errors.StoppedEarly("Environment did not finish within max iterations.")
