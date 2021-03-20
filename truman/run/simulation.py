"""Core loop to run a single agent on a single environment."""
from typing import Tuple

import time

from gym import Env
import pandas as pd

from truman import history as history_module
from truman.typing import Agent


def run(agent: Agent, env: Env, run_params: dict) -> Tuple[pd.DataFrame, int]:
    """Run an agent on an environment for a single episode.

    Returns:
      a tuple (dataframe of the full history, elapsed time in seconds)
    """
    obs = env.reset()
    history = history_module.History()
    history.append(None, obs, None, None, None, None)

    with Timer() as t:
        # Run the environment for a single "episode"
        for _ in range(run_params["max_iters"]):
            action, agent_info = agent.act(obs)
            obs, reward, done, env_info = env.step(action)
            history.append(action, obs, reward, done, env_info, agent_info)
            if done:
                break

    return history.to_df(), t.elapsed


class Timer:
    """Timer utility."""

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start
