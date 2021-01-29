"""Interfacing with env histories."""
import pandas as pd

from truman import plot


class History:
    """Captures and manages history of events & behaviour of the env."""

    def __init__(self):
        self.actions = []
        self.observations = []
        self.rewards = []
        self.dones = []
        self.infos = []

    def append(self, action, observation, reward, done, info):
        """Append events from a single step to the history."""
        self.actions.append(action)
        self.observations.append(observation)
        self.rewards.append(reward)
        self.dones.append(done)
        self.infos.append(info)

    def observable(self):
        """Return history of env events that should be strictly available to an observer."""
        return self.actions, self.observations, self.rewards, self.dones

    def all(self):
        """Return history of all env events including internal/latent variables."""
        return (*self.observable(), self.infos)

    def to_df(self):
        """Return history of all env events as a dataframe.

        Observations are given column names using their index within the observations list.
        """
        df = pd.DataFrame()
        df["action"] = self.actions
        for i in range(len(self.observations[0])):
            df[f"observation_{i}"] = [o[i] for o in self.observations]
        df["reward"] = self.rewards
        df["done"] = self.dones
        for i, info_name in enumerate(self.infos[1].keys()):  # info is None on step 0
            df[info_name] = [i[info_name] if i is not None else None for i in self.infos]
        return df

    def plot(
        self, alpha=0.7, use_cols="all", ax=None,
    ):
        """Plot history."""
        plot.plot(self, alpha=alpha, use_cols=use_cols, ax=ax)
