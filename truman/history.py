"""Interfacing with histories of env & agent (the system)."""
import pandas as pd

from truman import plot


class History:
    """Captures and manages history of events & behaviour of the system."""

    def __init__(self):
        self.actions = []
        self.observations = []
        self.rewards = []
        self.dones = []
        self.infos = []
        self.agent_infos = []

    def append(self, action, observation, reward, done, info, agent_info):
        """Append events from a single step to the history."""
        self.actions.append(action)
        self.observations.append(observation)
        self.rewards.append(reward)
        self.dones.append(done)
        self.infos.append(info)
        self.agent_infos.append(agent_info)

    def observable(self):
        """Return history of system events that should be strictly available to an observer."""
        return self.actions, self.observations, self.rewards, self.dones

    def all(self):
        """Return history of all system events including internal/latent variables."""
        return (
            *self.observable(),
            self.infos,
            self.agent_infos,
        )

    def to_df(self):
        """Return history of all system events as a dataframe.

        Observations are given column names using their index within the observations list.
        """
        df = pd.DataFrame()
        df["action"] = self.actions
        for i in range(len(self.observations[0])):
            df[f"observation_{i}"] = [o[i] for o in self.observations]
        df["reward"] = self.rewards
        df["done"] = self.dones
        for prefix, infos in zip(["", "agent_"], [self.infos, self.agent_infos]):
            for i, name in enumerate(infos[1].keys()):  # info is None on step 0
                df[prefix + name] = [i[name] if i is not None else None for i in infos]
        return df

    def plot(
        self,
        alpha=0.7,
        use_cols="all",
        ax=None,
    ):
        """Plot history."""
        df = self.to_df()
        plot.plot(df, alpha=alpha, use_cols=use_cols, ax=ax)
