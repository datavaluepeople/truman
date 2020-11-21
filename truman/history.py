"""Interfacing with env histories."""
import pandas as pd

from truman import plot


class History:
    def __init__(self):
        self.actions = []
        self.observations = []
        self.rewards = []
        self.dones = []
        self.infos = []

    def append(self, action, observation, reward, done, info):
        self.actions.append(action)
        self.observations.append(observation)
        self.rewards.append(reward)
        self.dones.append(done)
        self.infos.append(info)

    def observable(self):
        return self.actions, self.observations, self.rewards, self.dones

    def all(self):
        return (*self.observable(), self.infos)

    def to_df(self):
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
        plot.plot(self, alpha=alpha, use_cols=use_cols, ax=ax)
