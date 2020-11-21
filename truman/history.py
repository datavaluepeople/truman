"""Interfacing with env histories."""
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
