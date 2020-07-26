

# basic multi armed bandit first

class Bandit():

    def __init__(self, conversion_rate: float):
        pass

    def action() -> bool:
        """Use rand to see if success based on self.conversion_rate"""
        pass


class BasicDiscreteBernoulliBandits():
    """N bandits, each with static success rate."""

    def __init__(self, bandits: List[Bandit]):
        self.bandits = bandits

    def step(self, selected_bandit: int) -> bool:
        """Use selected_bandit to do bandit.action()"""
        pass
