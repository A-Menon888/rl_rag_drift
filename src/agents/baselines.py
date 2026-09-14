import random

class AlwaysDirect:
    def act(self, observation): return 0
class AlwaysRetrieve:
    def act(self, observation): return 1
class RandomPolicy:
    def __init__(self, seed=7): self.rng = random.Random(seed)
    def act(self, observation): return self.rng.randrange(2)
