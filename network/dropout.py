import numpy as np

class Dropout:
    def __init__(self, p):
        self.probability = p

    def forward(self, x, **kwargs):
        """Sets a random percentage of neurons to 0
        :param x: input
        :return: output with random dropped neurons (some neurons are zeroed)
        """
        idx = np.random.choice(a=x.shape[2], size=int(x.shape[2]*self.probability), replace=False)
        x[idx] = 0
        return x
