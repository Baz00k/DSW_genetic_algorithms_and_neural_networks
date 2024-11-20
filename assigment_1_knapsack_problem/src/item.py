import numpy as np


class Item(np.ndarray):
    """Represents an item with weight and value"""

    def __new__(cls, weight: int, value: int):
        return np.array([weight, value])

    @property
    def weight(self):
        return self[0]

    @property
    def value(self):
        return self[1]
