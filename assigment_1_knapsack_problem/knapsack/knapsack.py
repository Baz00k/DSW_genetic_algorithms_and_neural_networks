from .item import Item


class Knapsack:
    """Represents a knapsack with a capacity and items"""

    def __init__(self, capacity: int, items: [Item]):
        self.capacity = capacity
        self.items = items

    @property
    def total_value(self) -> int:
        return sum(item.value for item in self.items)

    @property
    def total_weight(self) -> int:
        return sum(item.weight for item in self.items)

    @property
    def is_overweight(self) -> bool:
        return self.total_weight > self.capacity
