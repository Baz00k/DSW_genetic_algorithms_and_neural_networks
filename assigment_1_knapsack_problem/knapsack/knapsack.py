from typing import List
import numpy as np

from .item import Item


class Knapsack:
    """Represents a knapsack with a capacity and items"""

    def __init__(
        self,
        available_items: List[Item],
        items: np.ndarray | None = None,
        capacity: int = 0,
    ):
        self.available_items = available_items
        self.capacity = capacity

        self._items = (
            items if items is not None else np.zeros(len(available_items), dtype=int)
        )
        self._total_weight = None
        self._total_value = None

    @property
    def items(self):
        return self._items

    @items.setter
    def items(self, value):
        self._items = value
        self._total_weight = None
        self._total_value = None

    @property
    def total_weight(self) -> int:
        """Total weight of the items in the knapsack"""
        if self._total_weight is None:
            self._total_weight = np.sum(
                self.items * np.array([item.weight for item in self.available_items])
            )
        return self._total_weight

    @property
    def total_value(self) -> int:
        """Total value of the items in the knapsack"""
        if self._total_value is None:
            self._total_value = np.sum(
                self.items * np.array([item.value for item in self.available_items])
            )
        return self._total_value

    @property
    def is_overweight(self) -> bool:
        """Whether the knapsack is overweight"""
        return self.total_weight > self.capacity

    def can_add_item(self, item: Item) -> bool:
        """Whether an item can be added to the knapsack"""
        return (
            not self.is_overweight and not self.items[self.available_items.index(item)]
        )

    def add_item(self, item: Item):
        """Add an item to the knapsack"""
        self.items[self.available_items.index(item)] = 1
        self._total_weight = None
        self._total_value = None

    def __str__(self):
        return (
            f"Knapsack("
            f"total_weight={self.total_weight}, "
            f"total_value={self.total_value}, "
            f"capacity={self.capacity}"
            f")"
        )
