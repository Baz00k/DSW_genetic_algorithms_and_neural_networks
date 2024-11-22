class Item:
    """Represents an item with weight and value"""

    def __init__(self, weight: int, value: int):
        self.weight = weight
        self.value = value

    def __str__(self):
        return f"Item(weight={self.weight}, value={self.value})"

    @property
    def value_per_weight(self):
        """Calculate the value per weight of the item"""

        return self.value / self.weight
