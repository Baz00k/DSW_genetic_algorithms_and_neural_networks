import pandas as pd

from .item import Item
from .knapsack import Knapsack


class KnapsackGASolver:
    """Solves the knapsack problem using a genetic algorithm"""

    def __init__(
            self,
            data: pd.DataFrame,
            max_capacity: int,
            population_size: int,
            generations: int,
            mutation_rate: float):
        """
        Initialize the solver

        Args:
            data: DataFrame containing the weight and value of each item
            max_capacity: Maximum weight the knapsack can hold
            population_size: Number of individuals in the population
            generations: Number of generations to run the algorithm for
            mutation_rate: Probability of a mutation occurring
        """
        self.available_items = [
            Item(row.weight, row.value) for row in data.itertuples()
        ]
        self.max_capacity = max_capacity
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
