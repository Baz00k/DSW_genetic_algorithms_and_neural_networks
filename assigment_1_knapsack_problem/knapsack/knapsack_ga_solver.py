from typing import List
from enum import Enum
from tqdm import tqdm
import pandas as pd
import numpy as np

from .item import Item
from .knapsack import Knapsack


class KnapsackGASolver:
    """Solves the knapsack problem using a genetic algorithm"""

    class CrossoverType(str, Enum):
        """Types of crossover operations"""

        SINGLE_POINT = "single_point"
        TWO_POINT = "two_point"

    class SelectionType(str, Enum):
        """Types of selection operations"""

        ROULETTE_WHEEL = "roulette_wheel"
        TOURNAMENT = "tournament"
        RANK = "rank"

    class FitnessType(str, Enum):
        """Types of fitness operations"""

        LINEAR = "linear"
        EXPONENTIAL = "exponential"

    def __init__(
        self,
        data: pd.DataFrame,
        max_capacity: int,
        population_size: int,
        generations: int,
        crossover_type: "KnapsackGASolver.CrossoverType",
        mutation_rate: float,
        selection_type: "KnapsackGASolver.SelectionType",
        fitness_type: "KnapsackGASolver.FitnessType",
    ):
        """
        Initialize the solver

        Args:
            data: DataFrame containing the weight and value of each item
            max_capacity: Maximum weight the knapsack can hold
            population_size: Number of individuals in the population
            generations: Number of generations to run the algorithm for
            crossover_type: Type of crossover to use
            mutation_rate: Probability of a mutation occurring
            selection_type: Type of selection to use
            fitness_type: Type of fitness function to use
        """
        self.available_items = [
            Item(row.weight, row.value) for row in data.itertuples()
        ]
        self.max_capacity = max_capacity
        self.population_size = population_size
        self.generations = generations
        self.crossover_type = crossover_type
        self.mutation_rate = mutation_rate
        self.selection_type = selection_type
        self.fitness_type = fitness_type
        self.fitness_history = []

        self.population: np.ndarray = []

        self.selection_methods = {
            self.SelectionType.ROULETTE_WHEEL: self._roulette_wheel_selection,
            self.SelectionType.TOURNAMENT: self._tournament_selection,
            self.SelectionType.RANK: self._rank_selection,
        }

        self.crossover_methods = {
            self.CrossoverType.SINGLE_POINT: self._single_point_crossover,
            self.CrossoverType.TWO_POINT: self._two_point_crossover,
        }

        self.fitness_methods = {
            self.FitnessType.LINEAR: self._evaluate_fitness_linear,
            self.FitnessType.EXPONENTIAL: self._evaluate_fitness_exponential,
        }

    def _initialize_population(self):
        """Initialize the population with valid and diverse individuals"""
        sorted_items = sorted(
            self.available_items, key=lambda i: i.value_per_weight, reverse=True
        )

        for _ in range(self.population_size):
            items = np.zeros(len(self.available_items), dtype=int)
            remaining_capacity = self.max_capacity

            for item in sorted_items:
                if item.weight <= remaining_capacity:
                    if np.random.rand() > 0.5:  # Randomly decide to include the item
                        index = self.available_items.index(item)
                        items[index] = 1
                        remaining_capacity -= item.weight

            knapsack = Knapsack(self.available_items, items, self.max_capacity)
            self.population = np.append(self.population, knapsack)

    def _evaluate_fitness(self, knapsack: Knapsack) -> int:
        """Evaluate the fitness of a knapsack"""
        return self.fitness_methods[self.fitness_type](knapsack)

    def _select_parents(self) -> List[Knapsack]:
        """Select parents for crossover based on the selection type"""
        return self.selection_methods[self.selection_type]()

    def _roulette_wheel_selection(self) -> List[Knapsack]:
        """Roulette wheel selection method"""
        fitness_values = np.array([self._evaluate_fitness(k) for k in self.population])
        total_fitness = np.sum(fitness_values)

        # Avoid division by zero
        if total_fitness == 0:
            return np.random.choice(self.population, size=self.population_size)

        probabilities = fitness_values / total_fitness

        return np.random.choice(
            self.population, size=self.population_size, p=probabilities
        )

    def _tournament_selection(self) -> List[Knapsack]:
        """Tournament selection method"""
        tournament_size = 5

        new_population = []
        for _ in range(self.population_size):
            tournament = np.random.choice(self.population, size=tournament_size)
            winner = max(tournament, key=self._evaluate_fitness)
            new_population.append(winner)

        return new_population

    def _rank_selection(self) -> List[Knapsack]:
        """Rank selection method"""
        ranked_population = sorted(self.population, key=self._evaluate_fitness)
        ranks = np.arange(1, len(ranked_population) + 1)
        probabilities = ranks / np.sum(ranks)
        return np.random.choice(
            ranked_population, size=self.population_size, p=probabilities
        )

    def _crossover(self, parent1: Knapsack, parent2: Knapsack) -> Knapsack:
        """Perform crossover between two parents to produce an offspring"""
        return self.crossover_methods[self.crossover_type](parent1, parent2)

    def _single_point_crossover(self, parent1: Knapsack, parent2: Knapsack) -> Knapsack:
        """Single point crossover method"""
        crossover_point = np.random.randint(1, len(parent1.items))
        child_items = np.concatenate(
            (parent1.items[:crossover_point], parent2.items[crossover_point:])
        )
        return Knapsack(
            self.available_items, items=child_items, capacity=self.max_capacity
        )

    def _two_point_crossover(self, parent1: Knapsack, parent2: Knapsack) -> Knapsack:
        """Two point crossover method"""
        point1, point2 = sorted(
            np.random.choice(range(1, len(parent1.items)), size=2, replace=False)
        )
        child_items = np.concatenate(
            (
                parent1.items[:point1],
                parent2.items[point1:point2],
                parent1.items[point2:],
            )
        )
        return Knapsack(
            self.available_items, items=child_items, capacity=self.max_capacity
        )

    def _evaluate_fitness_linear(self, knapsack: Knapsack) -> int:
        """Evaluate the fitness of a knapsack using a linear function"""
        return knapsack.total_value if not knapsack.is_overweight else 0

    def _evaluate_fitness_exponential(self, knapsack: Knapsack) -> int:
        """Evaluate the fitness of a knapsack using an exponential function"""
        return knapsack.total_value**2 if not knapsack.is_overweight else 0

    def _mutate(self, knapsack: Knapsack):
        """Mutate a knapsack by flipping a random bit"""
        mutation_mask = np.random.rand(len(knapsack.items)) < self.mutation_rate
        knapsack.items = np.logical_xor(knapsack.items, mutation_mask).astype(int)

    def _record_fitness(self):
        """Record the fitness of the best individual in the population"""
        best_knapsack = max(self.population, key=self._evaluate_fitness)
        self.fitness_history.append(self._evaluate_fitness(best_knapsack))

    def _select_parents(self) -> List[Knapsack]:
        """Return two random parents from the population"""
        return np.random.choice(self.population, size=2, replace=False)

    def _select_survivors(self) -> List[Knapsack]:
        """Select survivors to limit the population size based on the selection type"""
        return self.selection_methods[self.selection_type]()

    def solve(self) -> Knapsack:
        """Solve the knapsack problem using a genetic algorithm"""
        self._initialize_population()

        for _ in tqdm(range(self.generations)):
            for _ in range(self.population_size):
                parent1, parent2 = self._select_parents()
                child = self._crossover(parent1, parent2)
                self._mutate(child)
                self.population = np.append(self.population, child)

            self.population = self._select_survivors()
            self._record_fitness()

        best_knapsack = max(self.population, key=self._evaluate_fitness)

        return best_knapsack if self._evaluate_fitness(best_knapsack) > 0 else None
