import sys
from pydantic import ValidationError

from .settings import Settings
from .load_data import load_data
from .knapsack_ga_solver import KnapsackGASolver


def main():
    try:
        settings = Settings()
    except ValidationError as e:
        sys.exit(f"Invalid configuration: {e}")

    num_items, capacity, data = load_data(settings.input_file_name)

    print(
        (
            f"Solving knapsack problem with {num_items} items and a capacity of {capacity}\n"
            f"Using settings:\n"
            f"Population size: {settings.population_size}\n"
            f"Generations: {settings.generations}\n"
            f"Crossover type: {settings.crossover_type}\n"
            f"Mutation rate: {settings.mutation_rate}\n"
            f"Selection type: {settings.selection_type}\n"
            f"Fitness type: {settings.fitness_type}\n"
        )
    )

    solver = KnapsackGASolver(
        data=data,
        max_capacity=capacity,
        population_size=settings.population_size,
        generations=settings.generations,
        crossover_type=settings.crossover_type,
        mutation_rate=settings.mutation_rate,
        selection_type=settings.selection_type,
        fitness_type=settings.fitness_type,
    )

    solution = solver.solve()

    if solution is not None:
        print(f"Best solution found: {solution}")
    else:
        print("No valid solution found, try again with different settings")


if __name__ == "__main__":
    main()
