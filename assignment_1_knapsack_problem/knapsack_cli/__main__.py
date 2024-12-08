import sys
from pydantic import ValidationError

from knapsack.load_data import load_data
from knapsack.knapsack_ga_solver import KnapsackGASolver

from .settings import Settings
from .plotting import plot_fitness_histories


def solve_single(settings: Settings):
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

        if settings.enable_plots:
            plot_fitness_histories([solver.fitness_history], ["Solution"])
    else:
        print("No valid solution found, try again with different settings")


def compare_solvers(settings: Settings):
    num_items, capacity, data = load_data(settings.input_file_name)

    print(
        (
            f"Solving knapsack problem with {num_items} items and a capacity of {capacity}\n"
            f"Using settings:\n"
            f"Population size: {settings.population_size}\n"
            f"Generations: {settings.generations}\n"
            f"Comparing solvers with different settings"
        )
    )

    solutions = []
    fitness_histories = []
    labels = []

    for crossover_type in KnapsackGASolver.CrossoverType:
        for selection_type in KnapsackGASolver.SelectionType:
            for fitness_type in KnapsackGASolver.FitnessType:
                solver = KnapsackGASolver(
                    data=data,
                    max_capacity=capacity,
                    population_size=settings.population_size,
                    generations=settings.generations,
                    crossover_type=crossover_type,
                    mutation_rate=settings.mutation_rate,
                    selection_type=selection_type,
                    fitness_type=fitness_type,
                )

                solution = solver.solve()

                if solution is not None:
                    solutions.append(solution)

                    if settings.enable_plots:
                        fitness_histories.append(solver.fitness_history)
                        labels.append(
                            f"{crossover_type} - {selection_type} - {fitness_type}"
                        )

    print("Best solutions found:")
    for solution, label in zip(solutions, labels):
        print(f"{label}: {solution}")

    if settings.enable_plots:
        plot_fitness_histories(fitness_histories, labels)


def main():
    try:
        settings = Settings()
    except ValidationError as e:
        sys.exit(f"Invalid configuration: {e}")

    if settings.compare_solvers:
        compare_solvers(settings)
    else:
        solve_single(settings)


if __name__ == "__main__":
    main()
