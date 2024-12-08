import matplotlib.pyplot as plt


def plot_fitness_histories(fitness_histories: list, labels: list):
    """Plot multiple fitness histories on a single plot"""
    for fitness_history, label in zip(fitness_histories, labels):
        plt.plot(fitness_history, label=label)

    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness over Generations")
    plt.legend()
    plt.show()
