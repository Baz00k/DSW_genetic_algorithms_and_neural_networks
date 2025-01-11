import numpy as np
import matplotlib.pyplot as plt

from load_data import get_iris_binary_data


class Perceptron:
    """Klasyfikator - perceptron."""

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = None
        self.errors_ = []

    def fit(self, x, y):
        """Dopasowanie danych uczących."""
        self.w_ = np.zeros(1 + x.shape[1])  # Wektor wag z dodatkowym elementem na bias
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(x, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update  # Aktualizacja biasu
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, x):
        """Oblicza całkowite pobudzenie sieci"""
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def predict(self, x):
        """Zwraca etykietę klas po obliczeniu funkcji skoku jednostkowego"""
        return np.where(self.net_input(x) >= 0.0, 1, -1)


# Get preprocessed data
x, y = get_iris_binary_data()

# Visualization of data points
colors = ["red" if label == -1 else "blue" for label in y]
plt.scatter(x[:, 0], x[:, 1], c=colors, marker="o")
plt.xlabel("Długość działki")
plt.ylabel("Długość płatka")
plt.title("Zbiór danych Iris")
plt.show()

# Initialize perceptron with different learning rates
eta_values = [0.1, 0.01, 0.001]
results = {}

for eta in eta_values:
    perceptron = Perceptron(eta=eta, n_iter=10)
    perceptron.fit(x, y)
    results[eta] = perceptron.errors_

# Plot misclassifications
for eta, errors in results.items():
    plt.plot(range(1, len(errors) + 1), errors, label=f"eta={eta}")

plt.xlabel("Epoki")
plt.ylabel("Liczba błędów")
plt.legend()
plt.title("Liczba błędnych klasyfikacji w kolejnych epokach")
plt.show()
