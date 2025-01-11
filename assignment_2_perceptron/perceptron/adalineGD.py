import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from load_data import get_iris_binary_data


class AdalineGD:
    """Klasyfikator — ADAptacyjny LIniowy NEuron."""

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = None
        self.cost_ = []

    def fit(self, x, y):
        """Trenowanie za pomocą danych uczących."""
        self.w_ = np.zeros(1 + x.shape[1])
        self.cost_ = []

        for _ in range(self.n_iter):
            net_input = self.net_input(x)
            output = self.activation(net_input)
            errors = y - output
            self.w_[1:] += self.eta * x.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, x):
        """Oblicza całkowite pobudzenie."""
        return np.dot(x, self.w_[1:]) + self.w_[0]

    def activation(self, x):
        """Funkcja aktywacji."""
        return x

    def predict(self, x):
        """Zwraca etykietę klas po wykonaniu skoku jednostkowego."""
        return np.where(self.net_input(x) >= 0.0, 1, -1)


def plot_decision_regions(x, y, classifier, resolution=0.02):
    markers = ("o", "x")
    colors = ("red", "blue")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    # Rysowanie powierzchni decyzyjnej
    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Rysowanie punktów
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=x[y == cl, 0],
            y=x[y == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=f"Klasa {cl}",
        )


# Get preprocessed data
x, y = get_iris_binary_data()

# Training with different eta values
eta_values = [0.0001, 0.1]
results = {}

for eta in eta_values:
    adaline = AdalineGD(eta=eta, n_iter=10)
    adaline.fit(x, y)
    results[eta] = adaline.cost_

# Plot cost
for eta, cost in results.items():
    plt.plot(range(1, len(cost) + 1), cost, label=f"eta={eta}")

plt.xlabel("Epoki")
plt.ylabel("Koszt")
plt.legend()
plt.title("Koszt w kolejnych epokach dla różnych eta")
plt.show()

# Plot decision boundaries
for eta in eta_values:
    adaline = AdalineGD(eta=eta, n_iter=10)
    adaline.fit(x, y)
    plot_decision_regions(x, y, classifier=adaline)
    plt.title(f"Granice decyzyjne dla eta={eta}")
    plt.xlabel("Długość działki")
    plt.ylabel("Długość płatka")
    plt.legend()
    plt.show()
