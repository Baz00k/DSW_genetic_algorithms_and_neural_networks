import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class AdalineGD(object):
    """Klasyfikator — ADAptacyjny LIniowy NEuron."""

    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = None
        self.cost_ = []

    def fit(self, X, y):
        """Trenowanie za pomocą danych uczących."""
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        
        for _ in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Oblicza całkowite pobudzenie."""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Funkcja aktywacji."""
        return X

    def predict(self, X):
        """Zwraca etykietę klas po wykonaniu skoku jednostkowego."""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

def load_data(url):
    """Wczytuje i przetwarza dane z podanego URL."""
    columns = ['Długość działki', 'Szerokość działki', 'Długość płatka', 'Szerokość płatka', 'Etykieta']
    data = pd.read_csv(url, header=None, names=columns)
    data = data[data['Etykieta'].isin(['Iris-setosa', 'Iris-versicolor'])]
    data['Etykieta'] = data['Etykieta'].map({'Iris-setosa': -1, 'Iris-versicolor': 1})
    return data

# Wczytywanie danych
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
data = load_data(url)

# Dane uczące
X = data[['Długość działki', 'Długość płatka']].values
y = data['Etykieta'].values

# Trening z różnymi wartościami eta
eta_values = [0.0001, 0.1]
results = {}

for eta in eta_values:
    adaline = AdalineGD(eta=eta, n_iter=10)
    adaline.fit(X, y)
    results[eta] = adaline.cost_

# Wykres liczby błędnych klasyfikacji (kosztu)
for eta, cost in results.items():
    plt.plot(range(1, len(cost) + 1), cost, label=f'eta={eta}')

plt.xlabel('Epoki')
plt.ylabel('Koszt')
plt.legend()
plt.title('Koszt w kolejnych epokach dla różnych eta')
plt.show()

# Wizualizacja granic decyzyjnych
def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('o', 'x')
    colors = ('red', 'blue')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # Rysowanie powierzchni decyzyjnej
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Rysowanie punktów
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=f"Klasa {cl}"
        )

# Rysowanie granic decyzyjnych
for eta in eta_values:
    adaline = AdalineGD(eta=eta, n_iter=10)
    adaline.fit(X, y)
    plot_decision_regions(X, y, classifier=adaline)
    plt.title(f'Granice decyzyjne dla eta={eta}')
    plt.xlabel('Długość działki')
    plt.ylabel('Długość płatka')
    plt.legend()
    plt.show()
