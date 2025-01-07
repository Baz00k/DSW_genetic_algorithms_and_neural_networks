import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron(object):
    """Klasyfikator - perceptron."""
    
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        self.w_ = None
        self.errors_ = []

    def fit(self, X, y):
        """Dopasowanie danych uczących."""
        self.w_ = np.zeros(1 + X.shape[1])  # Wektor wag z dodatkowym elementem na bias
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update  # Aktualizacja biasu
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Oblicza całkowite pobudzenie sieci"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Zwraca etykietę klas po obliczeniu funkcji skoku jednostkowego"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
    
    # Wczytaj dane
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['Długość działki', 'Szerokość działki', 'Długość płatka', 'Szerokość płatka', 'Etykieta']
data = pd.read_csv(url, header=None, names=columns)

# Ograniczenie do dwóch klas
data = data[data['Etykieta'].isin(['Iris-setosa', 'Iris-versicolor'])]
data['Etykieta'] = data['Etykieta'].map({'Iris-setosa': -1, 'Iris-versicolor': 1})

# Wybór dwóch cech
X = data[['Długość działki', 'Długość płatka']].values
y = data['Etykieta'].values

colors = ['red' if label == -1 else 'blue' for label in y]

plt.scatter(X[:, 0], X[:, 1], c=colors, marker='o')
plt.xlabel('Długość działki')
plt.ylabel('Długość płatka')
plt.title('Zbiór danych Iris')
plt.show()

# Inicjalizacja perceptronu
eta_values = [0.1, 0.01, 0.001]
results = {}

for eta in eta_values:
    perceptron = Perceptron(eta=eta, n_iter=10)
    perceptron.fit(X, y)
    results[eta] = perceptron.errors_

# Wykres liczby błędnych klasyfikacji
for eta, errors in results.items():
    plt.plot(range(1, len(errors) + 1), errors, label=f'eta={eta}')

plt.xlabel('Epoki')
plt.ylabel('Liczba błędów')
plt.legend()
plt.title('Liczba błędnych klasyfikacji w kolejnych epokach')
plt.show()