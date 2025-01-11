import numpy as np
import matplotlib.pyplot as plt

class AdalineSGD:
    """Klasyfikator — ADAptacyjny LIniowy NEuron.

    Parametry
    ------------
    eta : zmiennoprzecinkowy
        Współczynnik uczenia (w zakresie pomiędzy 0.0 i 1.0)
    n_iter : liczba całkowita
        Liczba przebiegów po zestawie uczącym.
    shuffle : wartość boolowska (domyślnie: True)
        Jeżeli jest ustalona wartość True,
        tasuje dane uczące przed każdą epoką w celu zapobiegnięcia cykliczności.
    random_state : liczba całkowita (domyślnie: None)
        Ustanawia przypadkowy stan dla operacji tasowania
        oraz inicjacji wag.
    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.random_state = random_state
        if random_state:
            np.random.seed(random_state)

    def fit(self, X, y):
        """ Dopasowanie danych uczących. """
        self._initialize_weights(X.shape[1])
        self.errors_ = []
        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            errors = 0
            for xi, target in zip(X, y):
                update = self._update_weights(xi, target)
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def partial_fit(self, X, y):
        """ Dopasowanie danych uczących bez ponownej inicjacji wag. """
        if not hasattr(self, "w_"):
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] == 1:
            y = [y]
        for xi, target in zip(X, y):
            self._update_weights(xi, target)
        return self

    def _shuffle(self, X, y):
        """ Tasuje dane uczące. """
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """ Inicjuje wagi przydzielając im wartości zerowe. """
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """ Aktualizuje wagi wykorzystując regułę uczenia Adaline. """
        output = self.net_input(xi)
        error = target - output
        self.w_[1:] += self.eta * xi * error
        self.w_[0] += self.eta * error
        return error

    def net_input(self, X):
        """ Oblicza całkowite pobudzenie. """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """ Oblicza liniową funkcję aktywacji. """
        return self.net_input(X)

    def predict(self, X):
        """ Zwraca etykietę klas po wykonaniu skoku jednostkowego. """
        return np.where(self.activation(X) >= 0.0, 1, -1)
    
# Załóżmy, że dane X i y są już zaimportowane
adaline = AdalineSGD(eta=0.01, n_iter=15, random_state=1)
adaline.fit(X, y)

# Rysowanie wykresu liczby błędnych klasyfikacji
plt.plot(range(1, len(adaline.errors_) + 1), adaline.errors_, marker='o')
plt.xlabel('Epoki')
plt.ylabel('Liczba błędów klasyfikacji')
plt.title('Adaline - liczba błędów klasyfikacji w zależności od epoki')
plt.show()

