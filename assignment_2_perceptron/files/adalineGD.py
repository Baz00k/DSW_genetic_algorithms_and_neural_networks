class AdalineGD(object):
    """Klasyfikator — ADAptacyjny LIniowy NEuron.

    Parametry
    ------------
    eta : zmiennoprzecinkowy
        Współczynnik uczenia (w zakresie pomiędzy 0.0 i 1.0)
    n_iter : liczba całkowita
        Liczba przebiegów po zestawie uczącym.

    Atrybuty
    -----------
   w_ : jednowymiarowa tablica
        Wagi po dopasowaniu.
    errors_ : lista
        Liczba niewłaściwych klasyfikacji w każdej epoce.

    """
    def __init__(self, eta=0.01, n_iter=50):
        pass

    def fit(self, X, y):
        """ Trenowanie za pomocą danych uczących.

        Parametry
        ----------
        X : {tablicopodobny}, wymiary = [n_próbek, n_cech]
            Wektory uczenia,
            gdzie n_próbek oznacza liczbę próbek, a
            n_cech – liczbę cech.
        y : tablicopodobny, wymiary = [n_próbek]
            Wartości docelowe.

        Zwraca
        -------
        self : obiekt

        """
        pass

    def net_input(self, X):
        """oblicza całkowite pobudzenie"""
        pass

    def activation(self, X):
        """oblicza liniową funkcję aktywacji"""
        pass

    def predict(self, X):
        """zwraca etykietę klas po wykonaniu skoku jednostkowego"""
        pass