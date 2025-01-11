import numpy as np


class Perceptron(object):
    """Klasyfikator - perceptron.

    Parametry
    ------------
    eta : zmiennoprzecinkowy
        Współczynnik uczenia (w przedziale pomiędzy 0.0 a 1.0)
    n_iter : liczba całkowita
        Liczba przebiegów po zestawach uczących.

    Atrybuty
    -----------
    w_ : jednowymiarowa tablica
        Wagi po dopasowaniu.
    errors_ : lista
        Liczba nieprawidłowych klasyfikacji w każdej epoce.

    """
    def __init__(self, eta=0.01, n_iter=10):
        pass

    def fit(self, X, y):
        """Dopasowanie danych uczących.

        Parametry
        ----------
        X : {tablicopodobny}, wymiary = [n_próbek, n_cech]
            Wektory uczące, gdzie n_próbek
            oznacza liczbę próbek, a
            n_cech — liczbę cech.
        y : tablicopodobny, wymiary = [n_próbek]
            Wartości docelowe.

        Zwraca
        -------
        self : obiekt

        UWAGA: wykorzystuje metodę predict

        """
       pass

    def net_input(self, X):
        """Oblicza całkowite pobudzenie sieci
        """
        pass

    def predict(self, X):
        """Zwraca etykietę klas po obliczeniu funkcji skoku jednostkowego
        UWAGA: Wykorzystuje metodę net_input
        """
        pass