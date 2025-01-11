class AdalineSGD(object):
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
        Liczba nieprawidłowych klasyfikacji w każdej epoce.
    shuffle : wartość boolowska (domyślnie: True)
        Jeżeli jest ustalona wartość True,
        tasuje dane uczące przed każdą epoką w celu zapobiegnięcia cykliczności.
    random_state : liczba całkowita (domyślnie: None)
        Ustanawia przypadkowy stan dla operacji tasowania
        oraz inicjacji wag.
        
    """
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        pass

    def fit(self, X, y):
        """ Dopasowanie danych uczących.

        Parametry
        ----------
        X : {tablicopodobny}, wymiary = [n_próbek, n_cech]
            Wektory uczące, gdzie n_próbek
            oznacza liczbę próbek, a
            n_cech określa liczbę cech.
        y : tablicopodobny, wymiary = [n_próbek]
            Wartości docelowe.

        Zwraca
        -------
        self : obiekt

        """
        pass

    def partial_fit(self, X, y):
        """dopasowuje dane uczące bez ponownej inicjacji wag"""
        pass

    def _shuffle(self, X, y):
        """tasuje dane uczące"""
        pass
    
    def _initialize_weights(self, m):
        """inicjuje wagi przydzielając im wartości zerowe"""
        pass
        
    def _update_weights(self, xi, target):
        """wykorzystuje regułę uczenia Adaline do aktualizacji wag"""
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