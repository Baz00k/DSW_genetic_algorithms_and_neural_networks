import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Importowanie danych
data = pd.read_csv('data.csv')

# Usunięcie kolumny 'id' i zakodowanie kolumny 'diagnosis'
data.drop(columns=['id'], inplace=True)
data['diagnosis'] = data['diagnosis'].map({'M': 0, 'B': 1})

print(data.info())  # Wyświetl informacje o danych
print(data.head())  # Wyświetl pierwsze kilka wierszy

# Sprawdzenie i imputacja wartości NaN
if data.isnull().values.any():
    print("Wykryto wartości NaN. Zastępowanie średnią...")
    data.fillna(data.mean(), inplace=True)

if data.empty:
    raise ValueError("Dane wejściowe są puste. Upewnij się, że plik 'data.csv' zawiera dane.")

# Skalowanie cech do zakresu 0-1
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data.drop(columns=['diagnosis']))
X = scaled_features
y = data['diagnosis'].to_numpy()

# Funkcja do przeprowadzania badań i tworzenia wykresów
def evaluate_mlp(hidden_layer_sizes, activation='relu', solver='adam', title=''):
    scores = []
    for layers in hidden_layer_sizes:
        mlp = MLPClassifier(hidden_layer_sizes=layers, activation=activation, solver=solver, max_iter=1000, random_state=42)
        score = cross_val_score(mlp, X, y, cv=10, scoring="balanced_accuracy").mean()
        scores.append(score)

    # Wykres wyników
    plt.bar(range(len(hidden_layer_sizes)), scores, tick_label=[str(l) for l in hidden_layer_sizes])
    plt.title(title)
    plt.xlabel('Rozmiar warstw ukrytych')
    plt.ylabel('Balanced Accuracy')
    plt.show()

# Badanie wpływu liczby neuronów w warstwach ukrytych
hidden_layer_configs = [(30,), (30, 30), (60,), (60, 60), (100,), (50, 50, 50)]
evaluate_mlp(hidden_layer_configs, title='Wpływ liczby neuronów na wyniki')

# Badanie wpływu funkcji aktywacji
best_hidden_layer = (60, 60)
activation_functions = ['identity', 'logistic', 'tanh', 'relu']
for activation in activation_functions:
    evaluate_mlp([best_hidden_layer], activation=activation, title=f'Funkcja aktywacji: {activation}')

# Badanie wpływu solvera
best_activation = 'relu'
solvers = ['lbfgs', 'sgd', 'adam']
for solver in solvers:
    evaluate_mlp([best_hidden_layer], activation=best_activation, solver=solver, title=f'Solver: {solver}')
