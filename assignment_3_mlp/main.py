import warnings
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

# Suppress convergence warnings
warnings.filterwarnings("ignore")

# Create output directory if it doesn't exist
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and preprocess data
df = pd.read_csv("data.csv")
df = df.drop(["id", "Unnamed: 32"], axis=1, errors="ignore")
df = df.dropna(axis=1)
df["diagnosis"] = df["diagnosis"].map({"M": 0, "B": 1})

# Scale features
scaler = MinMaxScaler()
X = df.drop("diagnosis", axis=1)
y = df["diagnosis"].to_numpy()
X_scaled = scaler.fit_transform(X)


def evaluate_model(mlp_model, X, y, cv=5):
    scores = cross_val_score(mlp_model, X, y, cv=cv, scoring="balanced_accuracy")
    return scores.mean(), scores.std()


def plot_results(results, title, labels=None):
    plt.figure(figsize=(10, 6))
    means = [r[0] for r in results]
    stds = [r[1] for r in results]
    plt.bar(range(len(results)), means, yerr=stds, capsize=5)
    plt.title(title)
    plt.ylabel("Balanced Accuracy")

    if labels:
        plt.xticks(range(len(labels)), [str(label) for label in labels], rotation=45)

    for i, (mean, std) in enumerate(results):
        plt.text(i, mean, f"{mean:.3f}\n±{std:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    return plt


# Configure model parameters
base_params = {
    "random_state": 42,
    "max_iter": 2000,
}

# Test configurations
configs = {
    "hidden_layer_sizes": [(30,), (30, 30), (60,), (60, 60), (100,), (80,), (80, 80)],
    "activation": ["identity", "logistic", "tanh", "relu"],
    "solver": ["lbfgs", "sgd", "adam"],
}

# Test hidden layer sizes
size_results = []
for size in configs["hidden_layer_sizes"]:
    model = MLPClassifier(hidden_layer_sizes=size, **base_params)
    size_results.append(evaluate_model(model, X_scaled, y))

plot_results(
    size_results, "Hidden Layer Size Impact", configs["hidden_layer_sizes"]
).savefig(os.path.join(OUTPUT_DIR, "hidden_layer_sizes.png"))
plt.close()

# Test activation functions with best size
best_size = configs["hidden_layer_sizes"][np.argmax([r[0] for r in size_results])]
activation_results = []

for activation in configs["activation"]:
    model = MLPClassifier(
        hidden_layer_sizes=best_size, activation=activation, **base_params
    )
    activation_results.append(evaluate_model(model, X_scaled, y))

plot_results(
    activation_results, "Activation Function Impact", configs["activation"]
).savefig(os.path.join(OUTPUT_DIR, "activation_functions.png"))
plt.close()

# Test solvers with best activation
best_activation = configs["activation"][np.argmax([r[0] for r in activation_results])]
solver_results = []

for solver in configs["solver"]:
    model = MLPClassifier(
        hidden_layer_sizes=best_size,
        activation=best_activation,
        solver=solver,
        **base_params,
    )
    solver_results.append(evaluate_model(model, X_scaled, y))

plot_results(solver_results, "Solver Impact", configs["solver"]).savefig(
    os.path.join(OUTPUT_DIR, "solvers.png")
)
plt.close()
