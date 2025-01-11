import pandas as pd

IRIS_DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
)


def load_iris_data(url=IRIS_DATASET_URL):
    """Load iris dataset from URL."""
    columns = [
        "Długość działki",
        "Szerokość działki",
        "Długość płatka",
        "Szerokość płatka",
        "Etykieta",
    ]
    return pd.read_csv(url, header=None, names=columns)


def prepare_binary_classification_data(data, features=None):
    """Prepare data for binary classification."""

    label_setosa = "Iris-setosa"
    label_virginica = "Iris-virginica"

    binary_data = data[data["Etykieta"].isin([label_setosa, label_virginica])].copy()

    # Map classes to -1 and 1
    binary_data["Etykieta"] = binary_data["Etykieta"].map(
        {label_setosa: -1, label_virginica: 1}
    )

    if features is None:
        features = ["Długość działki", "Długość płatka"]

    x = binary_data[features].values
    y = binary_data["Etykieta"].values

    return x, y


def get_iris_binary_data(features=None):
    """Convenience function to get preprocessed iris data for binary classification."""
    data = load_iris_data()
    return prepare_binary_classification_data(data, features=features)
