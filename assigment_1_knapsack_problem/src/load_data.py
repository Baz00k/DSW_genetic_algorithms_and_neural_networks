import os
import pandas as pd

def load_data(file_name: str) -> tuple[int, int, pd.DataFrame]:
    """
    Load data for knapsack problem from a file

    Args:
        file_name: Name of the file to load data from

    Returns:
        A tuple containing the number of items, the capacity of the knapsack, and a DataFrame containing the weight and value of each item
    """

    data_path = os.path.join("data", file_name)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find file at {data_path}")

    # First line contains the number of items and the capacity of the knapsack
    with open(data_path, "r") as f:
        num_items, capacity = map(int, f.readline().split())

    # Load the data into a DataFrame
    # Each row contains the weight and value of an item
    data = pd.read_csv(data_path, skiprows=1, delimiter=" ", header=None, names=["weight", "value"])

    return num_items, capacity, data
