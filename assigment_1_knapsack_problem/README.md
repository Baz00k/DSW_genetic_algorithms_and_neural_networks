# Assigment 1 - KnapSack Problem

## Problem Statement

You are given a list of items, where each item has a weight and a value.
You are also given a knapsack that can only hold a certain amount of weight.
Create a genetic algorithm to find the best combination of items to maximize the value of the knapsack without exceeding the weight capacity.

You can check the project requirements in the [PDF file](./docs/AG.pdf).

## How to run the code

### Create a virtual environment

```bash
python3 -m venv .venv

*or

python -m venv .venv

*and after:

source .venv/bin/activate

*or

source .venv/Scripts/activate

```

### Install the requirements

```bash
pip install -r requirements.txt
```

### Run the code

```bash
python -m knapsack_cli --<parameter_name>=<parameter_value>
```

Example:

```bash
python -m knapsack_cli --population_size=100 --generations=100 --mutation_rate=0.001 --input_file_name=low-dimensional/f10_l-d_kp_20_879
```

You can check the available parameters by running:

```bash
python -m knapsack_cli --help
```
