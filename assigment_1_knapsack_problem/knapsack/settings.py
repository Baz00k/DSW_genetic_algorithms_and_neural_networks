from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

from .knapsack_ga_solver import KnapsackGASolver


class Settings(BaseSettings):
    # Enable parsing CLI arguments for settings
    model_config = SettingsConfigDict(
        cli_parse_args=True,
    )

    input_file_name: str = Field(
        description="The name of the input file with the data",
        examples=["low-dimensional/f1_l-d_kp_10_269"],
    )
    population_size: int = Field(
        description="Number of individuals in the population",
        examples=[10, 20, 50],
        default=10,
        ge=2,
    )
    generations: int = Field(
        description="Number of generations to run the algorithm for",
        examples=[10, 20, 50],
        default=10,
        ge=1,
    )
    crossover_type: KnapsackGASolver.CrossoverType = Field(
        description="Type of crossover to use",
        examples=[
            KnapsackGASolver.CrossoverType.SINGLE_POINT,
            KnapsackGASolver.CrossoverType.TWO_POINT,
        ],
        default=KnapsackGASolver.CrossoverType.SINGLE_POINT,
    )
    mutation_rate: float = Field(
        description="Probability of a mutation occurring",
        examples=[0.1, 0.2, 0.5],
        default=0.1,
        ge=0.0,
    )
    selection_type: KnapsackGASolver.SelectionType = Field(
        description="Type of selection to use",
        examples=[
            KnapsackGASolver.SelectionType.ROULETTE_WHEEL,
            KnapsackGASolver.SelectionType.TOURNAMENT,
        ],
        default=KnapsackGASolver.SelectionType.ROULETTE_WHEEL,
    )
    fitness_type: KnapsackGASolver.FitnessType = Field(
        description="Type of fitness function to use",
        examples=[
            KnapsackGASolver.FitnessType.LINEAR,
            KnapsackGASolver.FitnessType.EXPONENTIAL,
        ],
        default=KnapsackGASolver.FitnessType.LINEAR,
    )
