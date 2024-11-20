from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


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
    )
    generations: int = Field(
        description="Number of generations to run the algorithm for",
        examples=[10, 20, 50],
        default=10,
    )
    mutation_rate: float = Field(
        description="Probability of a mutation occurring",
        examples=[0.1, 0.2, 0.5],
        default=0.1,
    )
