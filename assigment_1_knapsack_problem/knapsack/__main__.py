from pydantic import ValidationError

from .settings import Settings
from .load_data import load_data


def main():
    try:
        settings = Settings()
    except ValidationError as e:
        exit(f"Invalid configuration: {e}")

    num_items, capacity, data = load_data(settings.input_file_name)


if __name__ == "__main__":
    main()
