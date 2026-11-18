import math
import random
import re
import statistics
import string
from typing import Any


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def remove_missing(values: list) -> list:
    return [value for value in values if not _is_missing(value)]


def fill_missing(values: list, filler: Any = 0) -> list:
    return [filler if _is_missing(value) else value for value in values]


def remove_duplicates(values: list) -> list:
    unique = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def unique_values(values: list) -> list:
    return remove_duplicates(values)


def normalize_min_max(values: list, new_min: float = 0.0, new_max: float = 1.0) -> list:
    numbers = [float(v) for v in values]
    minimum = min(numbers)
    maximum = max(numbers)
    if minimum == maximum:
        return [new_min for _ in numbers]
    return [
        (value - minimum) / (maximum - minimum) * (new_max - new_min) + new_min
        for value in numbers
    ]


def standardize_z_score(values: list) -> list:
    numbers = [float(v) for v in values]
    mean = statistics.fmean(numbers)
    deviation = statistics.pstdev(numbers)
    if deviation == 0:
        return [0.0 for _ in numbers]
    return [(value - mean) / deviation for value in numbers]


def clip_values(values: list, minimum: float = 0.0, maximum: float = 1.0) -> list:
    if minimum > maximum:
        raise ValueError("minimum must not be greater than maximum")
    numbers = [float(v) for v in values]
    return [max(minimum, min(value, maximum)) for value in numbers]


def convert_to_integers(values: list) -> list:
    integers = []
    for value in values:
        try:
            integers.append(int(value))
        except TypeError, ValueError:
            try:
                integers.append(int(float(value)))
            except TypeError, ValueError:
                continue
    return integers


def log_transform(values: list) -> list:
    transformed = []
    for value in values:
        try:
            number = float(value)
        except TypeError, ValueError:
            continue
        if number > 0:
            transformed.append(math.log(number))
    return transformed


def tokenize(text: str) -> list:
    return re.findall(r"[^\W_]+", text.lower())


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def select_alphanumeric_and_spaces(text: str) -> str:
    return "".join(char for char in text if char.isalnum() or char.isspace())


def remove_stop_words(text: str, stop_words: list) -> str:
    lowered_stop_words = {str(word).lower() for word in stop_words}
    return " ".join(
        word for word in text.lower().split() if word not in lowered_stop_words
    )


def flatten(values: list) -> list:
    return [value for sublist in values for value in sublist]


def shuffle(values: list, seed: int | None = None) -> list:
    shuffled = list(values).copy()
    random.Random(seed).shuffle(shuffled)
    return shuffled
