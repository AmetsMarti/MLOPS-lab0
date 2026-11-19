import ast
import json
from typing import Any

import click

try:
    from . import preprocessing
except ImportError:
    import preprocessing


def _parse_scalar(raw: str) -> Any:
    text = raw.strip()
    lowered = text.lower()
    if lowered in ("none", "null"):
        return None
    if lowered == "nan":
        return float("nan")
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text


def _parse_list(raw: str) -> list:
    text = raw.strip()
    if text == "":
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            value = parser(text)
        except Exception:
            continue
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]
    return [_parse_scalar(part) for part in text.split(",")]


@click.group()
def cli():
    """General CLI group for data preprocessing utilities."""
    pass


@cli.group(help="Functions related to data cleaning.")
def clean():
    """Functions related to data cleaning."""
    pass


@cli.group(help="Functions related to dealing with numerical attributes.")
def numeric():
    """Functions related to dealing with numerical attributes."""
    pass


@cli.group(help="Functions to deal with textual information.")
def text():
    """Functions to deal with textual information."""
    pass


@cli.group(help="Functions related to the structure of data.")
def struct():
    """Functions related to the structure of data."""
    pass


@clean.command("remove-missing")
@click.argument("values")
def clean_remove_missing(values):
    """Remove missing values from a list of string values."""
    click.echo(preprocessing.remove_missing(_parse_list(values)))


@clean.command("fill-missing")
@click.argument("values")
@click.option(
    "--value",
    "--filler",
    "filler",
    default="0",
    show_default=True,
    help="Value used to fill missing entries.",
)
def clean_fill_missing(values, filler):
    """Fill missing values in a list of string values."""
    click.echo(preprocessing.fill_missing(_parse_list(values), _parse_scalar(filler)))


@numeric.command("normalize")
@click.argument("values")
@click.option(
    "--min",
    "new_min",
    type=float,
    default=0.0,
    show_default=True,
    help="Minimum value of the new range.",
)
@click.option(
    "--max",
    "new_max",
    type=float,
    default=1.0,
    show_default=True,
    help="Maximum value of the new range.",
)
def numeric_normalize(values, new_min, new_max):
    """Normalize numerical values to a new range."""
    click.echo(preprocessing.normalize_min_max(_parse_list(values), new_min, new_max))


@numeric.command("standardize")
@click.argument("values")
def numeric_standardize(values):
    """Standardize numerical values."""
    click.echo(preprocessing.standardize_z_score(_parse_list(values)))


@numeric.command("clip")
@click.argument("values")
@click.option(
    "--min",
    "minimum",
    type=float,
    default=0.0,
    show_default=True,
    help="Minimum value to clip.",
)
@click.option(
    "--max",
    "maximum",
    type=float,
    default=1.0,
    show_default=True,
    help="Maximum value to clip.",
)
def numeric_clip(values, minimum, maximum):
    """Clip numerical values to a range."""
    click.echo(preprocessing.clip_values(_parse_list(values), minimum, maximum))


@numeric.command("convert-to-integers")
@click.argument("values")
def numeric_convert_to_integers(values):
    """Convert values to integers."""
    click.echo(preprocessing.convert_to_integers(_parse_list(values)))


@numeric.command("log-transform")
@click.argument("values")
def numeric_log_transform(values):
    """Transform values to logarithmic scale."""
    click.echo(preprocessing.log_transform(_parse_list(values)))


@text.command("tokenize")
@click.argument("text_value")
def text_tokenize(text_value):
    """Tokenize text into tokens."""
    click.echo(preprocessing.tokenize(text_value))


@text.command("remove-punctuation")
@click.argument("text_value")
def text_remove_punctuation(text_value):
    """Remove punctuation from text."""
    click.echo(preprocessing.remove_punctuation(text_value))


@text.command("remove-stop-words")
@click.argument("text_value")
@click.option(
    "--stop-words",
    "--stopwords",
    "stop_words",
    default="[]",
    show_default=True,
    help="List of strings conforming the stop-words to remove.",
)
def text_remove_stop_words(text_value, stop_words):
    """Remove stop-words from text."""
    click.echo(preprocessing.remove_stop_words(text_value, _parse_list(stop_words)))


@struct.command("shuffle")
@click.argument("values")
@click.option(
    "--seed",
    type=int,
    default=None,
    show_default=True,
    help="Integer value of the seed.",
)
def struct_shuffle(values, seed):
    """Shuffle a list of values."""
    click.echo(preprocessing.shuffle(_parse_list(values), seed))


@struct.command("flatten")
@click.argument("values")
def struct_flatten(values):
    """Flatten a list of lists."""
    click.echo(preprocessing.flatten(_parse_list(values)))


@struct.command("unique-values")
@click.argument("values")
def struct_unique_values(values):
    """Return unique values from a list."""
    click.echo(preprocessing.unique_values(_parse_list(values)))


# Aliases for convenience/compatibility
clean.add_command(clean_remove_missing, name="remove_missing")
clean.add_command(clean_fill_missing, name="fill_missing")
numeric.add_command(numeric_convert_to_integers, name="to-integers")
numeric.add_command(numeric_convert_to_integers, name="to_integers")
numeric.add_command(numeric_convert_to_integers, name="conversion-to-integers")
numeric.add_command(numeric_convert_to_integers, name="conversion_to_integers")
numeric.add_command(numeric_log_transform, name="log-scale")
numeric.add_command(numeric_log_transform, name="log_scale")
numeric.add_command(numeric_log_transform, name="transformation-to-logarithmic-scale")
numeric.add_command(numeric_log_transform, name="transformation_to_logarithmic_scale")
text.add_command(text_tokenize, name="tokenization")
text.add_command(text_remove_punctuation, name="remove_punctuation")
text.add_command(text_remove_stop_words, name="remove_stop_words")
struct.add_command(struct_shuffle, name="shuffle-values")
struct.add_command(struct_shuffle, name="shuffle_values")
struct.add_command(struct_flatten, name="flatten-list")
struct.add_command(struct_flatten, name="flatten_list")
struct.add_command(struct_unique_values, name="unique")
struct.add_command(struct_unique_values, name="unique_values")


if __name__ == "__main__":
    cli()
