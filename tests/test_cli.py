import ast
import math

import pytest
from click.testing import CliRunner

try:
    from src.cli import cli
except ImportError:
    from src.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


def parse_output(output):
    text = output.strip()
    if text == "":
        return text
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def test_clean_remove_missing(runner):
    result = runner.invoke(cli, ["clean", "remove-missing", '[1, null, "", "a"]'])
    assert result.exit_code == 0
    assert parse_output(result.output) == [1, "a"]


def test_clean_fill_missing(runner):
    result = runner.invoke(
        cli, ["clean", "fill-missing", '[1, null, ""]', "--value", "0"]
    )
    assert result.exit_code == 0
    assert parse_output(result.output) == [1, 0, 0]


def test_numeric_normalize(runner):
    result = runner.invoke(
        cli, ["numeric", "normalize", "[1, 2, 3]", "--min", "0", "--max", "1"]
    )
    assert result.exit_code == 0
    assert parse_output(result.output) == pytest.approx([0.0, 0.5, 1.0])


def test_numeric_standardize(runner):
    result = runner.invoke(cli, ["numeric", "standardize", "[1, 2, 3, 4]"])
    assert result.exit_code == 0
    assert parse_output(result.output) == pytest.approx(
        [
            -1.3416407864998738,
            -0.4472135954999579,
            0.4472135954999579,
            1.3416407864998738,
        ]
    )


def test_numeric_clip(runner):
    result = runner.invoke(
        cli, ["numeric", "clip", "[1, 5, 10]", "--min", "2", "--max", "8"]
    )
    assert result.exit_code == 0
    assert parse_output(result.output) == pytest.approx([2.0, 5.0, 8.0])


def test_numeric_convert_to_integers(runner):
    result = runner.invoke(cli, ["numeric", "convert-to-integers", '["1", "x", "2.7"]'])
    assert result.exit_code == 0
    assert parse_output(result.output) == [1, 2]


def test_numeric_log_transform(runner):
    result = runner.invoke(cli, ["numeric", "log-transform", "[1, 10]"])
    assert result.exit_code == 0
    assert parse_output(result.output) == pytest.approx([0.0, math.log(10)])


def test_text_tokenize(runner):
    result = runner.invoke(cli, ["text", "tokenize", "Hello, World! 123"])
    assert result.exit_code == 0
    assert parse_output(result.output) == ["hello", "world", "123"]


def test_text_remove_punctuation(runner):
    result = runner.invoke(cli, ["text", "remove-punctuation", "Hello, World!"])
    assert result.exit_code == 0
    assert result.output.strip() == "Hello World"


def test_text_remove_stop_words(runner):
    result = runner.invoke(
        cli,
        [
            "text",
            "remove-stop-words",
            "the quick brown fox",
            "--stop-words",
            '["the", "fox"]',
        ],
    )
    assert result.exit_code == 0
    assert result.output.strip() == "quick brown"


def test_struct_shuffle(runner):
    result = runner.invoke(cli, ["struct", "shuffle", "[1, 2, 3, 4]", "--seed", "1"])
    assert result.exit_code == 0
    values = parse_output(result.output)
    assert sorted(values) == [1, 2, 3, 4]

    result_repeat = runner.invoke(
        cli, ["struct", "shuffle", "[1, 2, 3, 4]", "--seed", "1"]
    )
    assert parse_output(result_repeat.output) == values


def test_struct_flatten(runner):
    result = runner.invoke(cli, ["struct", "flatten", "[[1, 2], [3, 4]]"])
    assert result.exit_code == 0
    assert parse_output(result.output) == [1, 2, 3, 4]


def test_struct_unique_values(runner):
    result = runner.invoke(cli, ["struct", "unique-values", "[1, 2, 2, 3, 1]"])
    assert result.exit_code == 0
    assert parse_output(result.output) == [1, 2, 3]
