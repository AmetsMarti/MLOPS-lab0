import math

import pytest

try:
    from src import preprocessing
except ImportError:
    import src.preprocessing


@pytest.fixture
def numbers():
    return [1.0, 2.0, 3.0, 4.0]


def test_remove_missing():
    assert preprocessing.remove_missing([1, None, "", "  ", float("nan"), 2, "a"]) == [
        1,
        2,
        "a",
    ]


@pytest.mark.parametrize(
    "values,filler,expected",
    [
        ([1, None, "", 2], 0, [1, 0, 0, 2]),
        (["a", None, "b"], "x", ["a", "x", "b"]),
        ([None, float("nan")], 9, [9, 9]),
    ],
)
def test_fill_missing(values, filler, expected):
    assert preprocessing.fill_missing(values, filler) == expected


def test_remove_duplicates():
    assert preprocessing.remove_duplicates([1, 2, 2, 3, 1, "a", "a"]) == [1, 2, 3, "a"]


def test_unique_values():
    assert preprocessing.unique_values([1, 2, 2, 3, 1]) == [1, 2, 3]


@pytest.mark.parametrize(
    "values,new_min,new_max,expected",
    [
        ([1, 2, 3], 0.0, 1.0, [0.0, 0.5, 1.0]),
        ([1, 2, 3], -1.0, 1.0, [-1.0, 0.0, 1.0]),
        ([5, 5, 5], 0.0, 1.0, [0.0, 0.0, 0.0]),
    ],
)
def test_normalize_min_max(values, new_min, new_max, expected):
    result = preprocessing.normalize_min_max(values, new_min, new_max)
    assert result == pytest.approx(expected)


def test_standardize_z_score(numbers):
    result = preprocessing.standardize_z_score(numbers)
    assert result == pytest.approx(
        [-1.3416407864998738, -0.4472135954999579, 0.4472135954999579, 1.3416407864998738]
    )
    assert preprocessing.standardize_z_score([2, 2, 2]) == [0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "values,minimum,maximum,expected",
    [
        ([1, 5, 10], 2.0, 8.0, [2.0, 5.0, 8.0]),
        ([-1, 0.5, 2], 0.0, 1.0, [0.0, 0.5, 1.0]),
    ],
)
def test_clip_values(values, minimum, maximum, expected):
    assert preprocessing.clip_values(values, minimum, maximum) == pytest.approx(expected)


def test_convert_to_integers():
    assert preprocessing.convert_to_integers(["1", "2.7", 3.9, "x", None, "4"]) == [
        1,
        2,
        3,
        4,
    ]


def test_log_transform():
    result = preprocessing.log_transform([1, math.e, 10, -1, 0, "x"])
    assert result == pytest.approx([0.0, 1.0, math.log(10)])


@pytest.mark.parametrize(
    "text_value,expected",
    [
        ("Hello, World! 123", ["hello", "world", "123"]),
        ("", []),
        ("Python3_is great", ["python3", "is", "great"]),
    ],
)
def test_tokenize(text_value, expected):
    assert preprocessing.tokenize(text_value) == expected


def test_remove_punctuation():
    assert preprocessing.remove_punctuation("Hello, World! It's #1.") == "Hello World Its 1"


@pytest.mark.parametrize(
    "text_value,stop_words,expected",
    [
        ("The quick brown fox", ["the", "fox"], "quick brown"),
        ("A A b c", ["a"], "b c"),
        ("keep all", [], "keep all"),
    ],
)
def test_remove_stop_words(text_value, stop_words, expected):
    assert preprocessing.remove_stop_words(text_value, stop_words) == expected


def test_flatten():
    assert preprocessing.flatten([[1, 2], [3], [4, 5]]) == [1, 2, 3, 4, 5]
    assert preprocessing.flatten([]) == []


def test_shuffle(numbers):
    original = list(numbers)
    result = preprocessing.shuffle(numbers, seed=42)
    assert result != original
    assert sorted(result) == sorted(original)
    assert numbers == original
    assert preprocessing.shuffle(numbers, seed=42) == result
