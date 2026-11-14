import pytest

from stats import mean, median, value_range


def test_mean():
    assert mean([1, 2, 3, 4]) == 2.5


def test_median_with_an_odd_count():
    assert median([5, 1, 3]) == 3


def test_median_with_an_even_count():
    assert median([4, 1, 3, 2]) == 2.5


def test_value_range():
    assert value_range([3, 9, 4]) == 6


def test_empty_input_is_rejected():
    with pytest.raises(ValueError):
        median([])
