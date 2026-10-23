import pytest

from parlay.prediction.odds import (
    american_to_decimal,
    fractional_to_decimal,
    hong_kong_to_decimal,
    indonesian_to_decimal,
    malay_to_decimal,
    to_decimal,
)


@pytest.mark.parametrize(
    "converter,value,expected",
    [
        (indonesian_to_decimal, 1.5, 2.5),
        (indonesian_to_decimal, -2.0, 1.5),
        (indonesian_to_decimal, -1.0, 2.0),
        (malay_to_decimal, 0.8, 1.8),
        (malay_to_decimal, -0.5, 3.0),
        (hong_kong_to_decimal, 0.9, 1.9),
        (american_to_decimal, 150, 2.5),
        (american_to_decimal, -200, 1.5),
    ],
)
def test_converters(converter, value, expected):
    assert converter(value) == pytest.approx(expected)


def test_fractional_and_dispatch():
    assert fractional_to_decimal("5/2") == pytest.approx(3.5)
    assert to_decimal("-1.25", "indo") == pytest.approx(1.8)
    assert to_decimal("2.10") == 2.10


@pytest.mark.parametrize(
    "value,odds_format",
    [("0.5", "indo"), ("1.5", "malay"), ("50", "american"), ("1.0", "decimal"), ("2", "klingon")],
)
def test_invalid_prices_are_rejected(value, odds_format):
    with pytest.raises(ValueError):
        to_decimal(value, odds_format)
