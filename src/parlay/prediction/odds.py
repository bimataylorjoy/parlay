"""Odds-format conversion (decimal, Indonesian, Malay, Hong Kong, American, fractional)."""

from __future__ import annotations

from fractions import Fraction

ODDS_FORMATS = ("decimal", "indo", "malay", "hk", "american", "fractional")


def indonesian_to_decimal(value: float) -> float:
    """Indonesian odds: ``+x`` → ``1 + x``; ``-x`` → ``1 + 1/x`` (``±1`` is evens)."""
    if abs(value) < 1:
        raise ValueError("Indonesian odds must satisfy |odds| >= 1")
    return 1.0 + (value if value > 0 else 1.0 / abs(value))


def malay_to_decimal(value: float) -> float:
    """Malay odds in (0, 1] → ``1 + x``; negative ``-x`` → ``1 + 1/x``."""
    if value == 0 or abs(value) > 1:
        raise ValueError("Malay odds must satisfy 0 < |odds| <= 1")
    return 1.0 + (value if value > 0 else 1.0 / abs(value))


def hong_kong_to_decimal(value: float) -> float:
    if value <= 0:
        raise ValueError("Hong Kong odds must be positive")
    return 1.0 + value


def american_to_decimal(value: float) -> float:
    if -100 < value < 100:
        raise ValueError("American odds must satisfy |odds| >= 100")
    return 1.0 + (value / 100.0 if value > 0 else 100.0 / abs(value))


def fractional_to_decimal(value: str) -> float:
    fraction = Fraction(value.strip())
    if fraction <= 0:
        raise ValueError("fractional odds must be positive")
    return 1.0 + float(fraction)


def to_decimal(value: str | float, odds_format: str = "decimal") -> float:
    """Convert one price in ``odds_format`` to decimal odds."""
    if odds_format == "fractional":
        return fractional_to_decimal(str(value))
    number = float(value)
    converters = {
        "decimal": lambda x: x,
        "indo": indonesian_to_decimal,
        "malay": malay_to_decimal,
        "hk": hong_kong_to_decimal,
        "american": american_to_decimal,
    }
    if odds_format not in converters:
        raise ValueError(f"unknown odds format {odds_format!r}; expected one of {ODDS_FORMATS}")
    decimal = float(converters[odds_format](number))
    if decimal <= 1.0:
        raise ValueError(f"decimal odds must be greater than 1 (got {decimal} from {value!r})")
    return decimal
