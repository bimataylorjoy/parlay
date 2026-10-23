"""Time-safe football probability research framework."""

from importlib.metadata import PackageNotFoundError, version

from .data.schemas import Match, OddsSnapshot

try:
    __version__ = version("parlay-research")
except PackageNotFoundError:  # pragma: no cover - running from a source checkout
    __version__ = "0.2.0"

__all__ = ["Match", "OddsSnapshot", "__version__"]
