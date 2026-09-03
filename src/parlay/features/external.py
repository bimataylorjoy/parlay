"""Leakage-safe summaries for optional external match information."""

from __future__ import annotations

from datetime import datetime


def external_information_features(items, *, as_of: datetime) -> dict[str, float | str]:
    """Summarize only information available by ``as_of``.

    This intentionally produces audit features, not arbitrary goal-rate
    adjustments. Player-level impact requires validated player ratings and is
    therefore left to a separately trained model.
    """
    available = [item for item in items if item.is_available_at(as_of)]
    lineups = [item for item in available if item.__class__.__name__ == "LineupStatus"]
    injuries = [item for item in available if item.__class__.__name__ == "InjuryReport"]
    news = [item for item in available if item.__class__.__name__ == "TeamNews"]
    confirmed = sum(1 for item in lineups if str(item.status).casefold() in {"starting", "confirmed"})
    return {
        "available_information_count": float(len(available)),
        "lineup_record_count": float(len(lineups)),
        "confirmed_lineup_count": float(confirmed),
        "injury_report_count": float(len(injuries)),
        "prematch_news_count": float(len(news)),
        "information_quality": "observed" if available else "missing",
    }
