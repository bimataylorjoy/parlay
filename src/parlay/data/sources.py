"""Reproducible raw-source acquisition with provenance metadata."""

from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
from urllib.request import Request, urlopen


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# football-data.co.uk URL pattern: mmz4281/YYZZ/E0.csv -> season "20YY-20ZZ"
_SEASON_RE = re.compile(r"mmz4281/(\d{2})(\d{2})/")


def detect_season(source: str) -> str:
    """Attempt to extract a season label from a football-data.co.uk path or URL.

    Returns a normalised ``"20YY-20ZZ"`` string when the pattern matches,
    or ``"unknown"`` otherwise.
    """
    match = _SEASON_RE.search(source)
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        return f"20{start:02d}-20{end:02d}"
    return "unknown"


# Standard football-data.co.uk season codes for EPL (E0.csv)
FOOTBALL_DATA_EPL_SEASONS: dict[str, str] = {
    "2017-18": "https://www.football-data.co.uk/mmz4281/1718/E0.csv",
    "2018-19": "https://www.football-data.co.uk/mmz4281/1819/E0.csv",
    "2019-20": "https://www.football-data.co.uk/mmz4281/1920/E0.csv",
    "2020-21": "https://www.football-data.co.uk/mmz4281/2021/E0.csv",
    "2021-22": "https://www.football-data.co.uk/mmz4281/2122/E0.csv",
    "2022-23": "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
    "2023-24": "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
    "2024-25": "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
}


def acquire_csv(source: str | Path, snapshot_dir: str | Path = "data/raw/snapshots",
                *, timeout: int = 30) -> tuple[Path, dict[str, object]]:
    """Copy a local file or download a CSV, then persist immutable provenance."""
    target_dir = Path(snapshot_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    source_text = str(source)
    if source_text.startswith(("http://", "https://")):
        request = Request(source_text, headers={"User-Agent": "parlay-research/0.1"})
        with urlopen(request, timeout=timeout) as response:
            data = response.read()
        original_name = Path(source_text.split("?", 1)[0]).name or "source.csv"
    else:
        source_path = Path(source)
        if not source_path.is_file():
            raise FileNotFoundError(f"Source file does not exist: {source_path}")
        data = source_path.read_bytes()
        original_name = source_path.name
    if not data:
        raise ValueError("Source returned an empty file")
    digest = _sha256(data)
    snapshot = target_dir / f"{digest[:16]}-{original_name}"
    snapshot.write_bytes(data)
    metadata = {
        "source": source_text,
        "snapshot": str(snapshot),
        "sha256": digest,
        "acquired_at": datetime.now(timezone.utc).isoformat(),
        "bytes": len(data),
        "season": detect_season(source_text),
    }
    snapshot.with_suffix(snapshot.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return snapshot, metadata
