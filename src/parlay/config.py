"""Local settings: load secrets from an env file outside the repository."""

from __future__ import annotations

import os
import re
from pathlib import Path

DEFAULT_ENV_FILE = Path("~/.config/parlay/env")
_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


def env_file_path() -> Path:
    return Path(os.environ.get("PARLAY_ENV_FILE", str(DEFAULT_ENV_FILE))).expanduser()


def read_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _LINE.match(line)
        if match:
            value = match.group(2)
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            values[match.group(1)] = value
    return values


def load_env_file(path: Path | None = None) -> dict[str, str]:
    """Export variables from the env file without overriding ones already set."""
    values = read_env_file(path or env_file_path())
    for key, value in values.items():
        os.environ.setdefault(key, value)
    return values


def set_env_value(key: str, value: str, path: Path | None = None) -> Path:
    """Create or update ``export KEY=value`` in the env file (permissions 600)."""
    target = path or env_file_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = target.read_text(encoding="utf-8").splitlines() if target.is_file() else []
    entry = f"export {key}={value}"
    for i, line in enumerate(lines):
        match = _LINE.match(line)
        if match and match.group(1) == key and not line.lstrip().startswith("#"):
            lines[i] = entry
            break
    else:
        lines.append(entry)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    target.chmod(0o600)
    os.environ[key] = value
    return target
