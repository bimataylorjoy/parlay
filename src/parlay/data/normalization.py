"""Team identity registry (explicit IDs) on top of canonical team names."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .teams import canonical_team_name, default_team_aliases

__all__ = ["Team", "TeamRegistry", "canonical_team_name", "default_team_aliases"]


@dataclass(frozen=True, slots=True)
class Team:
    team_id: str
    canonical_name: str


class TeamRegistry:
    """Resolve provider names to stable team IDs; unknown names raise ``KeyError``."""

    def __init__(self, teams: Iterable[Team], aliases: dict[str, str] | None = None):
        teams = list(teams)
        self._teams = {team.team_id: team for team in teams}
        self._by_name = {team.canonical_name.casefold(): team.team_id for team in teams}
        self._aliases = {key.casefold(): value for key, value in (aliases or {}).items()}
        unknown = set(self._aliases.values()) - set(self._teams)
        if unknown:
            raise ValueError(f"Aliases reference unknown team IDs: {sorted(unknown)}")

    def resolve(self, name: str) -> str:
        key = name.strip().casefold()
        team_id = self._aliases.get(key, self._by_name.get(key))
        if team_id is None:
            raise KeyError(f"Unknown team name: {name!r}")
        return team_id

    def canonical_name(self, name: str) -> str:
        return self._teams[self.resolve(name)].canonical_name

    def names(self) -> dict[str, str]:
        return {team_id: team.canonical_name for team_id, team in self._teams.items()}
