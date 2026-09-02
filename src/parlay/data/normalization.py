"""Canonical team identity handling."""

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True, slots=True)
class Team:
    team_id: str
    canonical_name: str


class TeamRegistry:
    def __init__(self, teams: Iterable[Team], aliases: dict[str, str] | None = None):
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

    def names(self) -> dict[str, str]:
        return {team_id: team.canonical_name for team_id, team in self._teams.items()}


def default_team_aliases() -> dict[str, str]:
    return {
        "manchester city": "Man City", "manchester united": "Man United",
        "manchester utd": "Man United", "tottenham hotspur": "Tottenham",
        "spurs": "Tottenham", "nottingham forest": "Nott'm Forest",
        "newcastle united": "Newcastle", "newcastle utd": "Newcastle",
        "afc bournemouth": "Bournemouth", "bournemouth": "Bournemouth",
        "coventry city": "Coventry", "coventry": "Coventry",
        "brighton & hove albion": "Brighton",
        "leeds united": "Leeds", "hull city": "Hull", "ipswich town": "Ipswich",
    }


def canonical_team_name(name: str) -> str:
    return default_team_aliases().get(name.strip().casefold(), name.strip())
