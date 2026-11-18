"""Synthetic demo league generator.

!!! ALL DATA PRODUCED HERE IS FICTIONAL !!!
Team and player names are invented; any resemblance to real persons is coincidental.

Purpose:
1. The demo works without real (personal) data.
2. Because we *know* the true parameters that generated the results, we can later
   check whether the model recovers them ("parameter recovery" test).

Generative model (the same Poisson model we will fit later):

    home goals ~ Poisson(base_rate * attack[home] * defense[away] * home_advantage)
    away goals ~ Poisson(base_rate * attack[away] * defense[home])

``attack > 1`` = scores more than average; ``defense > 1`` = concedes more than
average (a *weak* defence). Both are log-normal and normalised so that their
geometric mean is 1, which makes ``base_rate`` the goal rate of an average away side.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta

import numpy as np

from football_predictor.db.models import MatchStatus
from football_predictor.ingest.records import AppearanceRecord, GoalRecord, MatchRecord

SYNTHETIC_SEASON_NAME = "DEMO 2025/26 (synthetic)"
SYNTHETIC_COMPETITION_NAME = "DEMO League (synthetic)"

# Invented place names.
TEAM_NAMES = [
    "Ködvár SE", "Szélmalom FC", "Holdfény TC", "Csillagrét SK", "Vadkörte KSE",
    "Ezüstpatak FC", "Rókahegy SC", "Fenyvesalja SE", "Kőhíd ULE", "Darázsfalva FC",
    "Napfolt AC", "Mohosvölgy FC", "Bagolyliget SE", "Sóskút-Fiktív KSK",
]
LAST_NAMES = [
    "Ágas", "Berkes", "Csermely", "Dombi", "Erdős", "Fürge", "Gyökeres", "Hamvas",
    "Ivó", "Jegenye", "Kavicsos", "Lankás", "Mezei", "Nádas", "Ormos", "Patakos",
    "Rügyes", "Sásdi", "Tölgyes", "Ugari", "Vackor", "Zsályás", "Borókás", "Cserjés",
]
FIRST_NAMES = [
    "Ádám", "Bence", "Csaba", "Dániel", "Ervin", "Ferenc", "Gergő", "Huba", "Imre",
    "János", "Kristóf", "Levente", "Márk", "Noel", "Olivér", "Péter", "Richárd",
    "Szabolcs", "Tamás", "Vilmos", "Zalán", "Botond", "Kende", "Örs",
]
# Relative scoring weight by position.
ROLE_WEIGHTS = {"GK": 0.01, "DF": 0.5, "MF": 1.5, "FW": 4.0}
MATCH_MINUTES = 90


@dataclass(frozen=True)
class PlayerTruth:
    """True parameters of one synthetic player."""

    name: str
    team: str
    role: str
    scoring_share: float  # expected fraction of the team's (non-own) goals


@dataclass
class GroundTruth:
    """The parameters that generated the synthetic league."""

    seed: int
    base_rate: float
    home_advantage: float
    attack: dict[str, float]
    defense: dict[str, float]
    players: list[PlayerTruth] = field(default_factory=list)

    def to_json(self) -> str:
        """Serialise for later comparison with fitted model parameters."""
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


@dataclass
class SampleLeague:
    """Generated records plus the ground truth behind them."""

    records: list[MatchRecord]
    truth: GroundTruth


def double_round_robin(teams: list[str]) -> list[list[tuple[str, str]]]:
    """Schedule via the circle method: every pair meets twice, once at each home.

    One team stays fixed, the others rotate one position per round. Returns a list
    of rounds, each a list of (home, away) pairs.
    """
    n = len(teams)
    if n < 2 or n % 2:
        raise ValueError("Need an even number (>= 2) of teams")
    rotation = list(teams)
    first_half: list[list[tuple[str, str]]] = []
    for r in range(n - 1):
        pairs = []
        for i in range(n // 2):
            a, b = rotation[i], rotation[n - 1 - i]
            pairs.append((a, b) if (r + i) % 2 == 0 else (b, a))  # alternate home side
        first_half.append(pairs)
        rotation = [rotation[0], rotation[-1], *rotation[1:-1]]
    second_half = [[(away, home) for home, away in rnd] for rnd in first_half]
    return first_half + second_half


@dataclass
class _Squad:
    names: list[str]
    roles: list[str]
    shares: np.ndarray
    regularity: np.ndarray


def _make_squads(
    rng: np.random.Generator, teams: list[str], squad_size: int
) -> dict[str, _Squad]:
    if squad_size < 14:
        raise ValueError("squad_size must be >= 14")
    all_names = [f"{last} {first}" for last, first in itertools.product(LAST_NAMES, FIRST_NAMES)]
    picked = rng.choice(len(all_names), size=len(teams) * squad_size, replace=False)
    roles = ["GK", "GK"] + ["DF"] * 5 + ["MF"] * 5 + ["FW"] * (squad_size - 12)

    squads = {}
    for t_idx, team in enumerate(teams):
        idx = picked[t_idx * squad_size:(t_idx + 1) * squad_size]
        weights = np.array([ROLE_WEIGHTS[r] for r in roles]) * rng.lognormal(0, 0.5, squad_size)
        regularity = rng.uniform(0.4, 1.0, squad_size)
        squads[team] = _Squad(
            names=[all_names[i] for i in idx],
            roles=roles,
            shares=weights / weights.sum(),
            regularity=regularity,
        )
    return squads


def _simulate_lineup(rng: np.random.Generator, squad: _Squad) -> dict[int, tuple[int, int]]:
    """Return {player index: (on-minute, off-minute)} for players who appeared.

    One goalkeeper + 10 outfield starters (chosen by regularity), 3 substitutions
    in the second half.
    """
    outfield = [i for i, r in enumerate(squad.roles) if r != "GK"]
    keeper = 0 if rng.random() < 0.9 else 1
    p = squad.regularity[outfield] / squad.regularity[outfield].sum()
    starters = list(rng.choice(outfield, size=10, replace=False, p=p))
    bench = [i for i in outfield if i not in starters]

    intervals = {keeper: (0, MATCH_MINUTES)} | {i: (0, MATCH_MINUTES) for i in starters}
    for on, off in zip(
        rng.choice(bench, size=3, replace=False),
        rng.choice(starters, size=3, replace=False),
        strict=True,
    ):
        minute = int(rng.integers(46, 86))
        intervals[int(off)] = (0, minute)
        intervals[int(on)] = (minute, MATCH_MINUTES)
    return intervals


def _pick_on_pitch(
    rng: np.random.Generator, intervals: dict[int, tuple[int, int]], weights: np.ndarray | None
) -> tuple[int, int]:
    """Pick a player (weighted by weight x minutes on pitch) and a minute they were on."""
    players = list(intervals)
    minutes = np.array([intervals[i][1] - intervals[i][0] for i in players], dtype=float)
    w = minutes * (weights[players] if weights is not None else 1.0)
    player = players[int(rng.choice(len(players), p=w / w.sum()))]
    start, end = intervals[player]
    return player, int(rng.integers(start + 1, end + 1))


def _simulate_goals(
    rng: np.random.Generator,
    n_goals: int,
    team: str,
    opponent: str,
    squads: dict[str, _Squad],
    lineups: dict[str, dict[int, tuple[int, int]]],
    own_goal_rate: float,
    penalty_rate: float,
) -> list[GoalRecord]:
    goals = []
    for _ in range(n_goals):
        if rng.random() < own_goal_rate:
            opp_outfield = {
                i: iv for i, iv in lineups[opponent].items() if squads[opponent].roles[i] != "GK"
            }
            idx, minute = _pick_on_pitch(rng, opp_outfield, None)
            goals.append(GoalRecord(team=team, player=squads[opponent].names[idx],
                                    minute=minute, is_own_goal=True))
        else:
            idx, minute = _pick_on_pitch(rng, lineups[team], squads[team].shares)
            goals.append(GoalRecord(team=team, player=squads[team].names[idx], minute=minute,
                                    is_penalty=bool(rng.random() < penalty_rate)))
    return goals


def generate_sample_league(
    n_teams: int = 10,
    played_rounds: int = 14,
    seed: int = 42,
    start_date: date = date(2025, 8, 10),
    squad_size: int = 16,
    strength_sigma: float = 0.3,
    base_rate: float = 1.35,
    home_advantage: float = 1.3,
    own_goal_rate: float = 0.04,
    penalty_rate: float = 0.08,
    postponed_rate: float = 0.02,
    with_appearances: bool = True,
) -> SampleLeague:
    """Generate a fictional double round-robin league.

    Rounds ``1..played_rounds`` get results; the rest stay *scheduled* (these are
    what we predict). The same ``seed`` always yields exactly the same league.
    """
    if not 2 <= n_teams <= len(TEAM_NAMES):
        raise ValueError(f"n_teams must be between 2 and {len(TEAM_NAMES)}")
    rng = np.random.default_rng(seed)
    teams = TEAM_NAMES[:n_teams]

    # Log-normal strengths, centred so that the geometric mean is exactly 1.
    log_att = rng.normal(0, strength_sigma, n_teams)
    log_def = rng.normal(0, strength_sigma, n_teams)
    attack = dict(zip(teams, np.exp(log_att - log_att.mean()), strict=True))
    defense = dict(zip(teams, np.exp(log_def - log_def.mean()), strict=True))
    squads = _make_squads(rng, teams, squad_size)

    schedule = double_round_robin(teams)
    if not 0 <= played_rounds <= len(schedule):
        raise ValueError(f"played_rounds must be between 0 and {len(schedule)}")

    records: list[MatchRecord] = []
    for rnd_no, fixtures in enumerate(schedule, start=1):
        match_date = start_date + timedelta(weeks=rnd_no - 1)
        for home, away in fixtures:
            rec = MatchRecord(
                season=SYNTHETIC_SEASON_NAME, round=rnd_no, home_team=home, away_team=away,
                match_date=match_date, status=MatchStatus.SCHEDULED,
                source_url=f"synthetic://seed-{seed}/round-{rnd_no}",
                competition=SYNTHETIC_COMPETITION_NAME,
            )
            records.append(rec)
            if rnd_no > played_rounds:
                continue
            if rng.random() < postponed_rate:
                rec.status = MatchStatus.POSTPONED
                continue

            lam = base_rate * attack[home] * defense[away] * home_advantage
            mu = base_rate * attack[away] * defense[home]
            rec.home_goals, rec.away_goals = int(rng.poisson(lam)), int(rng.poisson(mu))
            rec.status = MatchStatus.PLAYED

            lineups = {home: _simulate_lineup(rng, squads[home]),
                       away: _simulate_lineup(rng, squads[away])}
            goals = _simulate_goals(rng, rec.home_goals, home, away, squads, lineups,
                                    own_goal_rate, penalty_rate)
            goals += _simulate_goals(rng, rec.away_goals, away, home, squads, lineups,
                                     own_goal_rate, penalty_rate)
            rec.goals = sorted(goals, key=lambda g: g.minute or 0)
            if with_appearances:
                rec.appearances = [
                    AppearanceRecord(team=t, player=squads[t].names[i], minutes_played=e - s)
                    for t in (home, away)
                    for i, (s, e) in sorted(lineups[t].items())
                ]

    truth = GroundTruth(
        seed=seed,
        base_rate=base_rate,
        home_advantage=home_advantage,
        attack={t: float(v) for t, v in attack.items()},
        defense={t: float(v) for t, v in defense.items()},
        players=[
            PlayerTruth(name=n, team=t, role=r, scoring_share=float(s))
            for t, sq in squads.items()
            for n, r, s in zip(sq.names, sq.roles, sq.shares, strict=True)
        ],
    )
    return SampleLeague(records=records, truth=truth)
