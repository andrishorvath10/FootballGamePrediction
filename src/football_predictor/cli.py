"""Command-line entry point.

Examples::

    football-predictor init-db
    football-predictor init-db --drop
    football-predictor db current | upgrade | check
    football-predictor db revision -m "add referee to match"
    football-predictor validate
    football-predictor generate-sample --seed 42
    football-predictor import-csv examples/matches.csv --goals examples/goals.csv
    football-predictor scrape --config my_league.toml [--offline | --refresh]
    football-predictor add-alias "Ködvár" "Ködvár SE"
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from football_predictor.config import get_settings
from football_predictor.ingest.records import DEFAULT_COMPETITION
from football_predictor.model.drift import DRIFT_DEFAULT

# Tuned on 2025/26 with `backtest-scorers --tune` (see README).
SCORER_DEFAULTS: dict = {"shrinkage": 8.0, "form_matches": 5, "dispersion": 1.0}

logger = logging.getLogger("football_predictor")


def setup_logging(level: str) -> None:
    """Configure root logging once for the whole application."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Alembic logs every plugin at INFO; our own messages already say what changed.
    logging.getLogger("alembic").setLevel(logging.WARNING)


def cmd_init_db(args: argparse.Namespace) -> int:
    """Create the schema, or migrate an existing database to the latest revision."""
    from football_predictor.db.session import init_db

    init_db(drop_existing=args.drop)
    return 0


def cmd_db(args: argparse.Namespace) -> int:
    """Schema migrations: show, apply, revert, generate or check revisions."""
    from football_predictor.db import migrate
    from football_predictor.db.session import get_engine

    engine = get_engine()
    if args.action == "current":
        current, head = migrate.current_revision(engine), migrate.head_revision()
        state = "up to date" if current == head else "upgrade pending"
        logger.info("Database revision %s, head %s (%s)", current, head, state)
    elif args.action == "upgrade":
        migrate.ensure_schema(engine)
        logger.info("Database at revision %s", migrate.current_revision(engine))
    elif args.action == "downgrade":
        if not args.revision:
            logger.error("downgrade needs a target revision, e.g. -1 or base")
            return 2
        migrate.downgrade(engine, args.revision)
        logger.info("Database at revision %s", migrate.current_revision(engine))
    elif args.action == "revision":
        if not args.message:
            logger.error("revision needs -m/--message")
            return 2
        migrate.autogenerate_revision(engine, args.message)
    elif args.action == "check":
        diff = migrate.schema_diff(engine)
        if diff:
            logger.error("Schema differs from the models (%d changes): %s", len(diff), diff)
            return 1
        logger.info("Schema matches the models")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Check that Goal rows match final scores; non-zero exit code on mismatch."""
    from football_predictor.db.session import session_scope
    from football_predictor.db.validation import find_goal_mismatches

    with session_scope() as session:
        mismatches = find_goal_mismatches(session, season_id=args.season_id)
    if mismatches:
        logger.error("%d goal/score mismatches found", len(mismatches))
        return 1
    logger.info("No goal/score mismatches found")
    return 0


def _load_and_validate(records: list) -> int:
    """Upsert records in one transaction, then run the goal/score validation."""
    from football_predictor.db.session import init_db, session_scope
    from football_predictor.db.validation import find_goal_mismatches
    from football_predictor.ingest.loader import LeagueLoader

    init_db()  # make sure tables exist (no-op if they do)
    with session_scope() as session:
        LeagueLoader(session).load(records)
        session.flush()
        mismatches = find_goal_mismatches(session)
    if mismatches:
        logger.warning("%d goal/score mismatches - see warnings above", len(mismatches))
    return 0


def cmd_generate_sample(args: argparse.Namespace) -> int:
    """Generate the fictional demo league and load it into the database."""
    from football_predictor.ingest.sample_data import generate_sample_league

    league = generate_sample_league(
        n_teams=args.teams, played_rounds=args.played_rounds, seed=args.seed,
        with_appearances=not args.no_appearances,
    )
    logger.info("Generated %d SYNTHETIC matches (seed=%d)", len(league.records), args.seed)
    _load_and_validate(league.records)

    args.truth_out.parent.mkdir(parents=True, exist_ok=True)
    args.truth_out.write_text(league.truth.to_json(), encoding="utf-8")
    logger.info("Ground-truth parameters written to %s", args.truth_out)
    for team in sorted(league.truth.attack, key=league.truth.attack.get, reverse=True):
        logger.info("  %-20s attack %.2f  defense %.2f", team,
                    league.truth.attack[team], league.truth.defense[team])
    return 0


def cmd_import_csv(args: argparse.Namespace) -> int:
    """Import matches (and optional goals/appearances) from CSV files."""
    from football_predictor.ingest.csv_import import CsvImportError, read_league_csv

    try:
        records = read_league_csv(
            args.matches, args.goals, args.appearances, args.season, args.competition
        )
    except CsvImportError as exc:
        logger.error("%s", exc)
        return 2
    logger.info("Read %d matches from %s", len(records), args.matches)
    return _load_and_validate(records)


def cmd_scrape(args: argparse.Namespace) -> int:
    """Fetch (or re-parse cached) pages and load the parsed matches."""
    from football_predictor.ingest.scraper import PoliteFetcher, ScraperConfig, scrape

    config = ScraperConfig.from_toml(args.config)
    fetcher = PoliteFetcher(
        raw_dir=get_settings().raw_data_dir, user_agent=config.user_agent,
        delay_seconds=config.delay_seconds, timeout_seconds=config.timeout_seconds,
        max_retries=config.max_retries,
    )
    records, failed = scrape(config, fetcher, refresh=args.refresh, offline=args.offline)
    if records:
        _load_and_validate(records)
    if failed:
        logger.error("%d page(s) failed: %s", len(failed), ", ".join(failed))
        return 1
    return 0


def cmd_scrape_mlsz(args: argparse.Namespace) -> int:
    """Scrape MLSZ adatbank competitions; load each competition-season as it finishes."""
    from football_predictor.db.session import init_db, session_scope
    from football_predictor.ingest.loader import LeagueLoader, ensure_competition
    from football_predictor.ingest.mlsz import MlszConfig, MlszScraper
    from football_predictor.ingest.scraper import PoliteFetcher

    config = MlszConfig.from_toml(args.config)
    fetcher = PoliteFetcher(
        raw_dir=get_settings().raw_data_dir / "mlsz", user_agent=config.user_agent,
        delay_seconds=config.delay_seconds, timeout_seconds=config.timeout_seconds,
        max_retries=config.max_retries,
    )
    scraper = MlszScraper(config, fetcher, fetch_details=not args.no_details,
                          refresh=args.refresh, offline=args.offline)
    targets = list(scraper.iter_targets(args.season, args.competition))
    if not targets:
        logger.error("Nothing to scrape - check --season / --competition names")
        return 1

    init_db()
    for comp, season in targets:
        logger.info("=== %s %s ===", comp.name, season)
        records = scraper.scrape(comp, season)
        with session_scope() as session:
            ensure_competition(session, comp.name, comp.region, comp.level)
        _load_and_validate(records)
        if (comp.name, season) in scraper.adjustments:
            with session_scope() as session:
                LeagueLoader(session).sync_point_adjustments(
                    season, comp.name, scraper.adjustments[(comp.name, season)])

    if scraper.failed:
        logger.error("%d page(s) failed (re-run to retry): %s",
                     len(scraper.failed), ", ".join(scraper.failed[:10]))
        return 1
    return 0


def cmd_competitions(args: argparse.Namespace) -> int:
    """List competition-seasons in the database."""
    from football_predictor.db.session import session_scope
    from football_predictor.model.dataset import list_competition_seasons

    with session_scope() as session:
        table = list_competition_seasons(session)
    print(table.to_string(index=False) if not table.empty else "No data yet.")
    return 0


def _resolve_competition_season(
    session: object, competition: str | None, season: str | None
) -> tuple[str, str] | None:
    from football_predictor.model.dataset import list_competition_seasons

    overview = list_competition_seasons(session)  # type: ignore[arg-type]
    if competition is None:
        names = overview["competition"].unique()
        if len(names) != 1:
            logger.error("Use --competition, one of: %s", ", ".join(map(repr, names)))
            return None
        competition = str(names[0])
    rows = overview[overview["competition"] == competition]
    if rows.empty:
        logger.error("Unknown competition %r (see: football-predictor competitions)", competition)
        return None
    if season is None:
        season = str(rows["season"].max())  # "2026/27" > "2025/26"
    elif season not in set(rows["season"]):
        logger.error("No season %r for %s", season, competition)
        return None
    return competition, season


def cmd_predict(args: argparse.Namespace) -> int:
    """Predict one round of a competition-season and print a table."""
    from football_predictor.db.session import session_scope
    from football_predictor.model.dataset import load_matches
    from football_predictor.model.predict import (
        catch_up_rounds,
        next_round,
        predict_round,
        predictions_to_frame,
    )
    from football_predictor.model.strength import StrengthParams

    with session_scope() as session:
        resolved = _resolve_competition_season(session, args.competition, args.season)
        if resolved is None:
            return 1
        competition, season = resolved
        matches = load_matches(session, competition, season)

    rnd = args.round or next_round(matches)
    if rnd is None:
        logger.error("%s %s has no unplayed fixtures", competition, season)
        return 1
    rounds = [rnd] if args.round else [*catch_up_rounds(matches, rnd), rnd]
    params = StrengthParams(model=args.model, xi=args.xi, shrinkage=args.shrinkage,
                            last_n=args.last_n, prior_sd=args.prior_sd)
    predictions, strengths = predict_round(matches, rounds, params)
    if not predictions:
        logger.error("Round %d has no unplayed fixtures", rnd)
        return 1

    extra = f" + catch-up matches from round(s) {rounds[:-1]}" if len(rounds) > 1 else ""
    print(f"\n{competition} {season} - round {rnd}{extra}   [{params.label()}]")
    rho = f", Dixon-Coles rho {strengths.rho:+.3f}" if params.model == "dc" else ""
    print(f"league: base {strengths.base:.2f} goals, "
          f"home advantage x{strengths.home_advantage:.2f}{rho}\n")
    table = predictions_to_frame(predictions)
    pct = ["p_1", "p_x", "p_2", "p_top_score", "p_over_2_5", "p_btts"]
    table[pct] = (table[pct] * 100).round(0).astype(int)
    table[["xg_home", "xg_away"]] = table[["xg_home", "xg_away"]].round(2)
    print(table.to_string(index=False))
    if args.show_strengths:
        print("\nTeam strengths (attack >1 = scores more, defense >1 = concedes more):")
        print(strengths.to_frame().round(2).to_string())
    return 0


def _walk_forward_all(
    competitions: list[str] | None, season: str, forecasters: list, start_round: int
) -> object:
    """Run the walk-forward backtest on every selected competition of a season."""
    import pandas as pd

    from football_predictor.db.session import session_scope
    from football_predictor.model.dataset import list_competition_seasons, load_matches
    from football_predictor.model.evaluate import walk_forward

    with session_scope() as session:
        overview = list_competition_seasons(session)
        rows = overview[overview["season"] == season]
        if competitions:
            rows = rows[rows["competition"].isin(competitions)]
        data = {c: load_matches(session, c, season) for c in rows["competition"]}
    parts = []
    for comp, matches in data.items():
        logger.info("Backtesting %s %s (%d forecasters)", comp, season, len(forecasters))
        preds = walk_forward(matches, forecasters, start_round)
        if not preds.empty:
            parts.append(preds.assign(competition=comp, season=season))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def cmd_backtest(args: argparse.Namespace) -> int:
    """Walk-forward backtest; with --tune: pick parameters on one season, test on another."""
    from football_predictor.model.evaluate import (
        Forecaster,
        baselines,
        calibration_table,
        parameter_grid,
        score_table,
    )
    from football_predictor.model.strength import StrengthParams

    out_dir = Path("data/backtest")
    out_dir.mkdir(parents=True, exist_ok=True)

    def fc(**kw: object) -> Forecaster:
        p = StrengthParams(**kw)  # type: ignore[arg-type]
        return Forecaster(p.label(), p)

    if args.tune:
        grid = parameter_grid()
        logger.info("Tuning %d configurations on %s", len(grid), args.tune_season)
        tuning = _walk_forward_all(args.competition, args.tune_season, grid, args.start_round)
        if tuning.empty:
            logger.error("No data for tuning season %s", args.tune_season)
            return 1
        table = score_table(tuning)
        table.to_csv(out_dir / f"tuning_{args.tune_season.replace('/', '-')}.csv", index=False)
        print(f"\nTuning on {args.tune_season} (top 15 by RPS):")
        print(table.head(15).round(4).to_string(index=False))
        best = {}
        for model in ("simple", "mle", "dc"):
            name = table[table["forecaster"].str.startswith(model + "(")].iloc[0]["forecaster"]
            best[model] = next(f for f in grid if f.name == name)
        forecasters = [*baselines(), fc(model="simple", shrinkage=2),
                       fc(model="simple", shrinkage=2, last_n=5), *best.values()]
    else:
        forecasters = [
            *baselines(),
            fc(model="simple", shrinkage=2),               # no form weighting
            fc(model="simple", shrinkage=2, last_n=5),     # "last 5 matches"
            fc(model="simple", shrinkage=2, xi=0.005),     # exponential decay
            fc(model="mle", prior_sd=0.4),
            fc(model="dc", prior_sd=0.4),
        ]

    preds = _walk_forward_all(args.competition, args.season, forecasters, args.start_round)
    if preds.empty:
        logger.error("No played matches to backtest for season %s", args.season)
        return 1
    tag = args.season.replace("/", "-")
    preds.to_csv(out_dir / f"predictions_{tag}.csv", index=False)

    print(f"\nWalk-forward backtest, season {args.season}, rounds >= {args.start_round}"
          f"{' (held-out: parameters tuned on ' + args.tune_season + ')' if args.tune else ''}")
    overall = score_table(preds)
    print(overall.round(4).to_string(index=False))
    overall.to_csv(out_dir / f"scores_{tag}.csv", index=False)

    best_model = overall[~overall["forecaster"].isin([b.name for b in baselines()])].iloc[0]
    print(f"\nPer competition ({best_model['forecaster']}):")
    per_comp = score_table(preds[preds["forecaster"] == best_model["forecaster"]],
                           by=["competition"])
    print(per_comp.drop(columns="forecaster").round(4).to_string(index=False))
    print(f"\nCalibration ({best_model['forecaster']}):")
    calib = calibration_table(preds[preds["forecaster"] == best_model["forecaster"]])
    print(calib.round(3).to_string(index=False))
    calib.to_csv(out_dir / f"calibration_{tag}.csv", index=False)
    return 0


def _load_competition(competition: str | None, season: str | None) -> tuple | None:
    """(competition, season, matches, appearances, goals) or None if not found."""
    from football_predictor.db.session import session_scope
    from football_predictor.model.dataset import load_appearances, load_goals, load_matches

    with session_scope() as session:
        resolved = _resolve_competition_season(session, competition, season)
        if resolved is None:
            return None
        comp, seas = resolved
        return (comp, seas, load_matches(session, comp, seas),
                load_appearances(session, comp, seas), load_goals(session, comp, seas))


def _stored_assumptions(competition: str, season: str) -> list:
    from football_predictor.db.assumption_store import list_assumptions
    from football_predictor.db.session import session_scope

    with session_scope() as session:
        return list_assumptions(session, competition, season)


def cmd_assume(args: argparse.Namespace) -> int:
    """Store, list, remove or check expert assumptions about teams."""
    import pandas as pd

    from football_predictor.db import assumption_store as store
    from football_predictor.db.session import session_scope
    from football_predictor.model.assumptions import calibration, check_shift, checks_frame
    from football_predictor.model.simulate import TeamShift

    if args.all_teams is not None or args.all_competitions:
        if args.action != "check" or args.all_teams is None:
            logger.error("--all-teams DATE is for 'assume check' (with --all-competitions "
                         "optional)")
            return 2
        return _check_all_teams(args)
    loaded = _load_competition(args.competition, args.season)
    if loaded is None:
        return 1
    competition, season, matches, _, _ = loaded
    teams = set(matches["home"]) | set(matches["away"])
    if args.action in ("add", "remove") and args.team not in teams:
        logger.error("Unknown team %r in %s %s", args.team, competition, season)
        return 2
    if args.action == "add":
        try:
            shift = TeamShift.parse(args.change or "0")
        except ValueError:
            logger.error("CHANGE must look like +20%%, -0.1 or +20%%@2027-03-01")
            return 2
        start = shift.start.date() if shift.start is not None else None
        with session_scope() as session:
            store.save_assumption(session, competition, season, args.team, shift.change,
                                  start, args.drift_factor, args.note)
        return 0
    if args.action == "remove":
        with session_scope() as session:
            found = store.delete_assumption(session, competition, season, args.team)
        logger.info("Removed" if found else "Nothing to remove")
        return 0
    stored = _stored_assumptions(competition, season)
    if not stored:
        print(f"\nNo assumptions stored for {competition} {season}.")
        return 0
    if args.action == "list":
        print(f"\nAssumptions for {competition} {season}\n")
        print(pd.DataFrame([a.__dict__ for a in stored]).to_string(index=False))
        return 0
    # check
    from football_predictor.model.strength import StrengthParams

    params = StrengthParams(model=args.model, xi=args.xi, prior_sd=args.prior_sd)
    checks = [check_shift(matches, a.team, a.change, a.start, params, a.drift_factor)
              for a in stored if a.change or a.drift_factor != 1]
    if not checks:
        print("\nThe stored assumptions say nothing (+0 %, drift factor 1) - nothing to check.")
        return 0
    out = checks_frame(checks)
    for col in ("assumed", "realized", "ci_low", "ci_high"):
        out[col] = (100 * out[col]).round(0)
    out["start"] = out["start"].dt.strftime("%Y-%m-%d")
    print(f"\nAssumption check {competition} {season} (changes in %, 95 % interval incl. "
          "baseline error; baseline fitted only on matches before the start date)\n")
    print(out.drop(columns=["drift_factor", "z", "implied_factor", "factor_gain",
                            "drift_verdict"]).round(3).to_string(index=False))
    print("\nDrift factor (z = deviation from the assumption / predictive sd; "
          "|z| > 1.96 -> too confident)\n")
    print(out[["team", "matches", "drift_factor", "z", "implied_factor", "factor_gain",
               "drift_verdict"]].round(2).to_string(index=False))
    cal = calibration(checks)
    if cal.n:
        pooled = f"{cal.pooled_factor:.2f}" if cal.pooled_factor is not None else "-"
        print(f"\nCalibration over {cal.n} checked assumption(s): mean z² {cal.mean_z2:.2f} "
              f"(1 = calibrated), outside ±1.96: {cal.outside:.0%} (expected 5 %), "
              f"p = {cal.p_value:.3f} -> {cal.verdict}; best-fitting drift factor {pooled}."
              + (" (Few assumptions: indicative only.)" if cal.n < 10 else ""))
    return 0


def _check_all_teams(args: argparse.Namespace) -> int:
    """'No change from DATE' for every team: a calibration test of the model itself.

    Nothing is stored. With --all-competitions the checks of all competitions that
    have the season are pooled (one row per competition, then the total).
    """
    import pandas as pd

    from football_predictor.db.session import session_scope
    from football_predictor.model.assumptions import calibration, no_change_checks
    from football_predictor.model.dataset import list_competition_seasons, load_matches
    from football_predictor.model.strength import StrengthParams

    try:
        start = pd.Timestamp(args.all_teams)
    except ValueError:
        logger.error("--all-teams needs a date like 2026-03-01, got %r", args.all_teams)
        return 2
    with session_scope() as session:
        if args.all_competitions:
            overview = list_competition_seasons(session)
            season = args.season or str(overview["season"].max())
            names = sorted(overview.loc[overview["season"] == season, "competition"])
            if args.competition:
                names = [c for c in names if c == args.competition]
        else:
            resolved = _resolve_competition_season(session, args.competition, args.season)
            if resolved is None:
                return 1
            names, season = [resolved[0]], resolved[1]
        data = {c: load_matches(session, c, season) for c in names}
    params = StrengthParams(model=args.model, xi=args.xi, prior_sd=args.prior_sd)
    per_comp = {c: no_change_checks(m, start, params) for c, m in data.items() if not m.empty}
    checks = [c for cs in per_comp.values() for c in cs]
    if not checks:
        print(f"\nNo team of {season} has played both before and after {start.date()}.")
        return 0

    def summary(cs: list) -> dict:
        cal = calibration(cs)
        return {"teams": cal.n,
                "direction_flagged": sum(c.verdict != "consistent" for c in cs) / len(cs),
                "mean_z2": cal.mean_z2, "outside": cal.outside, "p": cal.p_value,
                "pooled_k": cal.pooled_factor, "verdict": cal.verdict}

    print(f"\n'No change from {start.date()}' for every team, {season} "
          "(a test of the model's own uncertainty; nothing is stored)\n")
    if len(per_comp) > 1:
        rows = pd.DataFrame([{"competition": c, **summary(cs)} for c, cs in per_comp.items()])
        print(rows.round(3).to_string(index=False))
        print("\nPer-competition k is very noisy (about a dozen teams); read the total.")
    else:
        out = pd.DataFrame([{"team": c.team, "matches": c.matches,
                             "realized": round(100 * c.realized),
                             "ci_low": round(100 * c.ci_low), "ci_high": round(100 * c.ci_high),
                             "verdict": c.verdict, "z": c.z, "drift_verdict": c.drift_verdict}
                            for c in checks])
        out = out.reindex(out["z"].abs().sort_values(ascending=False).index)
        print("(changes in %, 95 % interval; sorted by |z|)\n")
        print(out.round(2).to_string(index=False))
    total = summary(checks)
    k = f"{total['pooled_k']:.2f}" if total["pooled_k"] is not None else "-"
    print(f"\nTotal over {total['teams']} teams: direction flagged "
          f"{total['direction_flagged']:.0%} (expected 5 % without real changes), mean z² "
          f"{total['mean_z2']:.2f} (1 = calibrated), outside ±1.96: {total['outside']:.0%}, "
          f"p = {total['p']:.3f} -> {total['verdict']}; best-fitting drift factor {k}.")
    if len(per_comp) > 1:
        top = sorted(checks, key=lambda c: -abs(c.z))[:5]
        print("Largest |z|: " + ", ".join(f"{c.team} {c.z:+.1f}" for c in top))
    return 0


def _point_adjustments(competition: str, season: str) -> dict[str, int]:
    """Stored point deductions of a competition-season (logged when present)."""
    from football_predictor.db.session import session_scope
    from football_predictor.model.dataset import load_point_adjustments

    with session_scope() as session:
        adjustments = load_point_adjustments(session, competition, season)
    if adjustments:
        logger.info("Point adjustments applied: %s", adjustments)
    return adjustments


def cmd_scorers(args: argparse.Namespace) -> int:
    """Anytime-scorer probabilities for the next round (or --round)."""
    import pandas as pd

    from football_predictor.model.predict import OPEN_STATUSES, catch_up_rounds, next_round
    from football_predictor.model.scorers import (
        ScorerParams,
        anonymize,
        fit_scorers,
        predict_fixture_scorers,
    )
    from football_predictor.model.strength import StrengthParams

    loaded = _load_competition(args.competition, args.season)
    if loaded is None:
        return 1
    competition, season, matches, apps, goals = loaded
    if apps.empty:
        logger.error("%s %s has no lineup data (scrape without --no-details)",
                     competition, season)
        return 1
    team_params = StrengthParams(model=args.model, xi=args.xi, prior_sd=args.prior_sd)
    params = ScorerParams(shrinkage=args.shrinkage, xi=args.scorer_xi,
                          form_matches=args.form_matches, dispersion=args.dispersion)

    if args.table:
        table = fit_scorers(matches, apps, goals, params).players
        table = table.sort_values(["goals", "share"], ascending=False).head(args.top)
        if args.anonymize:
            table["player"] = anonymize(table["player"])
            table["team"] = anonymize(table["team"], "Team")
        table["share"] = (table["share"] * 100).round(1)
        cols = ["player", "team", "apps", "minutes", "goals", "penalties", "share",
                "exp_minutes"]
        print(f"\n{competition} {season} - top scorers [{params.label()}]")
        print("share = % of the team's goals while on the pitch (shrunk)\n")
        print(table[cols].round(1).to_string(index=False))
        return 0

    rnd = args.round or next_round(matches)
    if rnd is None:
        logger.error("%s %s has no unplayed fixtures", competition, season)
        return 1
    rounds = [rnd] if args.round else [*catch_up_rounds(matches, rnd), rnd]
    fixtures = matches[matches["round"].isin(rounds) & matches["status"].isin(OPEN_STATUSES)]
    if args.team:
        fixtures = fixtures[(fixtures["home"] == args.team) | (fixtures["away"] == args.team)]
    pred = predict_fixture_scorers(matches, fixtures, apps, goals, team_params, params)
    if pred.empty:
        logger.error("No fixtures to predict")
        return 1
    if args.team:
        pred = pred[pred["team"] == args.team]
    if args.anonymize:
        all_teams = pd.concat([pred["team"], pred["opponent"]])
        codes = dict(zip(all_teams, anonymize(all_teams, "Team"), strict=True))
        pred["team"], pred["opponent"] = pred["team"].map(codes), pred["opponent"].map(codes)
        pred["player"] = anonymize(pred["player"])

    print(f"\n{competition} {season} - round {rnd} anytime scorers   "
          f"[{team_params.label()}, {params.label()}]")
    for (_, team), grp in pred.groupby(["match_id", "team"], sort=False):
        first = grp.iloc[0]
        venue = "home" if first["home"] else "away"
        print(f"\n{team} ({venue}) vs {first['opponent']} - team xG {first['team_xg']:.2f}")
        out = grp.head(args.top)[["player", "goals", "share", "exp_minutes", "xg",
                                  "p_score", "p_2plus"]].copy()
        out["share"] = (out["share"] * 100).round(1)
        out[["p_score", "p_2plus"]] = (out[["p_score", "p_2plus"]] * 100).round(0).astype(int)
        print(out.round(2).to_string(index=False))
    return 0


def cmd_backtest_scorers(args: argparse.Namespace) -> int:
    """Walk-forward evaluation of the scorer model; with --tune: tune on --tune-season."""
    from dataclasses import replace

    import pandas as pd

    from football_predictor.db.session import session_scope
    from football_predictor.model.dataset import (
        list_competition_seasons,
        load_appearances,
        load_goals,
        load_matches,
    )
    from football_predictor.model.scorers import (
        ScorerForecaster,
        ScorerParams,
        binary_calibration,
        scorer_grid,
        scorer_score_table,
        scorer_walk_forward,
    )
    from football_predictor.model.strength import StrengthParams

    team_params = StrengthParams(model="mle", xi=0.005, prior_sd=0.7)
    out_dir = Path("data/backtest")
    out_dir.mkdir(parents=True, exist_ok=True)

    def run(season: str, forecasters: list) -> pd.DataFrame:
        with session_scope() as session:
            overview = list_competition_seasons(session)
            comps = overview.loc[overview["season"] == season, "competition"]
            if args.competition:
                comps = comps[comps.isin(args.competition)]
            data = {c: (load_matches(session, c, season), load_appearances(session, c, season),
                        load_goals(session, c, season)) for c in comps}
        parts = []
        for comp, (m, a, g) in data.items():
            if a.empty:
                continue
            logger.info("Scorer backtest %s %s (%d forecasters)", comp, season,
                        len(forecasters))
            preds = scorer_walk_forward(m, a, g, forecasters, args.start_round)
            if not preds.empty:
                parts.append(preds.assign(competition=comp, season=season))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    best = ScorerParams(shrinkage=args.shrinkage, form_matches=args.form_matches,
                        dispersion=args.dispersion)
    if args.tune:
        grid = [ScorerForecaster(p.label(), p, team_params) for p in scorer_grid()]
        tuning = run(args.tune_season, grid)
        if tuning.empty:
            logger.error("No lineup data for tuning season %s", args.tune_season)
            return 1
        table = scorer_score_table(tuning)
        table.to_csv(out_dir / f"scorer_tuning_{args.tune_season.replace('/', '-')}.csv",
                     index=False)
        print(f"\nScorer tuning on {args.tune_season} (top 10 by log loss):")
        print(table.head(10).round(4).to_string(index=False))
        best = next(f.params for f in grid if f.name == table.iloc[0]["forecaster"])

    minutes_only = replace(best, use_skill=False)
    forecasters = [
        ScorerForecaster(best.label(), best, team_params),
        ScorerForecaster(best.label() + " + league-avg team xG", best, None),
        ScorerForecaster(minutes_only.label(), minutes_only, team_params),
    ]
    if best.dispersion is not None:
        poisson = replace(best, dispersion=None)
        forecasters.append(ScorerForecaster(poisson.label(), poisson, team_params))
    preds = run(args.season, forecasters)
    if preds.empty:
        logger.error("No lineup data for season %s", args.season)
        return 1
    tag = args.season.replace("/", "-")
    preds.to_csv(out_dir / f"scorer_predictions_{tag}.csv", index=False)
    held = f" (held-out: tuned on {args.tune_season})" if args.tune else ""
    print(f"\nScorer walk-forward backtest, season {args.season}{held}")
    print("one row = one expected player in one match; top1_hit = the team's most likely "
          "scorer scored\n")
    scores = scorer_score_table(preds)
    scores.to_csv(out_dir / f"scorer_scores_{tag}.csv", index=False)
    print(scores.round(4).to_string(index=False))
    print(f"\nCalibration ({best.label()}):")
    calib = binary_calibration(preds[preds["forecaster"] == best.label()])
    calib.to_csv(out_dir / f"scorer_calibration_{tag}.csv", index=False)
    print(calib.round(3).to_string(index=False))
    return 0


def cmd_table(args: argparse.Namespace) -> int:
    """Print the current league table."""
    from football_predictor.model.simulate import league_table

    loaded = _load_competition(args.competition, args.season)
    if loaded is None:
        return 1
    competition, season, matches, _, _ = loaded
    print(f"\n{competition} {season}\n")
    table = league_table(matches, _point_adjustments(competition, season))
    print(table.to_string(index=False))
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    """Monte Carlo simulation of the rest of the season."""
    from football_predictor.model.simulate import TeamShift, hide_results_from, simulate_season
    from football_predictor.model.strength import StrengthParams

    loaded = _load_competition(args.competition, args.season)
    if loaded is None:
        return 1
    competition, season, matches, _, _ = loaded
    if args.from_round:
        matches = hide_results_from(matches, args.from_round)
    params = StrengthParams(model=args.model, xi=args.xi, prior_sd=args.prior_sd)
    likelihood = args.model != "simple"
    uncertainty = likelihood and not args.no_uncertainty
    drift = args.drift if likelihood else 0.0
    teams = set(matches["home"]) | set(matches["away"])
    scales: dict[str, float] = {}
    shifts: dict[str, TeamShift] = {}
    if not args.no_assumptions:  # stored ones first; command-line flags override
        for a in _stored_assumptions(competition, season):
            if a.team in teams:
                if a.change:
                    shifts[a.team] = TeamShift(a.change, a.start)
                if a.drift_factor != 1:
                    scales[a.team] = a.drift_factor
                logger.info("Stored assumption: %s %+.0f%% from %s, drift x%g", a.team,
                            100 * a.change, f"{a.start:%Y-%m-%d}", a.drift_factor)
    for item in args.team_drift:
        name, _, factor = item.rpartition("=")
        if name not in teams or not factor:
            logger.error("--team-drift %r: expected TEAM=FACTOR with a team of this league", item)
            return 2
        scales[name] = float(factor)
    for item in args.team_shift:
        name, _, value = item.partition("=")
        try:
            shift = TeamShift.parse(value)
        except ValueError:
            shift = None
        if name not in teams or shift is None:
            logger.error("--team-shift %r: expected TEAM=+20%% or TEAM=-10%%@2027-03-01 "
                         "with a team of this league", item)
            return 2
        shifts[name] = shift
        since = f" from {shift.start:%Y-%m-%d}" if shift.start is not None else ""
        logger.info("Assumption: %s %+.0f%% strength%s", name, 100 * shift.change, since)
    result = simulate_season(matches, params, n_sims=args.sims, seed=args.seed,
                             uncertainty=uncertainty, drift=drift, drift_scales=scales,
                             shifts=shifts,
                             point_adjustments=_point_adjustments(competition, season))
    asof = f" as of round {args.from_round}" if args.from_round else ""
    mode = ("with" if uncertainty else "without") + " strength uncertainty"
    mode += f", drift {drift:g}/30 days" if drift else ", no drift"
    print(f"\n{competition} {season}{asof} - {result.n_sims} simulated seasons, "
          f"{result.remaining} remaining fixtures   [{params.label()}, {mode}]\n")
    out = result.summary.copy()
    pct = ["p_title", "p_top3", "p_last"]
    out[pct] = (out[pct] * 100).round(1)
    print(out.round(1).to_string(index=False))
    if args.positions:
        print("\nP(final position) in %:")
        print((result.positions * 100).round(0).astype(int).to_string())
    return 0


def cmd_backtest_simulation(args: argparse.Namespace) -> int:
    """Freeze finished seasons part-way and score the simulated final tables."""
    import pandas as pd

    from football_predictor.db.session import session_scope
    from football_predictor.model.dataset import list_competition_seasons, load_matches
    from football_predictor.model.predict import OPEN_STATUSES
    from football_predictor.model.simulate import backtest_simulation
    from football_predictor.model.strength import StrengthParams

    params = StrengthParams(model=args.model, xi=args.xi, prior_sd=args.prior_sd)
    with session_scope() as session:
        overview = list_competition_seasons(session)
        comps = overview.loc[overview["season"] == args.season, "competition"]
        if args.competition:
            comps = comps[comps.isin(args.competition)]
        data = {c: load_matches(session, c, args.season) for c in comps}
    frames = []
    for comp, matches in data.items():
        if matches["status"].isin(OPEN_STATUSES).any():
            logger.warning("Skipping %s %s: season not finished", comp, args.season)
            continue
        logger.info("Simulation backtest %s %s", comp, args.season)
        frames.append(backtest_simulation(matches, params, n_sims=args.sims, seed=args.seed,
                                          point_adjustments=_point_adjustments(comp, args.season),
                                          drift=args.drift)
                      .assign(competition=comp))
    if not frames:
        logger.error("Nothing to backtest")
        return 1
    per_league = pd.concat(frames, ignore_index=True)
    summary = per_league.groupby(["fraction", "strengths"], sort=False).agg(
        leagues=("competition", "size"), log_loss=("log_loss", "mean"), rps=("rps", "mean"),
        pos_mae=("pos_mae", "mean"), p_champion=("p_champion", "mean"),
        favourite_won=("favourite_won", "sum"),
    ).reset_index()
    print(f"\nSimulation backtest {args.season} [{params.label()}], {args.sims} sims; "
          "log_loss / rps per team over its final position (lower is better)\n")
    print(summary.round(4).to_string(index=False))
    out = Path("data/backtest")
    out.mkdir(parents=True, exist_ok=True)
    per_league.to_csv(out / f"simulation_{args.season.replace('/', '-')}.csv", index=False)
    return 0


def cmd_tune_drift(args: argparse.Namespace) -> int:
    """Estimate the strength drift σ (and test match-level noise ω) from multi-round-ahead
    predictions of every match of a season."""
    import pandas as pd

    from football_predictor.db.session import session_scope
    from football_predictor.model.dataset import (
        list_competition_seasons,
        load_appearances,
        load_matches,
    )
    from football_predictor.model.drift import drift_log_scores, drift_summary
    from football_predictor.model.strength import StrengthParams

    params = StrengthParams(model=args.model, xi=args.xi, prior_sd=args.prior_sd)
    with session_scope() as session:
        overview = list_competition_seasons(session)
        comps = overview.loc[overview["season"] == args.season, "competition"]
        if args.competition:
            comps = comps[comps.isin(args.competition)]
        data = {c: (load_matches(session, c, args.season),
                    load_appearances(session, c, args.season)) for c in comps}
    parts = []
    for comp, (matches, apps) in data.items():
        logger.info("Drift scoring %s %s", comp, args.season)
        parts.append(drift_log_scores(matches, params, sigmas=tuple(args.sigmas),
                                      omegas=tuple(args.omegas), start_round=args.start_round,
                                      n_draws=args.draws, appearances=apps,
                                      betas=tuple(args.betas), covariate=args.covariate,
                                      taus=tuple(args.taus), lags=tuple(args.lags))
                     .assign(competition=comp))
    if not parts:
        logger.error("Nothing to score")
        return 1
    scores = pd.concat(parts, ignore_index=True)
    summary = drift_summary(scores)
    print(f"\nDrift tuning {args.season} [{params.label()}]: log loss of the exact score, "
          "per forecast horizon (sigma = drift per 30 days, omega = match-level noise, "
          f"beta = team-specific drift by {args.covariate})\n")
    print(summary.round(4).to_string(index=False))
    best = summary.loc[summary["log_loss"].idxmin()]
    print(f"\nbest: sigma={best['sigma']:g}, omega={best['omega']:g}, beta={best['beta']:g}, "
          f"tau={best['tau']:g}, lag={best['lag']:g}")
    out = Path("data/backtest")
    out.mkdir(parents=True, exist_ok=True)
    scores.to_csv(out / f"drift_{args.season.replace('/', '-')}.csv", index=False)
    return 0


def cmd_add_alias(args: argparse.Namespace) -> int:
    """Register an alternative spelling of an existing team."""
    from football_predictor.db.session import session_scope
    from football_predictor.ingest.loader import add_team_alias

    try:
        with session_scope() as session:
            add_team_alias(session, args.alias, args.team)
    except ValueError as exc:
        logger.error("%s", exc)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with all sub-commands."""
    parser = argparse.ArgumentParser(
        prog="football-predictor",
        description="Amateur football league data & Poisson predictions.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-db", help="create or migrate the database schema")
    p_init.add_argument(
        "--drop", action="store_true", help="drop existing tables first (DESTROYS DATA)"
    )
    p_init.set_defaults(func=cmd_init_db)

    p_db = sub.add_parser("db", help="schema migrations (Alembic)")
    p_db.add_argument("action", choices=["current", "upgrade", "downgrade", "revision", "check"],
                      help="current: show revision; upgrade: apply pending migrations; "
                           "downgrade REV: revert; revision -m MSG: autogenerate a new "
                           "migration from model changes; check: compare schema with models")
    p_db.add_argument("revision", nargs="?", help="target for downgrade (e.g. -1, base)")
    p_db.add_argument("-m", "--message", help="description for a new revision")
    p_db.set_defaults(func=cmd_db)

    p_val = sub.add_parser("validate", help="check goal rows against final scores")
    p_val.add_argument("--season-id", type=int, default=None)
    p_val.set_defaults(func=cmd_validate)

    p_gen = sub.add_parser("generate-sample", help="generate and load a FICTIONAL demo league")
    p_gen.add_argument("--teams", type=int, default=10)
    p_gen.add_argument("--played-rounds", type=int, default=14,
                       help="rounds with results; the rest stay scheduled")
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.add_argument("--no-appearances", action="store_true",
                       help="do not generate line-up (minutes played) data")
    p_gen.add_argument("--truth-out", type=Path, default=Path("data/sample_truth.json"))
    p_gen.set_defaults(func=cmd_generate_sample)

    p_csv = sub.add_parser("import-csv", help="import matches/goals/appearances from CSV")
    p_csv.add_argument("matches", type=Path, help="matches CSV file")
    p_csv.add_argument("--goals", type=Path, default=None)
    p_csv.add_argument("--appearances", type=Path, default=None)
    p_csv.add_argument("--season", default=None, help="season name if not in the CSV")
    p_csv.add_argument("--competition", default=DEFAULT_COMPETITION,
                       help="competition name if not in the CSV")
    p_csv.set_defaults(func=cmd_import_csv)

    p_scr = sub.add_parser("scrape", help="scrape fixture pages defined in a TOML config")
    p_scr.add_argument("--config", type=Path, required=True)
    p_scr.add_argument("--refresh", action="store_true", help="re-download cached pages")
    p_scr.add_argument("--offline", action="store_true",
                       help="only re-parse pages already cached in data/raw")
    p_scr.set_defaults(func=cmd_scrape)

    p_mlsz = sub.add_parser("scrape-mlsz", help="scrape competitions from adatbank.mlsz.hu")
    p_mlsz.add_argument("--config", type=Path, default=Path("config/mlsz_gyms.toml"))
    p_mlsz.add_argument("--season", action="append", default=None,
                        help="season name, e.g. 2026/27 (repeatable; default: first in config)")
    p_mlsz.add_argument("--competition", action="append", default=None,
                        help='competition name, e.g. "GYMS Megye II Észak" (repeatable)')
    p_mlsz.add_argument("--no-details", action="store_true",
                        help="results only: skip match pages (no scorers/line-ups, much faster)")
    p_mlsz.add_argument("--refresh", action="store_true", help="re-download everything")
    p_mlsz.add_argument("--offline", action="store_true", help="only re-parse cached pages")
    p_mlsz.set_defaults(func=cmd_scrape_mlsz)

    p_comp = sub.add_parser("competitions", help="list competitions/seasons in the database")
    p_comp.set_defaults(func=cmd_competitions)

    p_pred = sub.add_parser("predict", help="predict a round (default: the next open round)")
    p_pred.add_argument("--competition", default=None)
    p_pred.add_argument("--season", default=None, help="default: latest season")
    p_pred.add_argument("--round", type=int, default=None)
    p_pred.add_argument("--xi", type=float, default=0.005,
                        help="time decay per day (0 = none; 0.005 = half-life ~139 days, "
                             "tuned on 2025/26)")
    p_pred.add_argument("--shrinkage", type=float, default=2.0,
                        help="pseudo-matches of average performance per team")
    p_pred.add_argument("--last-n", type=int, default=None,
                        help="only use each team's last N matches")
    p_pred.add_argument("--show-strengths", action="store_true")
    p_pred.add_argument("--model", choices=["simple", "mle", "dc"], default="mle",
                        help="default mle: best on the 2025/26 tuning season")
    p_pred.add_argument("--prior-sd", type=float, default=0.7,
                        help="mle/dc: sd of the Normal prior on log-strengths (tuned)")
    p_pred.set_defaults(func=cmd_predict)

    p_bt = sub.add_parser("backtest", help="walk-forward evaluation of models vs. baselines")
    p_bt.add_argument("--season", default="2026/27", help="season to evaluate")
    p_bt.add_argument("--competition", action="append", default=None,
                      help="restrict to competition(s); default: all in that season")
    p_bt.add_argument("--start-round", type=int, default=4,
                      help="first round to score (earlier rounds have too little data)")
    p_bt.add_argument("--tune", action="store_true",
                      help="grid-search parameters on --tune-season, then evaluate on --season")
    p_bt.add_argument("--tune-season", default="2025/26")
    p_bt.set_defaults(func=cmd_backtest)

    p_sc = sub.add_parser("scorers", help="anytime-scorer probabilities for the next round")
    p_sc.add_argument("--competition", default=None)
    p_sc.add_argument("--season", default=None)
    p_sc.add_argument("--round", type=int, default=None)
    p_sc.add_argument("--team", default=None, help="only this team's fixture")
    p_sc.add_argument("--top", type=int, default=5, help="players per team")
    p_sc.add_argument("--table", action="store_true", help="season top-scorer table instead")
    p_sc.add_argument("--anonymize", action="store_true",
                      help="replace player/team names (for screenshots, public demos)")
    p_sc.add_argument("--shrinkage", type=float, default=SCORER_DEFAULTS["shrinkage"],
                      help="pseudo team-goals of exposure at the league-average share")
    p_sc.add_argument("--scorer-xi", type=float, default=0.0, help="time decay of shares")
    p_sc.add_argument("--form-matches", type=int, default=SCORER_DEFAULTS["form_matches"],
                      help="expected minutes from the team's last N matches")
    p_sc.add_argument("--dispersion", type=float, default=SCORER_DEFAULTS["dispersion"],
                      help="negative binomial size r (smaller = goals cluster more)")
    p_sc.add_argument("--model", choices=["simple", "mle", "dc"], default="mle")
    p_sc.add_argument("--xi", type=float, default=0.005, help="team model time decay")
    p_sc.add_argument("--prior-sd", type=float, default=0.7)
    p_sc.set_defaults(func=cmd_scorers)

    p_bts = sub.add_parser("backtest-scorers", help="walk-forward evaluation of scorer model")
    p_bts.add_argument("--season", default="2026/27")
    p_bts.add_argument("--competition", action="append", default=None)
    p_bts.add_argument("--start-round", type=int, default=4)
    p_bts.add_argument("--tune", action="store_true")
    p_bts.add_argument("--tune-season", default="2025/26")
    p_bts.add_argument("--shrinkage", type=float, default=SCORER_DEFAULTS["shrinkage"])
    p_bts.add_argument("--form-matches", type=int, default=SCORER_DEFAULTS["form_matches"])
    p_bts.add_argument("--dispersion", type=float, default=SCORER_DEFAULTS["dispersion"],
                       help="negative binomial size r (omit for the tuned default)")
    p_bts.set_defaults(func=cmd_backtest_scorers)

    p_tab = sub.add_parser("table", help="current league table")
    p_tab.add_argument("--competition", default=None)
    p_tab.add_argument("--season", default=None)
    p_tab.set_defaults(func=cmd_table)

    p_sim = sub.add_parser("simulate", help="Monte Carlo simulation of the rest of the season")
    p_sim.add_argument("--competition", default=None)
    p_sim.add_argument("--season", default=None)
    p_sim.add_argument("--sims", type=int, default=10_000)
    p_sim.add_argument("--seed", type=int, default=0)
    p_sim.add_argument("--from-round", type=int, default=None,
                       help="pretend this round and later are unplayed (backtest)")
    p_sim.add_argument("--positions", action="store_true", help="full position distribution")
    p_sim.add_argument("--model", choices=["simple", "mle", "dc"], default="mle")
    p_sim.add_argument("--xi", type=float, default=0.005)
    p_sim.add_argument("--prior-sd", type=float, default=0.7)
    p_sim.add_argument("--no-uncertainty", action="store_true",
                       help="treat the fitted team strengths as exact (old behaviour)")
    p_sim.add_argument("--drift", type=float, default=DRIFT_DEFAULT,
                       help="strength random-walk sd per 30 days (0 = strengths stay fixed)")
    p_sim.add_argument("--team-drift", action="append", default=[], metavar="TEAM=FACTOR",
                       help="multiply the drift of one team, e.g. 'X SE=2' for a squad "
                            "rebuild you know about (repeatable)")
    p_sim.add_argument("--no-assumptions", action="store_true",
                       help="ignore assumptions stored with `assume add`")
    p_sim.add_argument("--team-shift", action="append", default=[],
                       metavar="TEAM=CHANGE[@DATE]",
                       help="expected strength change of one team, e.g. 'X SE=+20%%@2027-03-01' "
                            "(scores 1.2x, concedes 1/1.2x from that date; repeatable)")
    p_sim.set_defaults(func=cmd_simulate)

    p_as = sub.add_parser("assume", help="store / list / remove / check expert assumptions")
    p_as.add_argument("action", choices=["add", "list", "remove", "check"])
    p_as.add_argument("team", nargs="?", help="team name (add, remove)")
    p_as.add_argument("change", nargs="?", help="e.g. +20%%, -10%%@2027-03-01 (add)")
    p_as.add_argument("--competition", default=None)
    p_as.add_argument("--season", default=None)
    p_as.add_argument("--drift-factor", type=float, default=1.0,
                      help="also widen this team's form random walk (add)")
    p_as.add_argument("--note", default=None, help="why you assume this (add)")
    p_as.add_argument("--all-teams", metavar="DATE", default=None,
                      help="check: test 'no change from DATE' for every team (stores nothing)")
    p_as.add_argument("--all-competitions", action="store_true",
                      help="check --all-teams: pool every competition of the season")
    p_as.add_argument("--model", choices=["simple", "mle", "dc"], default="mle")
    p_as.add_argument("--xi", type=float, default=0.005)
    p_as.add_argument("--prior-sd", type=float, default=0.7)
    p_as.set_defaults(func=cmd_assume)

    p_bsim = sub.add_parser("backtest-simulation",
                            help="score season simulations against finished final tables")
    p_bsim.add_argument("--season", default="2025/26")
    p_bsim.add_argument("--competition", action="append", default=None)
    p_bsim.add_argument("--sims", type=int, default=5000)
    p_bsim.add_argument("--seed", type=int, default=0)
    p_bsim.add_argument("--model", choices=["mle", "dc"], default="mle")
    p_bsim.add_argument("--xi", type=float, default=0.005)
    p_bsim.add_argument("--prior-sd", type=float, default=0.7)
    p_bsim.add_argument("--drift", type=float, default=DRIFT_DEFAULT)
    p_bsim.set_defaults(func=cmd_backtest_simulation)

    p_drift = sub.add_parser("tune-drift",
                             help="estimate how fast team strengths change during a season")
    p_drift.add_argument("--season", default="2025/26")
    p_drift.add_argument("--competition", action="append", default=None)
    p_drift.add_argument("--sigmas", type=float, nargs="+",
                         default=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3])
    p_drift.add_argument("--omegas", type=float, nargs="+", default=[0.0])
    p_drift.add_argument("--betas", type=float, nargs="+", default=[0.0],
                         help="team-specific drift strengths to test (0 = same σ for all)")
    p_drift.add_argument("--taus", type=float, nargs="+", default=[0.0],
                         help="persistent per-team jumps at the freeze to test (0 = none)")
    p_drift.add_argument("--lags", type=float, nargs="+", default=[0.0, 1.0],
                         help="multipliers of the fit's lag behind drifting teams "
                              "(1 = what the random walk implies, used by simulate)")
    p_drift.add_argument("--covariate", choices=["turnover", "past_change"], default="turnover",
                         help="team measure behind beta: squad turnover or past strength change")
    p_drift.add_argument("--start-round", type=int, default=5)
    p_drift.add_argument("--draws", type=int, default=400)
    p_drift.add_argument("--model", choices=["mle", "dc"], default="mle")
    p_drift.add_argument("--xi", type=float, default=0.005)
    p_drift.add_argument("--prior-sd", type=float, default=0.7)
    p_drift.set_defaults(func=cmd_tune_drift)

    p_alias = sub.add_parser("add-alias", help="map an alternative team name to a team")
    p_alias.add_argument("alias")
    p_alias.add_argument("team", help="existing team name")
    p_alias.set_defaults(func=cmd_add_alias)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI main function; returns the process exit code."""
    setup_logging(get_settings().log_level)
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
