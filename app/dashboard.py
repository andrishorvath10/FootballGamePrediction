"""Streamlit dashboard: pick a league, see predictions, scorers, table and simulation.

Run:  streamlit run app/dashboard.py

All numbers come from the same functions as the CLI (``football_predictor.model``),
so the dashboard and ``football-predictor predict`` always agree.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from football_predictor.config import get_settings
from football_predictor.db.assumption_store import (
    StoredAssumption,
    list_assumptions,
    replace_assumptions,
)
from football_predictor.db.session import session_scope
from football_predictor.model.assumptions import calibration, check_shift
from football_predictor.model.dataset import (
    list_competition_seasons,
    load_appearances,
    load_goals,
    load_matches,
    load_point_adjustments,
    played_only,
)
from football_predictor.model.drift import DRIFT_DEFAULT
from football_predictor.model.poisson import predict_match
from football_predictor.model.predict import OPEN_STATUSES, catch_up_rounds, next_round
from football_predictor.model.scorers import ScorerParams, fit_scorers, scorer_probabilities
from football_predictor.model.simulate import TeamShift, league_table, simulate_season
from football_predictor.model.strength import StrengthParams, TeamStrengths, fit_model

logging.getLogger("football_predictor").setLevel(logging.WARNING)

# Colours (validated reference palette): home / draw / away is an ordered, polar
# scale -> two poles + neutral midpoint; magnitudes use one blue ramp.
HOME, DRAW, AWAY = "#2a78d6", "#9a9890", "#eb6834"
BLUE_RAMP = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
             "#0d366b"]
BACKTEST_DIR = Path("data/backtest")

TEAM_DEFAULTS = StrengthParams(model="mle", xi=0.005, prior_sd=0.7)  # tuned on 2025/26
SCORER_DEFAULTS = ScorerParams(shrinkage=8.0, form_matches=5, dispersion=1.0)  # tuned

st.set_page_config(page_title="Amatőr foci előrejelző", page_icon="⚽", layout="wide")


# ------------------------------------------------------------------ data access
@st.cache_data(ttl=600)
def overview() -> pd.DataFrame:
    with session_scope() as s:
        return list_competition_seasons(s)


@st.cache_data(ttl=600)
def league_data(competition: str, season: str) -> tuple[pd.DataFrame, ...]:
    with session_scope() as s:
        return (load_matches(s, competition, season), load_appearances(s, competition, season),
                load_goals(s, competition, season))


@st.cache_data(ttl=600)
def point_adjustments(competition: str, season: str) -> dict[str, int]:
    with session_scope() as s:
        return load_point_adjustments(s, competition, season)


@st.cache_data(ttl=600)
def team_model(competition: str, season: str, params: StrengthParams) -> TeamStrengths:
    matches = league_data(competition, season)[0]
    teams = set(matches["home"]) | set(matches["away"])
    return fit_model(played_only(matches), params, teams=teams)


@st.cache_data(ttl=600)
def simulation(competition: str, season: str, params: StrengthParams, n_sims: int,  # noqa: ANN201
               uncertainty: bool, drift: float,
               assumptions: tuple[tuple[str, float, str | None, float], ...] = ()):
    """``assumptions``: (team, change, start ISO date or None, drift factor) rows."""
    shifts = {t: TeamShift(change, pd.Timestamp(start) if start else None)
              for t, change, start, _ in assumptions if change}
    scales = {t: factor for t, _, _, factor in assumptions if factor != 1}
    return simulate_season(league_data(competition, season)[0], params, n_sims=n_sims,
                           uncertainty=uncertainty, drift=drift, drift_scales=scales,
                           shifts=shifts, point_adjustments=point_adjustments(competition, season))


# ---------------------------------------------------------------- anonymisation
class Names:
    """Maps real names to pseudonyms when anonymisation is on (identity otherwise)."""

    def __init__(self, teams: list[str], players: pd.DataFrame, on: bool) -> None:
        self.on = on
        self.teams = {t: f"Csapat {chr(65 + i)}" if i < 26 else f"Csapat {i + 1}"
                      for i, t in enumerate(sorted(teams))}
        ids = sorted(players["player_id"].unique()) if not players.empty else []
        self.players = {int(p): f"Játékos {i + 1:03d}" for i, p in enumerate(ids)}

    def team(self, name: str) -> str:
        return self.teams.get(name, name) if self.on else name

    def player(self, pid: int, name: str) -> str:
        return self.players.get(int(pid), "Játékos ?") if self.on else name

    def frame(self, df: pd.DataFrame, team_cols: tuple[str, ...] = ("team",)) -> pd.DataFrame:
        if not self.on:
            return df
        df = df.copy()
        for c in team_cols:
            if c in df:
                df[c] = df[c].map(self.team)
        if "player" in df and "player_id" in df:
            df["player"] = [self.player(p, n) for p, n in zip(df["player_id"], df["player"],
                                                                strict=True)]
        return df


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def to_pct(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Probabilities 0..1 -> percentages 0..100 (compact display in tables)."""
    df = df.copy()
    df[cols] = df[cols] * 100
    return df


PCT = st.column_config.NumberColumn(format="%.0f%%", width="small")
PCT1 = st.column_config.NumberColumn(format="%.1f%%", width="small")
PCT_BAR = st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100,
                                          width="small")


# ------------------------------------------------------------------------ sidebar
ov = overview()
if ov.empty:
    st.error("Üres az adatbázis. Futtasd: `football-predictor generate-sample` vagy "
             "`football-predictor scrape-mlsz`.")
    st.stop()

with st.sidebar:
    st.header("⚽ Bajnokság")
    # real seasons newest first ("2026/27" > "2025/26"), synthetic demo data last
    seasons = sorted(ov["season"].unique(),
                     key=lambda x: (not x[0].isdigit(), [-ord(c) for c in x]))
    qp = st.query_params  # e.g. ?season=2026/27&competition=GYMS Megye I (shareable links)
    season = st.selectbox("Szezon", seasons, key="season",
                          index=seasons.index(qp["season"]) if qp.get("season") in seasons
                          else 0)
    comps = ov.loc[ov["season"] == season, "competition"].tolist()
    default_comp = (comps.index(qp["competition"]) if qp.get("competition") in comps
                    else next((i for i, c in enumerate(comps) if "II Észak" in c), 0))
    competition = st.selectbox("Bajnokság", comps, index=default_comp, key="competition")
    anonymize = st.toggle("Nevek anonimizálása", value=get_settings().anonymize, key="anon",
                          help="Játékos- és csapatnevek elrejtése (képernyőképekhez, "
                               "nyilvános bemutatóhoz).")

    with st.expander("Modell beállítások"):
        model = st.selectbox("Csapatmodell", ["mle", "simple", "dc"], key="model",
                             help="mle: közös maximum likelihood (alapértelmezett, hangolt); "
                                  "simple: átlagok aránya; dc: mle + Dixon–Coles")
        xi = st.slider("Időbeli súlyozás ξ (1/nap)", 0.0, 0.02, TEAM_DEFAULTS.xi, 0.001,
                       format="%.3f", help="0.005 ≈ 139 napos felezési idő")
        if model == "simple":
            shrink = st.slider("Zsugorítás k (álmeccs)", 0.0, 10.0, 2.0, 0.5)
            team_params = StrengthParams(model=model, xi=xi, shrinkage=shrink)
        else:
            sd = st.slider("Prior szórás", 0.1, 1.5, TEAM_DEFAULTS.prior_sd, 0.05)
            team_params = StrengthParams(model=model, xi=xi, prior_sd=sd)
    st.caption(f"Modell: `{team_params.label()}`")

matches, apps, goals = league_data(competition, season)
played = played_only(matches)
names = Names(sorted(set(matches["home"]) | set(matches["away"])), apps, anonymize)

st.title(f"{competition} · {season}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Lejátszott meccs", f"{len(played)} / {len(matches)}")
c2.metric("Gól / meccs", f"{(played['home_goals'] + played['away_goals']).mean():.2f}"
          if not played.empty else "–")
c3.metric("Hazai győzelem", pct((played["home_goals"] > played["away_goals"]).mean())
          if not played.empty else "–")
c4.metric("Döntetlen", pct((played["home_goals"] == played["away_goals"]).mean())
          if not played.empty else "–")

if len(played) < 5:
    st.warning("Túl kevés lejátszott meccs a predikcióhoz.")
    st.stop()

strengths = team_model(competition, season, team_params)
tab_pred, tab_table, tab_scorers, tab_strength, tab_results, tab_quality = st.tabs(
    ["🔮 Következő forduló", "📊 Tabella és szimuláció", "🥅 Gólszerzők",
     "💪 Csapaterősségek", "📅 Eredmények", "✅ Modell minősége"])


# ---------------------------------------------------------------- next round tab
def outcome_bar(p_home: float, p_draw: float, p_away: float) -> go.Figure:
    fig = go.Figure()
    for label, value, color in (("Hazai", p_home, HOME), ("Döntetlen", p_draw, DRAW),
                                ("Vendég", p_away, AWAY)):
        fig.add_bar(y=[""], x=[value], orientation="h", name=label, marker_color=color,
                    text=f"{label} {pct(value)}", textposition="inside",
                    insidetextanchor="middle", hovertemplate=f"{label}: %{{x:.1%}}<extra></extra>",
                    marker_line_color="rgba(0,0,0,0)")
    fig.update_layout(barmode="stack", height=48, margin=dict(l=0, r=0, t=0, b=0),
                      showlegend=False, xaxis=dict(visible=False, range=[0, 1]),
                      yaxis=dict(visible=False), bargap=0)
    return fig


def score_heatmap(matrix: np.ndarray, home: str, away: str) -> go.Figure:
    n = 7
    m = matrix[:n, :n] * 100
    fig = go.Figure(go.Heatmap(
        z=m, x=list(range(n)), y=list(range(n)), colorscale=BLUE_RAMP, showscale=False,
        text=np.round(m, 1), texttemplate="%{text}", xgap=2, ygap=2,
        hovertemplate=f"{home} %{{y}} – %{{x}} {away}: %{{z:.1f}}%<extra></extra>"))
    fig.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis=dict(title=f"{away} gólja", dtick=1, side="top"),
                      yaxis=dict(title=f"{home} gólja", dtick=1, autorange="reversed"))
    return fig


with tab_pred:
    rnd = next_round(matches)
    if rnd is None:
        st.info("Nincs hátralévő mérkőzés ebben a szezonban.")
    else:
        rounds = [*catch_up_rounds(matches, rnd), rnd]
        fixtures = matches[matches["round"].isin(rounds) & matches["status"].isin(OPEN_STATUSES)]
        preds = []
        for row in fixtures.itertuples():
            lam, mu = strengths.expected_goals(row.home, row.away)
            preds.append((row, predict_match(row.home, row.away, lam, mu, rho=strengths.rho)))
        catch = (" + pótolt meccsek (" + ", ".join(f"{r}." for r in rounds[:-1])
                 + " forduló)") if len(rounds) > 1 else ""
        st.subheader(f"{rnd}. forduló{catch}")
        table = pd.DataFrame([{
            "Forduló": row.round,
            "Dátum": row.date.date() if pd.notna(row.date) else None,
            "Hazai": names.team(row.home), "Vendég": names.team(row.away),
            "xG": f"{p.lam:.2f} – {p.mu:.2f}",
            "1": p.p_home, "X": p.p_draw, "2": p.p_away,
            "Tipp": p.most_likely_outcome,
            "Legvalószínűbb": f"{p.top_scores[0][0]}–{p.top_scores[0][1]}",
            "2,5 felett": p.p_over_2_5, "GG": p.p_btts,
        } for row, p in preds])
        table = to_pct(table, ["1", "X", "2", "2,5 felett", "GG"])
        if table["Forduló"].nunique() == 1:
            table = table.drop(columns="Forduló")
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "1": PCT_BAR, "X": PCT_BAR, "2": PCT_BAR,
            "2,5 felett": PCT,
            "GG": st.column_config.NumberColumn("GG", format="%.0f%%", width="small",
                                                help="Mindkét csapat szerez gólt"),
            "Forduló": st.column_config.NumberColumn(width="small"),
            "Hazai": st.column_config.TextColumn(width="medium"),
            "Vendég": st.column_config.TextColumn(width="medium"),
        })

        st.divider()
        labels = [f"{names.team(r.home)} – {names.team(r.away)}" for r, _ in preds]
        choice = st.selectbox("Mérkőzés részletei", range(len(preds)),
                              format_func=lambda i: labels[i])
        row, p = preds[choice]
        h, a = names.team(row.home), names.team(row.away)
        st.plotly_chart(outcome_bar(p.p_home, p.p_draw, p.p_away), width="stretch",
                        config={"displayModeBar": False})
        left, right = st.columns([3, 2])
        with left:
            st.markdown("**Pontos eredmény valószínűsége (%)**")
            st.plotly_chart(score_heatmap(p.matrix, h, a), width="stretch",
                            config={"displayModeBar": False})
        with right:
            st.markdown(f"**Várható gólok:** {h} **{p.lam:.2f}** – **{p.mu:.2f}** {a}")
            st.markdown("**Leggyakoribb eredmények**")
            st.dataframe(pd.DataFrame([{"Eredmény": f"{i}–{j}", "Valószínűség": 100 * q}
                                       for i, j, q in p.top_scores]),
                         hide_index=True, width="stretch",
                         column_config={"Valószínűség": PCT1})
            st.markdown(f"2,5 gól felett: **{pct(p.p_over_2_5)}** · "
                        f"Mindkét csapat szerez gólt: **{pct(p.p_btts)}**")

        if not apps.empty:
            st.markdown("#### Valószínű gólszerzők")
            sc_model = fit_scorers(played, apps, goals, SCORER_DEFAULTS)
            cols = st.columns(2)
            for col, team, xg in ((cols[0], row.home, p.lam), (cols[1], row.away, p.mu)):
                probs = to_pct(names.frame(scorer_probabilities(sc_model, team, xg).head(6)),
                               ["p_score", "p_2plus"])
                col.markdown(f"**{names.team(team)}** (csapat xG {xg:.2f})")
                col.dataframe(
                    probs[["player", "goals", "exp_minutes", "p_score", "p_2plus"]].rename(
                        columns={"player": "Játékos", "goals": "Gól eddig",
                                 "exp_minutes": "Várható perc", "p_score": "Betalál",
                                 "p_2plus": "2+ gól"}),
                    hide_index=True, width="stretch", column_config={
                        "Várható perc": st.column_config.NumberColumn(format="%.0f"),
                        "Betalál": PCT_BAR, "2+ gól": PCT})


# ------------------------------------------------------------------ table tab
with tab_table:
    left, right = st.container(), st.container()
    with left:
        st.subheader("Jelenlegi tabella")
        lt = names.frame(league_table(matches, point_adjustments(competition, season)))
        if not lt["adjustment"].any():
            lt = lt.drop(columns="adjustment")
        st.dataframe(lt.rename(columns={
            "pos": "#", "team": "Csapat", "played": "M", "won": "GY", "drawn": "D",
            "lost": "V", "gf": "LG", "ga": "KG", "gd": "GK", "adjustment": "Levonás",
            "points": "P"}),
            hide_index=True, width="stretch", height=38 * (len(lt) + 1))
        st.caption("Holtverseny az MLSZ versenykiírás szerint: több győzelem, gólkülönbség, "
                   "lőtt gól, majd az egymás elleni pontok és gólkülönbség. A pontlevonások "
                   "(büntetőpont) benne vannak a P oszlopban.")
    with right:
        st.subheader("Monte Carlo szimuláció")
        n_sims = st.select_slider("Szimulált szezonok", [1000, 5000, 10000, 20000], 10000)
        has_likelihood = team_params.model != "simple"
        uncertainty = st.toggle(
            "Erősség-bizonytalanság", value=has_likelihood, disabled=not has_likelihood,
            key="uncertainty",
            help="Minden szimulált szezon a csapaterősségeket is újrasorsolja a becslés "
                 "bizonytalansága szerint (Laplace-közelítés). Csak az MLE/DC modellnél.")
        form_drift = st.toggle(
            "Formaváltozás a szezon során", value=has_likelihood, disabled=not has_likelihood,
            key="drift",
            help="A csapaterősségek véletlen bolyongással változnak a hátralévő hetekben "
                 f"(havonta kb. ±{DRIFT_DEFAULT:.0%} a log-skálán), így a tavaszi meccsek "
                 "bizonytalanabbak a jövő hétieknél. A téli szünet után azt is figyelembe "
                 "veszi, hogy a becslés még nagyrészt az őszi (régi) meccsekre épül.")
        with st.expander("Saját feltevések (pl. téli igazolás, edzőváltás)"):
            st.caption("Változás: +20% = a csapat 1,2-szer annyi gólt lő és 1/1,2 annyit kap, "
                       "a megadott naptól (üresen: azonnal). Bizonytalanság ×: a csapat "
                       "formaváltozását szorozza, ha nem tudod, merre megy majd. A mentett "
                       "feltevéseket később a valódi eredményekkel ellenőrzi.")
            display = {names.team(t): t for t in set(matches["home"]) | set(matches["away"])}
            with session_scope() as s_:
                stored = list_assumptions(s_, competition, season)
            editor_key = f"assumptions|{competition}|{season}"
            edited = st.data_editor(
                pd.DataFrame({
                    "Csapat": pd.Series([names.team(a.team) for a in stored], dtype="object"),
                    "Változás (%)": pd.Series([100 * a.change for a in stored], dtype="float"),
                    "Ettől a naptól": pd.Series([a.start.date() for a in stored], dtype="object"),
                    "Bizonytalanság ×": pd.Series([a.drift_factor for a in stored],
                                                  dtype="float")}),
                key=editor_key, num_rows="dynamic", hide_index=True, width="stretch",
                column_config={
                    "Csapat": st.column_config.SelectboxColumn(options=sorted(display)),
                    "Változás (%)": st.column_config.NumberColumn(
                        min_value=-50, max_value=100, step=5, default=0, format="%+d%%"),
                    "Ettől a naptól": st.column_config.DateColumn(format="YYYY-MM-DD"),
                    "Bizonytalanság ×": st.column_config.NumberColumn(
                        min_value=1.0, max_value=5.0, step=0.5, default=1.0),
                })
            assumptions = tuple(
                (display[row["Csapat"]],
                 float(row["Változás (%)"] or 0) / 100,
                 None if pd.isna(row["Ettől a naptól"])
                 else pd.Timestamp(row["Ettől a naptól"]).date().isoformat(),
                 float(row["Bizonytalanság ×"]) if pd.notna(row["Bizonytalanság ×"]) else 1.0)
                for _, row in edited.iterrows() if row["Csapat"] in display)
            if st.button("Feltevések mentése", key="save_assumptions",
                         help="Elmenti az adatbázisba (a valódi nevekkel), hogy később "
                              "ellenőrizni lehessen őket. Üres dátum = mától."):
                today = pd.Timestamp.today().normalize()
                with session_scope() as s_:
                    replace_assumptions(s_, competition, season, [
                        StoredAssumption(team, change, pd.Timestamp(start) if start else today,
                                         factor) for team, change, start, factor in assumptions])
                st.session_state.pop(editor_key, None)  # the saved rows are now the base
                st.rerun()
            checks = [check_shift(matches, a.team, a.change, a.start, team_params,
                                  a.drift_factor)
                      for a in stored if a.change or a.drift_factor != 1]
            if checks:
                st.markdown("**Mentett feltevések ellenőrzése**")
                st.caption("Alapvonal: a modell csak a kezdőnap előtti meccsekből. Tényleges = "
                           "a kezdőnap óta lejátszott meccsekből becsült eltolás, 95%-os "
                           "intervallummal (az alapvonal becslési hibájával együtt). Log-score: "
                           "+ ha a feltevés javította az eredmények valószínűségét. z: az eltérés "
                           "a feltevéstől a megadott bizonytalansághoz mérve; |z| > 1,96 = túl "
                           "magabiztos volt.")
                verdicts = {"pending": "még nincs meccs", "consistent": "összhangban",
                            "too optimistic": "túl optimista",
                            "too pessimistic": "túl pesszimista"}
                st.dataframe(pd.DataFrame([{
                    "Csapat": names.team(c.team),
                    "Feltevés": f"{c.assumed:+.0%} ({c.start:%Y-%m-%d})",
                    "Meccs": c.matches,
                    "Gólok (várt)": (f"{c.goals_for}–{c.goals_against} "
                                     f"({c.expected_for:.1f}–{c.expected_against:.1f})"
                                     if c.matches else ""),
                    "Tényleges": (f"{c.realized:+.0%} [{c.ci_low:+.0%}, {c.ci_high:+.0%}]"
                                  if c.realized is not None else ""),
                    "Ítélet": verdicts[c.verdict],
                    "Log-score / meccs": c.log_gain,
                    "Bizonytalanság ×": c.drift_factor,
                    "z": c.z,
                    "Bizonytalanság ítélet": {"pending": "még nincs meccs",
                                              "consistent": "összhangban",
                                              "too confident": "túl magabiztos"}[
                        c.drift_verdict],
                } for c in checks]), hide_index=True, width="stretch",
                    column_config={"Log-score / meccs": st.column_config.NumberColumn(
                        format="%+.3f"), "z": st.column_config.NumberColumn(format="%+.2f")})
                cal = calibration(checks)
                if cal.n >= 2:
                    words = {"calibrated": "jól kalibrált", "overconfident": "túl magabiztos",
                             "too cautious": "túl óvatos"}
                    st.caption(
                        f"Összesítve ({cal.n} feltevés): átlagos z² = {cal.mean_z2:.2f} "
                        f"(1 = jól kalibrált), a ±1,96-os sávon kívül {cal.outside:.0%} "
                        f"(várható 5%) → **{words[cal.verdict]}**"
                        + (f"; az utólag legjobban illeszkedő bizonytalanság-szorzó "
                           f"{cal.pooled_factor:.1f}." if cal.pooled_factor is not None else ".")
                        + (" Kevés feltevés: csak tájékoztató." if cal.n < 10 else ""))
        sim = simulation(competition, season, team_params, n_sims,
                         uncertainty and has_likelihood,
                         DRIFT_DEFAULT if form_drift and has_likelihood else 0.0,
                         assumptions)
        summ = to_pct(names.frame(sim.summary), ["p_title", "p_top3", "p_last"])
        note = ("Az erősségek bizonytalanságát is szimulálja." if sim.uncertainty else
                "Az erősségeket ismertnek veszi, ezért (főleg ősszel) túl magabiztos.")
        if sim.drift:
            note += " A formaváltozással is számol."
        if assumptions:
            starts = sorted({a[2] for a in assumptions if a[2]})
            note += f" Saját feltevések: {len(assumptions)} csapat"
            note += f" ({', '.join(starts)}-tól)." if starts else "."
        st.caption(f"{sim.remaining} hátralévő mérkőzés, {sim.n_sims} lejátszott szezon. {note}")
        st.dataframe(summ[["team", "points_now", "exp_points", "points_p10", "points_p90",
                           "exp_position", "p_title", "p_top3", "p_last"]].rename(columns={
                               "team": "Csapat", "points_now": "Pont most",
                               "exp_points": "Várható pont", "points_p10": "P10",
                               "points_p90": "P90", "exp_position": "Várható hely",
                               "p_title": "Bajnok", "p_top3": "Dobogó",
                               "p_last": "Utolsó"}),
                     hide_index=True, width="stretch", column_config={
                         "Várható pont": st.column_config.NumberColumn(format="%.1f"),
                         "Várható hely": st.column_config.NumberColumn(format="%.1f"),
                         "Bajnok": PCT_BAR, "Dobogó": PCT1, "Utolsó": PCT1})
    pos = sim.positions.copy()
    pos.index = [names.team(t) for t in pos.index]
    fig = go.Figure(go.Heatmap(
        z=pos.to_numpy() * 100, x=[str(c) for c in pos.columns], y=pos.index,
        colorscale=BLUE_RAMP, showscale=False, xgap=2, ygap=2,
        text=np.where(pos.to_numpy() >= 0.005, np.round(pos.to_numpy() * 100).astype(int)
                      .astype(str), ""), texttemplate="%{text}",
        hovertemplate="%{y}: %{x}. hely – %{z:.1f}%<extra></extra>"))
    st.markdown("**Végső helyezés valószínűsége (%)**")
    fig.update_layout(height=34 * len(pos) + 60, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis=dict(title="Helyezés", side="top"),
                      yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


# ---------------------------------------------------------------- scorers tab
with tab_scorers:
    if apps.empty:
        st.info("Ehhez a bajnoksághoz nincs felállás-adat (jegyzőkönyv).")
    else:
        sc = fit_scorers(played, apps, goals, SCORER_DEFAULTS)
        tbl = to_pct(names.frame(sc.players).sort_values(["goals", "share"], ascending=False),
                     ["share"])
        st.subheader("Góllövőlista és gólarány")
        st.caption("Gólarány = a csapat góljainak hány százalékát szerzi, amíg pályán van "
                   f"(zsugorítva a liga átlagához, ami {pct(sc.league_share)}). Várható perc "
                   "= átlag a csapat utolsó 5 meccsén.")
        st.dataframe(
            tbl.head(40)[["player", "team", "apps", "minutes", "goals", "penalties", "share",
                          "exp_minutes"]].rename(columns={
                              "player": "Játékos", "team": "Csapat", "apps": "Meccs",
                              "minutes": "Perc", "goals": "Gól", "penalties": "11-es",
                              "share": "Gólarány", "exp_minutes": "Várható perc"}),
            hide_index=True, width="stretch", height=600, column_config={
                "Perc": st.column_config.NumberColumn(format="%.0f"),
                "Gólarány": PCT1,
                "Várható perc": st.column_config.NumberColumn(format="%.0f")})


# --------------------------------------------------------------- strengths tab
with tab_strength:
    df = strengths.to_frame().reset_index(names="team")
    df["label"] = df["team"].map(names.team)
    st.subheader("Támadás vs. védekezés")
    st.caption("Támadás > 1: többet lő az átlagnál. Védekezés < 1: kevesebbet kap. "
               "Jobb felső = erős csapat (a védekezés tengely fordított).")
    fig = go.Figure(go.Scatter(
        x=df["attack"], y=df["defense"], mode="markers+text", text=df["label"],
        textposition="top center", marker=dict(size=11, color=HOME,
                                               line=dict(color="white", width=2)),
        customdata=df[["matches"]],
        hovertemplate="%{text}<br>támadás %{x:.2f}<br>védekezés %{y:.2f}"
                      "<br>%{customdata[0]} meccs<extra></extra>"))
    fig.add_hline(y=1, line_color=DRAW, line_width=1)
    fig.add_vline(x=1, line_color=DRAW, line_width=1)
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis=dict(title="Támadás (×, log)", type="log"),
                      yaxis=dict(title="Védekezés (×, log)", type="log", autorange="reversed"))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown(f"Liga alap: **{strengths.base:.2f}** gól (vendég) · hazai előny "
                f"**×{strengths.home_advantage:.2f}**"
                + (f" · Dixon–Coles ρ **{strengths.rho:+.3f}**" if team_params.model == "dc"
                   else ""))


# ----------------------------------------------------------------- results tab
with tab_results:
    res = played.sort_values(["round", "date"], ascending=False)
    res = res.assign(home=res["home"].map(names.team), away=res["away"].map(names.team),
                     score=res["home_goals"].astype(int).astype(str) + "–"
                     + res["away_goals"].astype(int).astype(str),
                     date=res["date"].dt.date)
    st.dataframe(res[["round", "date", "home", "score", "away"]].rename(columns={
        "round": "Forduló", "date": "Dátum", "home": "Hazai", "score": "Eredmény",
        "away": "Vendég"}), hide_index=True, width="stretch", height=600)


# ----------------------------------------------------------------- quality tab
with tab_quality:
    st.subheader("Mennyire jók az előrejelzések?")
    st.markdown(
        "Walk-forward backtest: minden fordulót csak az **előtte** lejátszott meccsekből "
        "jósol a modell. A paramétereket a 2025/26-os szezonon hangoltuk, és a 2026/27-esen "
        "értékeltük (ezt a hangolás nem látta). Futtatás: `football-predictor backtest --tune`.")
    scores = sorted(BACKTEST_DIR.glob("scores_2*.csv"))
    calib = sorted(BACKTEST_DIR.glob("calibration_2*.csv"))
    if not scores:
        st.info("Még nincs backtest eredmény (data/backtest/).")
    else:
        sc_tab = pd.read_csv(scores[-1])
        st.markdown(f"**{scores[-1].stem.removeprefix('scores_')}** – kisebb RPS / log loss "
                    "a jobb; az alsó két sor a naiv viszonyítási alap.")
        st.dataframe(to_pct(sc_tab, ["accuracy"]).rename(columns={
            "forecaster": "Modell", "matches": "Meccs", "accuracy": "Találati arány"}),
                     hide_index=True, width="stretch", column_config={
                         "log_loss": st.column_config.NumberColumn(format="%.3f"),
                         "brier": st.column_config.NumberColumn(format="%.3f"),
                         "rps": st.column_config.NumberColumn(format="%.4f"),
                         "Találati arány": PCT1})
    if calib:
        cal = pd.read_csv(calib[-1])
        fig = go.Figure()
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="tökéletes",
                        line=dict(color=DRAW, dash="dot", width=2), hoverinfo="skip")
        fig.add_scatter(x=cal["predicted"], y=cal["observed"], mode="lines+markers",
                        name="modell", line=dict(color=HOME, width=2),
                        marker=dict(size=9, line=dict(color="white", width=2)),
                        customdata=cal[["count"]],
                        hovertemplate="jósolt %{x:.0%} → bekövetkezett %{y:.0%}"
                                      "<br>%{customdata[0]} esemény<extra></extra>")
        st.markdown("**Kalibráció: jósolt vs. bekövetkezett gyakoriság (1X2)**")
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis=dict(title="Jósolt valószínűség", tickformat=".0%",
                                     range=[0, 1]),
                          yaxis=dict(title="Bekövetkezett", tickformat=".0%", range=[0, 1]),
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    scorer_scores = sorted(BACKTEST_DIR.glob("scorer_scores_2*.csv"))
    if scorer_scores:
        st.divider()
        st.markdown("**Gólszerző-modell** – egy sor = egy várható játékos egy meccsen; "
                    "top1 = a csapat legesélyesebb játékosa betalált.")
        ss = pd.read_csv(scorer_scores[-1])
        st.dataframe(to_pct(ss, ["mean_p", "observed", "top1_hit"]).rename(columns={
            "forecaster": "Modell", "player_matches": "Játékos-meccs", "mean_p": "Átl. jósolt",
            "observed": "Betalált", "top1_hit": "Top1 talált"}),
            hide_index=True, width="stretch", column_config={
                "log_loss": st.column_config.NumberColumn(format="%.4f"),
                "brier": st.column_config.NumberColumn(format="%.4f"),
                "Átl. jósolt": PCT1, "Betalált": PCT1, "Top1 talált": PCT1})
