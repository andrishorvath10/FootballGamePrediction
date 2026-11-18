# Football Predictor

Data collection, storage and Poisson-based prediction (match results, goalscorers,
season simulation) for amateur football leagues - built on the nine county divisions
of Győr-Moson-Sopron (Hungary, MLSZ data) and a fully synthetic demo league.

![Dashboard: next round](docs/img/next_round.png)

**What it does**

* scrapes results, scorers and line-ups politely from the MLSZ data bank into a
  normalised SQL database (SQLite by default, PostgreSQL via one env variable);
* rates teams with a time-weighted, regularised Poisson model (maximum likelihood,
  optional Dixon–Coles) and prices 1X2, exact scores, over/under and BTTS;
* predicts **who will score** (Poisson thinning + negative binomial over-dispersion);
* simulates the rest of the season 10 000 times (title / podium / last-place odds);
* proves it with a **walk-forward backtest**: parameters tuned on 2025/26, evaluated
  once on the held-out 2026/27 season, against naive baselines;
* Streamlit dashboard with a one-click anonymised view.

## Architecture

```mermaid
flowchart LR
    MLSZ[(adatbank.mlsz.hu)] -->|PoliteFetcher<br/>cache, delay, retry| MP[mlsz.py parser]
    CSV[CSV files] --> CI[csv_import.py]
    GEN[sample_data.py<br/>synthetic league] --> R
    MP --> R[MatchRecord<br/>GoalRecord<br/>AppearanceRecord]
    CI --> R
    R --> L[LeagueLoader<br/>idempotent upsert] --> DB[(SQLAlchemy DB)]
    MIG[Alembic migrations] -.->|schema| DB
    DB --> D[dataset.py<br/>pandas frames]
    D --> S[strength.py / mle.py<br/>team ratings]
    S --> P[poisson.py<br/>score matrix, markets]
    S --> SIM[simulate.py<br/>Monte Carlo season]
    D --> SC[scorers.py<br/>goalscorer model]
    S --> SC
    P & SC & SIM --> CLI[cli.py] & APP[Streamlit dashboard]
    S & SC --> EV[evaluate.py<br/>walk-forward backtest]
```

The model code only sees pandas DataFrames, never ORM objects, so it is fast to test
and independent of the storage layer. CLI and dashboard call the same functions, so
they always show the same numbers.

## Quick start

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
cp .env.example .env          # optional; defaults to SQLite in data/football.db
football-predictor init-db
football-predictor generate-sample          # fictional demo league (no scraping needed)
football-predictor predict                  # next round of the demo league
streamlit run app/dashboard.py              # dashboard on http://localhost:8501
pytest                                      # 100+ tests
```

Real data (Győr-Moson-Sopron county divisions, ~15 min per season with match details):

```bash
football-predictor scrape-mlsz                       # current season
football-predictor scrape-mlsz --season 2025/26      # previous season (for tuning)
football-predictor competitions
```

Switch to PostgreSQL without code changes:

```bash
uv pip install -e ".[postgres]"
export FP_DATABASE_URL="postgresql+psycopg://user:pass@localhost:5432/football"
football-predictor init-db
```

### Command overview

| command | what it does |
|---|---|
| `init-db [--drop]` | create the schema / migrate to the latest revision |
| `db current\|upgrade\|downgrade\|revision\|check` | Alembic schema migrations |
| `generate-sample` | load a fictional league (+ ground truth JSON) |
| `import-csv`, `scrape`, `scrape-mlsz` | load real data (idempotent) |
| `validate` | goal rows vs. final scores |
| `competitions`, `table` | what is in the database, current standings (MLSZ tie-breakers, deductions) |
| `predict` | next round: xG, 1X2, scores, O/U, BTTS (`--show-strengths`) |
| `scorers` | anytime-scorer probabilities (`--table`, `--anonymize`) |
| `simulate` | Monte Carlo rest of season (`--positions`, `--from-round`, `--no-uncertainty`, `--drift`, `--team-drift`, `--team-shift`) |
| `backtest-simulation` | freeze finished seasons part-way, score the simulated final tables |
| `tune-drift` | estimate how fast team strengths change during a season |
| `assume add\|list\|remove\|check` | store expert assumptions, check them against later results (`check --all-teams DATE [--all-competitions]`: calibration test of the model itself) |
| `backtest [--tune]` | walk-forward evaluation of team models |
| `backtest-scorers [--tune]` | walk-forward evaluation of the scorer model |

## Data ingestion

```
 sample_data.py ─┐
 csv_import.py  ─┼─> list[MatchRecord] ──> LeagueLoader (idempotent upsert) ──> DB
 scraper.py     ─┘        (records.py)
   └─ PoliteFetcher (robots.txt, delay, retry, cache in data/raw/) -> parse_fixtures()
```

```bash
football-predictor generate-sample --seed 42          # FICTIONAL demo league + data/sample_truth.json
football-predictor import-csv examples/matches.csv --goals examples/goals.csv
football-predictor scrape --config my_league.toml     # see examples/scraper.example.toml
football-predictor scrape --config my_league.toml --offline   # re-parse cached HTML only
football-predictor add-alias "Ködvár" "Ködvár SE"
```

Every command can be run repeatedly: the second run reports `0 created, 0 updated, N unchanged`.

### MLSZ adatbank (Győr-Moson-Sopron, Megye I–III)

`config/mlsz_gyms.toml` lists the nine senior men's county divisions with their MLSZ
league IDs for 2026/27 and 2025/26.

```bash
football-predictor scrape-mlsz                                   # all divisions, current season
football-predictor scrape-mlsz --competition "GYMS Megye II Észak"
football-predictor scrape-mlsz --season 2025/26                  # previous season (backtesting)
football-predictor scrape-mlsz --no-details                      # results only, ~5x faster
football-predictor scrape-mlsz --offline                         # re-parse cached pages only
```

* Round pages give fixtures, results and dates; match pages give scorers (with
  minute, penalty, own goal) and line-ups, from which minutes played are derived
  using substitution and red-card events.
* Players are identified by their stable MLSZ ID (`mlsz:<id>`), so transfers and
  namesakes are handled correctly. Team IDs change every season, so teams are
  matched by name (+ aliases).
* Politeness: no robots.txt exists on the site (404 = no restrictions), we still wait
  3 s between requests, cache every page and only re-download round pages that
  may have received new results.
* **Privacy:** scraped pages and the database contain real names and stay local
  (`data/` is git-ignored). Test fixtures use fictional names with the real HTML
  structure.

> Schema changes are versioned with Alembic (see [Schema migrations](#schema-migrations)):
> after pulling one, `football-predictor db upgrade` migrates the database in place,
> so scraped data does not have to be downloaded again.

## Prediction model (simple version)

Expected goals for home team *h* against away team *a*:

$$\lambda = \text{base}\cdot\text{att}_h\cdot\text{def}_a\cdot\text{home},\qquad
\mu = \text{base}\cdot\text{att}_a\cdot\text{def}_h$$

* `base` = average away goals, `home` = average home goals / average away goals
* `att_i = (Σ w·scored + k·m) / (Σ w·expected_scored + k·m)` where *expected* is what an
  average team scores at that venue; `def_i` likewise with goals conceded
  (`def > 1` = weak defence)
* `w = exp(-ξ·days_ago)` time decay; `k` = shrinkage in pseudo-matches of average
  performance (= posterior mean under a Gamma–Poisson prior), `m` = mean goals per team
* Score matrix `P(i, j) = Pois(i; λ)·Pois(j; μ)` → 1X2, top-5 exact scores,
  over/under 2.5, both teams to score (computed on a grid large enough for λ ≈ 10).

Parameter recovery on synthetic leagues (30 seeds, σ = 0.3, mean correlation of
estimated vs. true log-strength / RMSE):

| data | k = 0 | k = 2 | k = 5 |
|---|---|---|---|
| 10 teams, 7 rounds | fails (log 0) | att 0.72, def 0.62 / 0.229 | 0.72, 0.62 / 0.205 |
| 10 teams, 14 rounds | 0.83, 0.71 / 0.211 | 0.83, 0.71 / 0.181 | 0.83, 0.72 / 0.166 |
| 14 teams, 26 rounds | 0.88, 0.87 / 0.156 | 0.88, 0.87 / 0.143 | 0.88, 0.87 / 0.135 |

```bash
football-predictor competitions
football-predictor predict --competition "GYMS Megye II Észak" --show-strengths
football-predictor predict --competition "GYMS Megye II Észak" --model simple --xi 0.005 --shrinkage 3
```

Known limitation: opponent strength is ignored (a team that has played the weakest
side twice looks too strong). The maximum-likelihood model fixes this.

## Prediction model (advanced): MAP Poisson + Dixon–Coles

$$\log\lambda = \beta + \eta + a_h + d_a,\qquad \log\mu = \beta + a_a + d_h,
\qquad \textstyle\sum a = \sum d = 0$$

All teams are fitted **jointly** by minimising the time-weighted negative log-likelihood
plus a Normal(0, sd²) prior on the log-strengths (L2 penalty = shrinkage), with analytic
gradients and L-BFGS-B (`model/mle.py`; the gradient is checked against finite
differences in the tests). `dc` adds the Dixon–Coles factor τ(x, y; λ, μ, ρ) that
re-weights the 0-0, 1-0, 0-1 and 1-1 cells.

### Evaluation

Walk-forward backtest: for every round the model is trained only on matches played
before that round (no look-ahead), scored with log loss, Brier, **RPS** (ranked
probability score, the standard 1X2 metric) and accuracy, and compared with two
baselines. Hyper-parameters are tuned on **2025/26** (1321 scored matches, 9 GYMS
leagues, grid of 68 settings) and then evaluated once on the held-out **2026/27**
season:

```bash
football-predictor backtest --tune              # tune on 2025/26, evaluate on 2026/27
```

| forecaster (tuned on 2025/26) | log loss | Brier | RPS | accuracy |
|---|---|---|---|---|
| simple (ξ=0.005, k=2) | **0.841** | **0.489** | **0.194** | 63.9 % |
| mle (ξ=0.005, sd=0.7) | 0.850 | 0.490 | 0.194 | **64.6 %** |
| dc (ξ=0.005, sd=0.7) | 0.850 | 0.490 | 0.194 | 64.6 % |
| simple, no decay | 0.846 | 0.492 | 0.195 | 63.2 % |
| simple, last 5 matches | 0.863 | 0.501 | 0.199 | 64.0 % |
| league H/D/A frequencies | 0.978 | 0.599 | 0.249 | 48.1 % |
| uniform 1/3 | 1.099 | 0.667 | 0.259 | 51.2 % |

559 held-out matches (rounds ≥ 4). Findings:

* Every model beats both baselines by a wide margin (RPS 0.194 vs 0.249).
* On 2025/26 the MLE model is best (RPS 0.1575 vs 0.1605 for the simple model); on the
  held-out season the two are within noise of each other.
* Time decay helps (half-life ≈ 139 days); a hard "last 5 matches" window is worse.
* Dixon–Coles brings nothing here: these leagues average 4.5–6 goals per match, so the
  low-score cells it corrects are rare and the fitted ρ is unstable (sign flips between
  seasons, sometimes hits the ±0.25 bound).
* Calibration is good: e.g. events forecast at 15.8 % happened 15.8 % of the time,
  95 % → 92 %.

`predict` defaults to the tuned `mle(ξ=0.005, sd=0.7)`. Data quirk: a few official
results differ from the goals in the match report (decisions taken off the pitch);
`validate` lists them and the model uses the official score.

## Goalscorer model

```bash
football-predictor scorers --competition "GYMS Megye II Észak"            # next round
football-predictor scorers --competition "GYMS Megye II Észak" --table --anonymize
football-predictor backtest-scorers --tune     # tune on 2025/26, evaluate on 2026/27
```

**Poisson thinning.** If a team's goals are Poisson(λ) and each goal is scored by
player *i* independently with probability *q_i*, then player *i*'s goals are
Poisson(λ·q_i). So the team model (λ) and the player model (q_i) are estimated
separately and multiplied:

$$xG_i = \lambda_{team}\cdot \text{share}_i \cdot \frac{\text{expected minutes}_i}{90}$$

* **expected minutes** = average minutes over the team's last 5 matches (0 if absent:
  handles injuries, transfers, being dropped);
* **share** = fraction of the team's goals the player scores *while on the pitch*.
  Exposure `E_i = Σ team_goals · minutes_i / 90`; goals_i ~ Poisson(share_i · E_i) with a
  Gamma prior centred on the league share s0 gives the shrunk estimate
  `share_i = (goals_i + k·s0) / (E_i + k)` (same Gamma–Poisson idea as the team model;
  s0 ≈ 9 %, below 1/11 because own goals and unknown scorers are credited to nobody).

**Over-dispersion → negative binomial.** The first backtest showed that xG was right on
average (6414 expected vs 6336 actual goals) but P(score) was too high for favourites
(forecast 80 % → happened 65 %): goals cluster (a hat-trick in a 9-0 win), so
Var(goals) ≈ 1.4–2.2 × mean instead of = mean. Modelling a player's goals as negative
binomial with mean xG and size *r* (a Gamma mixture of Poissons)

$$P(\text{scores}) = 1 - \left(\frac{r}{r + xG}\right)^{r}$$

fixed the calibration (*r* = 1 tuned).

**Fair evaluation.** Every model is scored on the same rows - every player who has
played for the team so far, before kick-off, with probability 0 if outside the form
window. (An earlier version scored only each model's own candidates, which made long
form windows look better simply because they added many easy "≈0 %" rows.)

Tuned on 2025/26 (64 484 player-matches, grid over k, form window, r), evaluated once on
2026/27 (24 395 player-matches):

| model | log loss | Brier | mean forecast | observed | top-1 hit |
|---|---|---|---|---|---|
| **k=8, form 5, NB r=1** | **0.2376** | **0.0658** | 8.2 % | 8.3 % | 37.4 % |
| same, Poisson | 0.2386 | 0.0662 | 9.0 % | 8.3 % | 37.4 % |
| same, league-average team λ | 0.2439 | 0.0673 | 8.6 % | 8.3 % | 37.4 % |
| minutes only (no player skill) | 0.2615 | 0.0722 | 8.8 % | 8.3 % | 15.6 % |

*top-1 hit* = the team's most likely scorer did score. Player skill is the biggest gain
(top-1 hit 15.6 % → 37.4 %), team strength adds a clear improvement, and the negative
binomial fixes the calibration:

| forecast | 0–5 % | 5–10 % | 10–20 % | 20–30 % | 30–40 % | 40–50 % | 50–70 % |
|---|---|---|---|---|---|---|---|
| mean forecast | 2.1 % | 7.1 % | 13.9 % | 24.2 % | 34.4 % | 44.2 % | 57.3 % |
| happened | 2.6 % | 6.7 % | 13.7 % | 23.9 % | 32.8 % | 46.9 % | 63.3 % |

Limitations: goals are assumed spread evenly over 90 minutes (substitutes' exact
on-pitch window is not used), penalty takers are not modelled separately.

## Season simulation (Monte Carlo)

```bash
football-predictor table --competition "GYMS Megye I"
football-predictor simulate --competition "GYMS Megye I" --positions
football-predictor simulate --competition "GYMS Megye I" --season 2025/26 --from-round 14
football-predictor backtest-simulation --season 2025/26     # score against final tables
football-predictor tune-drift --season 2025/26              # estimate the strength drift
```

1. fit the team model on the matches played so far;
2. draw a **separate set of team strengths for every simulated season** from the
   posterior (below), compute (λ, μ) for every remaining fixture and draw all scores at
   once (numpy arrays of shape `(n_sims, n_fixtures)`; with Dixon–Coles by rejection
   sampling: propose independent Poissons, accept with probability τ(x, y) / max τ);
3. add the simulated points / goal difference with `np.add.at` (a team appears in
   several fixtures), rank every simulated season with `np.lexsort` by the official
   MLSZ rules (below); the few seasons with teams still level after goals scored get
   a head-to-head pass, the rest is decided by lot (random key);
4. count: P(title), P(top 3), P(last), expected points with a 10–90 % band and the full
   position distribution. 10 000 seasons take ~0.4 s.

### Strength uncertainty (Laplace approximation)

Treating the fitted strengths as exact ignores that after 10 matches we do not really
know how good a team is. The MAP fit already gives us what we need: near its maximum
the log posterior is roughly quadratic, so the posterior is roughly Normal,

    θ | data  ≈  N(θ̂_MAP, H⁻¹),   H = Σ_m w_m λ_m x_m x_mᵀ + Σ_m w_m μ_m z_m z_mᵀ + I / prior_sd²

where x_m, z_m are the 0/1 design rows of log λ_m and log μ_m. For a Poisson GLM the
second derivative of −log L with respect to the linear predictor is simply the mean,
so H is cheap and exact (checked against finite differences). Few matches → little
curvature → wide posterior; the prior term keeps H invertible even for a team that has
barely played. The sum-to-zero constraint is handled by working in an orthonormal
basis of the constrained subspace (`scipy.linalg.null_space`), so every draw satisfies
it. Check on synthetic leagues whose true strengths are drawn from the prior: the 90 %
intervals cover the truth 89.8 % of the time.

Each of the 10 000 simulated seasons uses its own θ draw, so the uncertainty about the
teams propagates into the table (mixture of Poissons = over-dispersed points).

### Backtest

`backtest-simulation`: freeze every 2025/26 division part-way, simulate the rest, score
the predicted distribution of each team's final position against where it really
finished (log loss, and RPS over positions - ordinal, so predicting 2nd for the
champion costs less than predicting 9th). "Frozen" means the state just before the
freeze round started: results of postponed games played later are hidden too (an
earlier version leaked them, which flattered every backtest number). 9 divisions,
116 teams per row:

| frozen after | strengths | log loss | RPS | P(actual champion) | favourite won | \|exp. pos − final\| |
|---|---|---|---|---|---|---|
| 1/3 of the season | fixed | 2.219 | 0.0863 | 0.58 | 6 / 9 | 1.38 |
| | uncertain | 1.864 | 0.0819 | 0.52 | 6 / 9 | 1.41 |
| | uncertain + drift | **1.820** | **0.0809** | 0.50 | 6 / 9 | 1.42 |
| 1/2 of the season | fixed | 1.532 | 0.0578 | 0.59 | 5 / 9 | 1.01 |
| | uncertain | **1.448** | **0.0569** | 0.57 | 5 / 9 | 1.02 |
| | uncertain + drift | 1.452 | 0.0573 | 0.54 | 5 / 9 | 1.04 |
| 2/3 of the season | fixed | 1.191 | 0.0473 | 0.68 | 6 / 9 | 0.81 |
| | uncertain | **1.185** | 0.0464 | 0.67 | 6 / 9 | 0.82 |
| | uncertain + drift | 1.195 | **0.0461** | 0.66 | 6 / 9 | 0.83 |

The proper scores improve with parameter uncertainty, most in autumn (log loss −16 %
after a third of the season) and the gain shrinks as the season goes on, exactly as
the posterior narrows. The fixed-strength model's losses come from teams it gave ~0 %
for the place they really finished; uncertainty removes those blunders at the cost of
a little sharpness (the actual champion gets slightly less, the point-estimate MAE is
marginally worse - MAE does not reward honest spread). Small sample (9 leagues), but
consistent across all three freeze points. Uncertainty and drift (next section) are on
by default in the CLI (`--no-uncertainty`, `--drift 0` for the old behaviour) and the
dashboard (toggles); the simple ratio model has no likelihood, so it runs without them.

### Form changes during the season (strength drift)

The strengths describe a team *today*; by May, players have left, coaches changed and
the three-month winter break has reshuffled squads. The simulation therefore lets every
team's log attack and log defence follow a **random walk** until each fixture:

    θ(t) = θ(now) + W(t),   W(t) − W(s) ~ N(0, σ² · (t − s) / 30 days)

* the variance grows with *calendar* time, so the winter break counts as three months,
  not as one round;
* the walk is *persistent* within a simulated season (a team that drops off in March
  stays weaker in April), which is what spreads out the final tables - independent
  per-match noise would mostly average out over 13 spring fixtures;
* increments are centred across teams, so strengths stay relative.

**Estimating σ.** Not from the final tables (9 data points per season), but from
predictions *several rounds ahead*: `tune-drift` freezes each season every 2 rounds,
fits the posterior, and scores the exact-score probability of every later match under
each σ. The predictive distribution is a mixture over posterior draws and drift draws,
`P(x, y) = E[Pois(x; λ(θ + W_d)) · Pois(y; μ(θ + W_d))]`, evaluated by Monte Carlo
with common random numbers across σ. 2025/26, 8 167 (match, horizon) predictions,
log loss of the exact score:

| σ per 30 days | all | ≤ 30 days | 31–90 d | 91–180 d | > 180 d |
|---|---|---|---|---|---|
| 0 (no drift) | 3.8997 | 3.7555 | 3.9025 | 3.8344 | 4.0315 |
| 0.05 | 3.8862 | 3.7542 | 3.8974 | 3.8132 | 4.0067 |
| **0.10** | **3.8677** | 3.7511 | 3.8862 | 3.7841 | **3.9762** |
| 0.15 | 3.8681 | 3.7472 | 3.8768 | **3.7838** | 3.9862 |
| 0.20 | 3.8910 | 3.7440 | 3.8758 | 3.8185 | 4.0391 |
| 0.30 | 3.9821 | 3.7443 | 3.9090 | 3.9558 | 4.2165 |

σ = 0.10 (on the log scale ≈ ±10 % attack/defence per month, ±20 % over a winter) is
the choice: overall it ties with 0.15, and it is best for the long horizons drift is
about. Paired by match (standard error clustered over a match's repeated forecasts),
it improves the 2025/26 predictions by 0.032 ± 0.007 nats per match, and by
0.054 ± 0.012 beyond 90 days. On the 2026/27 autumn (holdout, horizons ≤ 90 days
only) it is neutral (+0.000 ± 0.002), as expected without long horizons. On the
final-table backtest above, drift helps a little in autumn and is neutral later - an
effect too small for 9 leagues to resolve either way.

*A rejected idea:* short horizons preferred a larger σ, which hinted at independent
match-level noise (over-dispersed scores, a log-normal factor ω on λ and μ). Given
the drift (and the lag below) ω = 0.2 adds nothing on 2025/26 (+0.002 ± 0.003), and on
the 2026/27 holdout it is *worse* (−0.007 ± 0.003 nats per match), so it is not used
(`tune-drift --omegas` re-runs the experiment).

**Team-specific drift: tested, not supported by the data.** Some teams surely change
more than others - but can we tell *which ones* in advance? A single extra parameter β
redistributes the drift between teams by a measure taken before the freeze,
σ²_team ∝ exp(2β·z_team) (z standardised within the league, normalised so the average
variance stays σ², so β only moves drift around). Two candidate measures:

* **squad turnover** from the line-ups: 1 − mean overlap (Jaccard) of the players with
  ≥ 45 minutes in consecutive matches. It varies a lot (16–62 %, median 39 %) and makes
  sense - reserve ("II.") teams, whose players come and go with the first team, average
  53 % vs. 38 %;
* **past strength change** ("volatility clustering"): fit the first and second half of
  the pre-freeze matches separately; a team that already moved a lot keeps moving?

Change in log score per match vs. β = 0 (σ = 0.10 with lag, clustered by match, + = better):

| measure | β | 2025/26 all | 2025/26 > 90 days | 2026/27 (holdout) |
|---|---|---|---|---|
| turnover | −0.5 | −0.002 ± 0.002 | −0.003 ± 0.004 | −0.000 ± 0.001 |
| turnover | +0.5 | −0.007 ± 0.003 | −0.013 ± 0.005 | −0.001 ± 0.001 |
| turnover | +1.0 | −0.015 ± 0.004 | −0.028 ± 0.007 | −0.002 ± 0.002 |
| past change | −0.5 | −0.002 ± 0.001 | −0.003 ± 0.002 | −0.000 ± 0.000 |
| past change | +0.5 | −0.006 ± 0.001 | −0.010 ± 0.003 | −0.001 ± 0.000 |
| past change | +1.0 | −0.015 ± 0.003 | −0.029 ± 0.005 | −0.002 ± 0.001 |

Both are best at β = 0, and moving away in either direction gets significantly worse.
A plausible reading: a high-turnover team's estimate already averages over many line-ups,
so its rotation shows up as match-to-match noise, not as a persistent change of level.
Past change measured on half a season is mostly estimation noise. The simulation
therefore uses one σ for all teams. Two things remain available:
`tune-drift --betas ... --covariate turnover|past_change` re-runs the test (e.g. on a
new season), and `simulate --team-drift "TEAM=2"` lets *you* widen one team's drift when
you know something the data cannot (a squad rebuild, a new coach).

**The fit lags behind (extra uncertainty after the winter break).** Checking stored
assumptions (below) showed that from mid-April the model was still overconfident
(mean z² 1.35, p = 0.014) although the drift adds little over six weeks. Splitting the
`tune-drift` predictions by phase located it: a constant extra jump τ per team
(`--taus`) *hurts* before the winter break (and on the 2026/27 holdout, which has no
break yet), but helps a lot for forecasts made *after* the break (+0.019 ± 0.006 nats
per match at τ = 0.15). The reason is the fit, not the future: a time-weighted fit
estimates a weighted *average* of a team's past strengths, θ̂ ≈ Σ w_j θ(t_j) / Σ w_j.
If θ drifts, the lag θ(now) − θ̂ = Σ w_j (θ(now) − θ(t_j)) / Σ w_j has variance

    Var(lag) = σ²/30 · Σ_jk w_j w_k · min(age_j, age_k) / (Σ_j w_j)²

(looking back in time the random walk again has covariance ∝ min). In March the
autumn matches are 4+ months old and still carry most of the weight, so the lag is
large; in October it is small. **No new parameter**: given σ it is fully determined.
The simulation adds it as one persistent offset per team and season (sd σ·lag_sd,
`drift.lag_sd`), whenever drift is on. Change in log score per match against drift
without lag (σ = 0.10, + = better):

| | before the break | across the break | after the break | all | 2026/27 holdout |
|---|---|---|---|---|---|
| lag (theory, no tuning) | +0.001 ± 0.001 | +0.001 ± 0.001 | **+0.017 ± 0.006** | **+0.005 ± 0.002** | −0.001 ± 0.001 |
| best constant jump τ = 0.15 | −0.001 ± 0.003 | +0.002 ± 0.002 | +0.019 ± 0.006 | +0.006 ± 0.003 | −0.005 ± 0.003 |
| extra discount ×0.7 of pre-break matches | 0 | 0 | +0.004 ± 0.004 | +0.001 ± 0.001 | – |

The lag does what the tuned τ does after the break, without hurting before it; the
alternative fix on the fit side (down-weighting pre-break matches) helps less.
`tune-drift --lags 0 1` re-runs it; jointly with the lag, σ = 0.10 stays the best. On
the final tables the lag is within noise (log loss per team +0.009 ± 0.005 at 2/3 of
the season, RPS unchanged). And the motivating check: with the lag, the stored-
assumption calibration is fine at every start date (below).

Also tested analytically (one remaining match: E[points] = 3·P(win) + P(draw) from the
score matrix), for consistency (every position is taken exactly once per simulation;
the vectorised ranking equals `league_table` season by season), and the DC rejection
sampler against the exact score matrix.

### Official table rules: tie-breakers and point deductions

The MLSZ county competition rules (*versenykiírás*, section "A bajnokság helyezéseinek
eldöntése") rank teams level on points by

1. more wins → 2. goal difference → 3. goals scored → 4. points in the matches between
the tied teams → 5. goal difference in those → 6. fair-play ranking → 7. drawing lots.

Note that *wins* come before goal difference. Fair play is not in the data, so 6–7
become alphabetical order in `league_table` and a random draw in the simulation.
Head-to-head is computed as a mini-table of the whole tied group; in the simulation
only the seasons that still have a tie after criterion 3 get this (Python-level)
pass, so it costs little.

**Point deductions** ("büntetőpont", e.g. for fielding an ineligible player) appear
as a note in the official table. `scrape-mlsz` reads them from the freshest cached
standings into the `point_adjustment` table (added by the second Alembic migration);
`table`, `simulate`, `backtest-simulation` and the dashboard apply them. 9 teams
had a deduction in the scraped data (−1 to −9 points).

**Validation against the official tables.** Every round page carries the official
standings after that round, so the 387 cached pages give 387 official tables to check
`league_table` against (matches up to that round + the deduction shown at that time):

* points and matches played agree in all 387;
* the **order is identical in 383**. Under the previous order (goal difference first)
  55 adjacent pairs level on points would have been in the wrong order;
* the 4 differences are all in rounds 1–4, between teams level on points, wins, goal
  difference *and* goals scored: twice the teams had not met (fair play / lot, which
  we cannot know), and twice the website lists them against the head-to-head result.
  Of the 3 in-season cases where head-to-head would decide, the website follows it
  once, so its live table apparently does not apply criterion 4; the competition rules
  explicitly define the *final* order with it, which is what the simulation predicts,
  so we keep it;
* withdrawn teams (no fixtures in our data) are listed last by MLSZ and are ignored.

### Expert assumptions

The data cannot know that a team signed three players in January or lost its coach;
you can. Two per-team knobs, in the CLI and in the dashboard ("Saját feltevések"
editor above the simulation):

```bash
# from 1 March team X scores 1.2x and concedes 1/1.2x the goals the model expects,
# and its form is twice as uncertain as usual
football-predictor simulate --competition "GYMS Megye II Észak" \
    --team-shift "X SE=+20%@2027-03-01" --team-drift "X SE=2"
```

* `--team-shift TEAM=CHANGE[@DATE]` moves the team's strength from that date on
  (log attack += ln(1 + change), log defence −= ln(1 + change)); fixtures before the
  date are untouched, so a winter signing only affects spring;
* `--team-drift TEAM=FACTOR` widens (or narrows) its random walk.

The shift is applied to log λ / log μ of the affected fixtures, so it works with every
model and combines with parameter uncertainty and drift. Tested analytically (one
open fixture: E[points] equals 3·P(win) + P(draw) of the shifted score matrix).

#### Storing and checking assumptions

Assumptions can be stored (third Alembic migration, `assumption` table) - then the
simulation applies them automatically (`--no-assumptions` to ignore) and, once matches
have been played after the start date, they can be **checked against reality**:

```bash
football-predictor assume add "X SE" "+20%@2027-03-01" --note "two strikers signed"
football-predictor assume check          # later in spring
```

1. *Baseline:* fit the team model only on matches **before** the start date and compute
   the expected goals for (a_i) and against (c_i) of the team's later matches - what we
   would have expected without the assumption, untouched by the period being judged.
2. *Realised shift:* in "goals for ~ Pois(a_i·e^δ), goals against ~ Pois(c_i·e^−δ)" the
   log-likelihood is ℓ(δ) = GF·δ − A·e^δ − GA·δ − C·e^−δ (A = Σa_i, C = Σc_i). ℓ'(δ) = 0
   is a quadratic in u = e^δ, so the MLE is closed-form,
   e^δ̂ = [(GF − GA) + √((GF − GA)² + 4AC)] / 2A, and the Fisher information
   A·e^δ̂ + C·e^−δ̂ gives its sampling error (tested against numerical optimisation).
3. *Baseline error:* the a_i, c_i are estimates too. By the implicit function theorem
   ∂δ̂/∂log a_i = −a_i·e^δ̂/I and ∂δ̂/∂log c_i = +c_i·e^−δ̂/I, so the delta method with the
   Laplace covariance Σ of the baseline fit adds gᵀΣg to the variance - plus V_lag, the
   baseline's lag behind the drifting teams (previous section; a constant offset over
   the judged matches, so with the summed weights: σ²·lag_T²·((Σw_a)² + (Σw_c)²) + the
   same for the opponents).
4. *Verdict:* assumed change inside the 95 % interval → "consistent", above → "too
   optimistic", below → "too pessimistic"; plus the change in log score of the actual
   results (> 0 = the assumption helped). The dashboard has a save button and shows the
   same table under the editor.

**Checking the drift factor.** Read as a forecast, an assumption says δ̂ will land near
ln(1 + change) with predictive variance

    V(k) = 1/I  +  gᵀΣg  +  V_lag  +  k²·V_team  +  V_opp

where V_team is what the random walk of the team's own attack and defence adds to δ̂:
σ²/30 · (w_aᵀMw_a + w_cᵀMw_c), with M_ij = min(d_i, d_j) (Brownian covariance, d = days
after the last match the baseline saw) and w the same weights as above; V_opp is the
same for the opponents (factor 1). The residual z = (δ̂ − ln(1 + change)) / √V(k) beyond
±1.96 means **"too confident"**. A single team can never show the opposite - a small
deviation fits any k - so *too cautious* is judged over many assumptions: a calibrated
forecaster has mean z² ≈ 1 (two-sided χ² test), and the pooled maximum-likelihood k
says which factor would have fitted. `assume check` prints both tables and this summary.

Validation:

* a synthetic league with a known +50 % jump: the interval covers 0.5, "+0 %" is
  flagged as too pessimistic and, with little drift allowed, as too confident;
* synthetic leagues whose strengths drift all season (centred, σ = 0.25 or 0.15),
  checked from halfway with the true σ: mean z² 1.15 and 0.99, pooled k 1.19 and 1.00,
  "calibrated" - this needs V_lag, without it σ = 0.25 came out overconfident
  (z² ≈ 1.6); on leagues *without* drift a stated σ = 0.25 is "too cautious" (k ≈ 0),
  and σ = 0 gives mean z² 0.96 - sampling and baseline error alone are accounted for;
* **real data, "no change" for all 116 teams of 2025/26** (σ = 0.10, k = 1;
  `assume check --all-teams 2026-03-01 --all-competitions --season 2025/26` - stores
  nothing, prints one row per competition and the pooled total):

  | start date | mean z² | outside ±1.96 | verdict | pooled k | same without any drift |
  |---|---|---|---|---|---|
  | 1 October | 1.09 | 7 % | calibrated (p = 0.50) | 1.17 | z² 1.67, overconfident |
  | 1 March | 1.01 | 3 % | calibrated (p = 0.88) | 1.06 | z² 2.16, overconfident |
  | 15 April | 1.08 | 3 % | calibrated (p = 0.50) | 1.7 (6 weeks: noisy) | z² 1.51, overconfident |

  On 2026/27, which none of the tuning saw ("no change from 1 October", 113 teams):
  mean z² 0.98, 4 % outside, p = 0.89, calibrated.

  The drift model's uncertainty is about right - an independent check of the σ = 0.10
  that `tune-drift` chose by a different criterion. Before the lag term was added, the
  15 April row was z² 1.35 (p = 0.014): that finding is what led to it.

Corrections to an earlier figure: the direction check first flagged 37 % of the teams
for "no change from 1 March"; counting the baseline's estimation error it is 20 %, and
counting its lag as well **15 %** - still three times the 5 % expected without real
changes. The implied factor of a single assumption is very noisy (a few matches); only
the pooled one is worth reading.

Not modelled: deductions handed out *after* the freeze in the backtest (the final
ones are applied throughout). Assumptions are applied at face value in the simulation;
`assume check` judges them only afterwards.

## Dashboard

```bash
streamlit run app/dashboard.py
# shareable view of one league:
#   http://localhost:8501/?season=2026/27&competition=GYMS%20Megye%20II%20%C3%89szak
FP_ANONYMIZE=true streamlit run app/dashboard.py   # start with names hidden
```

Sidebar: season, division, anonymisation toggle, model settings (team model, time
decay, prior / shrinkage). Tabs:

| tab | content |
|---|---|
| Következő forduló | next round incl. postponed matches: xG, 1X2, most likely score, O2.5, BTTS; per match a score-matrix heat map, top scores and likely scorers |
| Tabella és szimuláció | current table, Monte Carlo summary, final-position heat map |
| Gólszerzők | scorers with minutes, goal share while on the pitch, expected minutes |
| Csapaterősségek | attack vs. defence scatter (log scales) |
| Eredmények | all results |
| Modell minősége | backtest tables and calibration curve (read from `data/backtest/`) |

| | |
|---|---|
| ![table](docs/img/table.png) | ![strengths](docs/img/strengths.png) |

Screenshots show the **synthetic** demo league (fictional clubs and players).
A smoke test (`tests/test_dashboard.py`) runs the app headless on a temporary
synthetic database.

## Data model

```mermaid
erDiagram
    SEASON ||--o{ MATCH : contains
    COMPETITION ||--o{ MATCH : contains
    TEAM ||--o{ MATCH : "home / away"
    TEAM ||--o{ TEAM_ALIAS : "known as"
    MATCH ||--o{ GOAL : has
    TEAM ||--o{ GOAL : "credited to"
    PLAYER |o--o{ GOAL : scores
    PLAYER ||--o{ PLAYER_TEAM : "member of"
    TEAM ||--o{ PLAYER_TEAM : has
    SEASON ||--o{ PLAYER_TEAM : during
    MATCH ||--o{ APPEARANCE : lineup
    PLAYER ||--o{ APPEARANCE : plays
    TEAM ||--o{ POINT_ADJUSTMENT : "deducted"
    SEASON ||--o{ POINT_ADJUSTMENT : in
    COMPETITION ||--o{ POINT_ADJUSTMENT : in
    TEAM ||--o{ ASSUMPTION : "assumed about"
```

### Schema migrations

The ORM models (`db/models.py`) are the source of truth; Alembic revisions in
`src/football_predictor/db/migrations/versions/` record how the schema got there.
Migrations ship inside the package and the CLI configures Alembic in code, so no
`alembic.ini` is needed (one exists for using the plain `alembic` command).

```bash
football-predictor db current                       # revision of the DB vs. head
football-predictor db upgrade                       # apply pending migrations (init-db does too)
football-predictor db revision -m "add referee"     # autogenerate from model changes, then review
football-predictor db downgrade -1                  # revert the last one
football-predictor db check                         # live schema == models? (exit 1 if not)
```

* **SQLite batch mode:** SQLite cannot `ALTER` constraints or column types, so `env.py`
  turns on `render_as_batch`: Alembic copies the table into a new one with the changed
  definition. A fixed `naming_convention` on the metadata gives every constraint a
  deterministic name, which batch mode and PostgreSQL both need to drop it later.
* **Pre-Alembic databases** (built with `create_all`) have the tables but no
  `alembic_version`. `init-db` / `db upgrade` compare their schema with the models and,
  only if they are identical, *stamp* them at head; otherwise they stop with an error
  instead of guessing.
* **Tests** check that `upgrade head` produces exactly the models' schema (so a model
  change without a migration fails CI), that `downgrade base` works, and that legacy
  databases are stamped with their data intact.

## Privacy

Real amateur players' names are personal data. Only synthetic or anonymised data
belongs in the public repository; databases, `data/raw/` and `data/real/` are
git-ignored.
