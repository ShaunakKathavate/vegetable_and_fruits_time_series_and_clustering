# Kalimati Commodity Price Intelligence

A price-intelligence system for Nepal's Kalimati Tarkari wholesale market — built to answer two
kinds of questions at once: *"what does the data say?"* (clustering, forecasting, backtesting)
and *"so what should I actually do?"* (Monte Carlo decision-support tools for farmers, retailers,
and consumers). Six analysis notebooks feed a small set of reusable pipelines, two trained
models, and a three-page Streamlit app.

## Dataset

**Kalimati Tarkari dataset** — daily wholesale price reports (Minimum / Maximum / Average, per
unit) from Kalimati Fruits and Vegetable Market, Kathmandu.

- 197,161 price observations across **132 commodities**
- 2013-06-16 to 2021-05-13 (~8 years)
- Coverage is uneven: 91 commodities have the full 8-year history, 41 were only tracked from
  ~2019 onward — this split (`Tier1_full_history` / `Tier2_sparse`) is handled explicitly
  throughout, not glossed over

---

## What's here

```
notebooks/          6 analysis notebooks, run in order (see below)
src/                 reusable pipelines extracted from the notebooks
  data_pipeline.py          raw CSV -> clean -> tier -> volatility/seasonality features
  simple_kmeans.py          pure-NumPy KMeans (see "Why not sklearn's KMeans?" below)
  train_clustering_models.py   fits + saves the two clustering models
  forecasting_pipeline.py      raw CSV -> weekly panel -> neuralforecast-ready format
  train_forecasting_model.py   fits + saves the winning forecast model
  simulation_pipeline.py       the Monte Carlo engine + business-decision functions
models/              saved, ready-to-load artifacts (joblib / neuralforecast checkpoint)
app/                 3-page Streamlit app (Clustering, Forecasting, Business Recommendations)
data/raw/            the source CSV
reports/from notebooks/   cached feature tables, backtest results, and charts each notebook produced
```

---

## The analysis, notebook by notebook

| # | Notebook | What it does |
|---|---|---|
| 01 | `01_data_exploration` | First look at the raw data |
| 02 | `02_eda_seasonality_analysis` | Data-quality/coverage report, volatility ranking, STL decomposition, year-over-year heatmaps |
| 03 | `03_commodity_clustering` | Groups commodities by *behavior*: volatility, seasonality strength, and seasonal-curve shape (DTW) |
| 04 | `04_business_simulations_clusters` | Monte Carlo price simulation engine + 4 decision-support tools |
| 05 | `05_forecasting_backtest` | Backtests 5 forecasting models against 2 baselines, picks a winner honestly |
| 06 | `06_advanced_business_simulations` | 4 more decision-support tools built on the same simulation engine |

---

## 1. Clustering — segmenting commodities by how they *behave*

**Technical approach:** volatility features (coefficient of variation, avg. daily min–max range)
computed for all 132 commodities; seasonal strength/trend strength via STL decomposition for the
91 commodities with enough history; a third view clusters commodities by the DTW distance between
their normalized month-of-year price curves — so items with the same *seasonal shape* land
together even when their volatility differs. All three views are combined into one lookup table.

**Result:** two KMeans clusterings (k=2 each, chosen by silhouette score) — "High Volatility" vs.
"Stable / Low Volatility" (silhouette 0.38), and a volatility+seasonality split — plus 6
DTW-shape clusters labeled by peak season (Spring/Monsoon/Autumn/Winter).

**Why it matters:** cluster membership is the input to the diversification tool in Section 3 below
— a retailer sourcing across multiple *behavioral* clusters (not just multiple SKUs) is
meaningfully less exposed to any single bad month, because commodities in the same cluster tend
to spike and dip together.

---

## 2. Forecasting — backtested, not assumed

**Technical approach:** five candidate model families are compared via **rolling-origin
backtesting** (4 origins × 8-week horizon, refit from scratch at each fold — no single lucky
train/test split deciding the winner):

| Model | Avg. backtest MAPE |
|---|---|
| **TFT (Temporal Fusion Transformer)** ✅ winner | **18.5%** |
| LSTM | 20.3% |
| LightGBM (global, reference only — not in the deployable pool) | ~20.5% |
| Prophet | 24.4% |
| SARIMA | 76.7% (wrecked by one non-converged fold) |
| Moving Average (baseline) | 24.9% |
| Naive Seasonal (baseline) | 30.5% |

TFT wins and clears the strongest baseline by 6.4 points. The winning model is then retrained
**once**, globally across all 132 commodities' weekly history (bigger training budget, explicit
calendar features), and saved to `models/forecast_final_model/`.

**Why it matters:** picking a forecaster by "which one looked good on one chart" is how you end
up deploying an overfit model. Backtesting first, then only spending real training budget on the
model that actually earned it, is what makes the forecast trustworthy enough to put in front of a
procurement decision.

---

## 3. Business recommendations — the Monte Carlo engine

All eight tools below share one engine (`src/simulation_pipeline.py`): STL-decompose a
commodity's price into trend + seasonal + residual, extrapolate trend and seasonal forward, then
**block-bootstrap** the historical residuals thousands of times — producing a distribution of
plausible future prices, not one point estimate. Numbers below are the notebooks' own worked
example on **Ginger** (the highest-volume commodity).

| Tool | Question it answers | Example result |
|---|---|---|
| **Farmer selling window** | Which month should I sell in? | July recommended (risk-adjusted): expected 76.6, typical range 60.6–110.3, CV 0.30 |
| **Consumer buy-timing** | Is today's price actually a good deal? | At 85.0 in May, cheaper than 38% of typical May prices → "roughly average pricing" |
| **Retailer budget risk (VaR/CVaR)** | What's my realistic worst-case procurement cost? | Correlated-basket 95% CVaR 134,538 vs. an independent-simulation model's 133,197 — naive independence understates tail risk |
| **Portfolio diversification** | Does sourcing across clusters actually reduce risk? | Cost CV dropped from 0.021 (1 cluster) to 0.008 (6 clusters) — a 62% relative risk reduction |
| **Shock stress-testing** | What does a deliberate supply shock cost me? | A 30%, 6-week shock adds 11,303 to expected cost and 15,997 to tail-risk exposure (500 units) |
| **Buy-now vs. wait-and-see** | Is flexibility worth anything? | Adaptive buying vs. buying now saves 7,651 in expectation (500 units, 90-day window) |
| **Price-guarantee / insurance pricing** | What premium is fair for a floor-price contract? | A 90%-of-current floor: fair premium 711.78, priced premium 854.14 (20% loading), 38% chance of any payout |
| **Staggered order timing** | Does splitting a purchase over time help? | Cost CV drops from 0.290 (1 order) to 0.126 (6 orders) — 56% risk reduction, mean cost nearly flat |

**Why it matters:** every one of these maps to a real decision a specific person makes — a farmer
picking a selling month, a retailer setting a procurement budget, a cooperative pricing an
insurance contract. None of them present a single confident number; every result carries its own
range and its own stated assumptions (recent-trend continuing, seasonality repeating, no
unprecedented shocks) — which is the right posture for something meant to inform a real decision,
not just look impressive in a demo.

---

## The app

A 3-page Streamlit app (`app/`) puts all of the above in front of a user without touching a
notebook:

- **Clustering** — pick a commodity, see its cluster assignments and where it sits on the
  volatility landscape
- **Forecasting** — pick any of the 132 commodities, see its 12-week TFT forecast plus the
  backtest comparison that justified the model choice
- **Business Recommendations** — the four notebook-04/06 tools above, as interactive tabs with
  adjustable parameters (risk aversion, basket size, purchase quantity, decision horizon)

### Running it

See **[how_to_run_the_app.md](how_to_run_the_app.md)** for exact steps from Anaconda Prompt or a
VS Code terminal. Short version, from the project root with the `vfm_agri_env` environment active:

```bash
python app/run_app.py
```

Then open `http://localhost:8501`. Use `app/run_app.py`, not `streamlit run app/Home.py` directly
— see the note below.

### Reading it

See **[how_to_read_the_app.md](how_to_read_the_app.md)** for a page-by-page guide to every chart
and card in the app — what each one shows, how to read it step by step, and what to actually infer
from it, with real worked examples.

---

## Tech stack

| Purpose | Libraries |
|---|---|
| Data wrangling | pandas, numpy |
| Classical time series | statsmodels (SARIMA, STL), prophet |
| Gradient boosting | lightgbm |
| Deep learning forecasting | neuralforecast, torch, pytorch-lightning (TFT, LSTM) |
| Clustering | scikit-learn (StandardScaler, Pipeline), scipy (hierarchical/DTW clustering) |
| Model persistence | joblib, neuralforecast's own checkpoint format |
| App / visualization | streamlit, plotly |
| Environment | conda (`environment.yml`, env name `vfm_agri_env`) |

### Why not sklearn's KMeans?

The `vfm_agri_env` conda environment on the machine this was built on has a broken native
dependency: `sklearn.cluster.KMeans`'s compiled Cython/OpenMP code — and, separately,
`np.polyfit`/LAPACK and matplotlib's PNG encoder — crash the process outright when invoked off the
main thread (which is exactly how Streamlit runs page scripts). Rather than mask that with retries
or silently produce wrong results, the clustering pipeline uses a small pure-NumPy KMeans
(`src/simple_kmeans.py`), trend extrapolation uses a closed-form OLS calculation instead of
`np.polyfit`, and all app charts use Plotly instead of matplotlib. `app/run_app.py` also works
around an unrelated corrupted-certificate issue in the OS's certificate store that otherwise
crashes Streamlit's import chain. None of this affects results — it's environment plumbing, not
methodology — but it's the reason those specific choices look slightly unconventional if you're
reading the code.

---

## Known limitations

- **Trend/seasonality extrapolation assumes continuity.** The Monte Carlo engine assumes the
  recent trend keeps drifting the way it has and that seasonality repeats as it has historically —
  it will not anticipate a genuine structural break (new policy, climate-driven shift in growing
  calendars) that hasn't shown up yet in the data.
- **No unprecedented-shock awareness**, by construction — that's exactly why the shock
  stress-testing tool exists: to let you *deliberately* model a scenario instead of hoping the
  bootstrap happens to contain something similar.
- **MAPE can mislead** for commodities with near-zero or occasionally very cheap prices; the
  backtest reports MAE/RMSE alongside it for exactly this reason.
- **The forecasting model only sees price and calendar features** — no weather, festival calendar,
  or supply/arrivals data, which are the most obvious next signals to add.
- **Weekly granularity.** Forecasting resamples to weekly; a business decision needing daily
  precision would need the winning model family refit at daily granularity.

---

## Possible extensions

- Weather and festival/event calendars as forecasting features
- Live price feed instead of a static historical CSV, to make the consumer buy-timing indicator
  genuinely real-time
- Prediction intervals on the forecast (not just a point forecast) once a signal like the above
  reduces the model's under-fit tendency on volatile series
- Extending the backtest's candidate pool per-commodity rather than picking one global winner, for
  commodities where the notebook's own results show SARIMA or Prophet outperforming TFT

---

## Author

Shaunak Kathavate

- LinkedIn: https://www.linkedin.com/in/shaunak-kathavate-7322321a4/
- GitHub: https://github.com/ShaunakKathavate
