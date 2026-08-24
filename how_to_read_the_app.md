# How to Read the App

A page-by-page, chart-by-chart guide to what you're looking at in the Streamlit app, how to read
it, and what to actually infer from it. Every example below uses real numbers pulled from the
app/data, not made-up placeholders — so if you open the app and select the same commodity, you
should see the same thing.

If you haven't launched the app yet, see [how_to_run_the_app.md](how_to_run_the_app.md) first.

---

## Home page

### The four dataset cards (Commodities / Price observations / Date range start / Date range end)

**What it is:** a summary of the raw dataset everything downstream is built on.

**How to read it:** these four numbers just orient you — how many distinct commodities exist,
how many individual price reports there are in total, and the span of history covered.

**Example:** `132 commodities`, `197,161 price observations`, `2013-06-16` to `2021-05-13`.
**Infer:** roughly 8 years of daily wholesale price reports across 132 commodities — enough
history for yearly seasonal patterns to show up reliably for most (but not all — see the
Clustering page) commodities.

### The three forecasting model cards (Winning model / Backtest MAPE / Beats baseline)

**What it is:** a one-glance summary of which forecasting model the app actually uses, and whether
it's proven to be better than doing nothing clever.

**How to read it:**
1. **Winning model** — which model family was selected after backtesting (see the Forecasting
   page for the full comparison).
2. **Backtest MAPE** — that model's average forecast error (Mean Absolute Percentage Error) across
   backtested commodities. Lower is better; treat it as "on average, the forecast is off by about
   this percentage."
3. **Beats baseline** — whether the winning model actually beats the strongest *naive* baseline
   (e.g., "just use last year's price" or "just use the recent average"). This is the most
   important of the three cards: a fancy model that *doesn't* clear this bar isn't worth deploying.

**Example:** Winning model `TFT`, Backtest MAPE `18.5%`, Beats baseline (Moving Average) `Yes`.
**Infer:** the deployed model isn't just "the most sophisticated one available" — it's the one
that actually earned its place by beating a naive guess, and by how much (24.9% → 18.5% is a real,
non-trivial improvement).

---

## Page 1: Clustering

### The three cluster cards (Volatility cluster / Volatility+seasonality cluster / Seasonal shape cluster)

**What it is:** the selected commodity's assignment in three different clusterings, each built
from a different slice of its price behavior.

**How to read each one:**
- **Volatility cluster** — "High Volatility" or "Stable / Low Volatility," based purely on how
  much the price swings day to day. Computed for *every* commodity, including short-history ones.
- **Volatility + seasonality cluster** — a richer split that also factors in how strongly the
  price follows a repeating yearly pattern. Only available for the 91 commodities with enough
  history (shows "n/a (short history)" otherwise).
- **Seasonal shape cluster** — which month the commodity's price *peaks* in, on average (e.g.
  "Autumn Peak (Month 10)"). This tells you *when* it behaves unusually, not *how much*.

**Example — Tomato Big(Nepali)** (the app's default selection):
- Volatility cluster: `High Volatility`
- Volatility + seasonality cluster: `High-Volatility, Strongly Seasonal`
- Seasonal shape cluster: `Autumn Peak (Month 10)`

**Infer:** tomato prices swing a lot, that swing follows a real yearly rhythm (not just noise),
and the rhythm peaks around October. Practically: a retailer stocking tomatoes should expect
October to be the expensive month and should not treat month-to-month swings as random — they're
predictable enough to plan around.

### The four stat cards (Coefficient of variation / Avg daily min-max range / Seasonal strength / Mean price)

**What it is:** the raw numbers behind the cluster labels above.

**How to read it:**
- **Coefficient of variation (CV)** — standard deviation of price ÷ mean price. `0.36` means the
  typical day-to-day swing is about 36% of the average price. Above ~0.30 is generally "High
  Volatility" territory in this dataset.
- **Avg daily min-max range** — how wide the *reported* min-to-max range is within a single day's
  report, as a fraction of that day's average price.
- **Seasonal strength** — 0 to 1, how much of the price variation is explained by a repeating
  yearly pattern vs. everything else (trend + noise). Closer to 1 = very seasonal; closer to 0 =
  the calendar barely matters for this commodity.
- **Mean price** — the average price over the whole history, for scale.

**Example — Tomato Big(Nepali):** CV `0.36`, avg daily range `13.2%`, seasonal strength `0.22`,
mean price `48.7`.
**Infer:** seasonal strength of `0.22` is moderate, not extreme — tomato's October peak is real
but the day-to-day noise (captured in that 0.36 CV) is actually the bigger driver of its price on
any given day. Compare this to a commodity with seasonal strength above 0.7 (several exist in this
dataset) — for those, the calendar is doing most of the work, and a farmer's selling-month
decision (see the Business Recommendations page) matters a lot more.

### The volatility landscape scatter plot

**What it is:** every commodity plotted by CV (x-axis) against avg daily min-max range (y-axis),
colored by volatility cluster, with your selected commodity marked as a black star.

**How to read it:**
1. Points further right = more volatile on average (bigger day-to-day price swings).
2. Points higher up = wider min-to-max spread *within* a single day's report.
3. The color split shows where the model actually drew the line between "High Volatility" and
   "Stable" — it's a real decision boundary from the fitted KMeans model, not a fixed threshold.
4. Hover over any point to see which commodity it is.

**Infer:** use this to sanity-check a specific commodity against its peers, or to spot commodities
that are unusual outliers even within their own cluster (e.g., far to the right of the rest of
its color group).

### The "All commodities" table

**What it is:** the same cluster labels and stats as above, for all 132 commodities, sortable.

**How to read it:** sorted by CV descending by default — scroll to see the most volatile
commodities first. Use this when you need to compare several commodities at once rather than one
at a time (e.g., picking a diversified basket manually).

---

## Page 2: Forecasting

### The backtest comparison table (inside the expander)

**What it is:** the same model comparison summarized on the Home page, shown in full — every
candidate model's average error, plus the two baselines, sorted best to worst.

**How to read it:** lower `avg_mape_pct` is better. The winner should be at the top and clearly
ahead of both baseline rows. If a model is *below* a baseline in this table, that model lost the
backtest and isn't the one deployed — this table is the actual evidence, not just a label.

**Example:** TFT `18.5%` < LSTM `20.3%` < Prophet `24.4%` < SARIMA `76.7%`; baselines Moving
Average `24.9%`, Naive Seasonal `30.5%`. TFT is the only candidate that clearly beats both
baselines by a wide margin.
**Infer:** SARIMA's 76.7% looks disqualifying, but that's mostly one bad fold dragging the average
up (a non-converged fit produced an explosive forecast on one volatile commodity) — a reminder to
read backtest numbers as "which model to trust on average," not as a verdict on every single
commodity.

### The forecast chart

**What it is:** the selected commodity's last 52 weeks of actual prices (black line) followed by
the model's 12-week forward forecast (blue line with markers), with a dashed vertical line marking
where history ends and forecast begins.

**How to read it:**
1. Everything left of the dashed line is real, observed data.
2. Everything right of it is the model's prediction — it has never seen these values.
3. A forecast that just flatlines near the last historical value, especially if the historical
   line was volatile, is a warning sign of an under-fit model rather than a genuine "prices will
   hold steady" prediction (this is explicitly checked for in the training notebook — see
   `notebooks/05_forecasting_backtest.ipynb`, section 11).
4. A sharp jump right at the dashed line is not automatically wrong — the model is estimating the
   expected *future level* from the whole history, not just extending the last data point.

**Example — Apple(Jholey):** forecast starts around `259` in late May 2021, dips toward `228` by
early July, then recovers to about `244` by early August.
**Infer:** this reads as a real seasonal dip-then-recover shape, not a flat line — consistent with
the model having learned an actual pattern rather than defaulting to "just predict the last known
price."

### The forecast values table

**What it is:** the exact numbers behind the chart's blue line, one row per week.

**How to read it:** use this when you need the actual number for a specific future week (e.g. for
a procurement plan), rather than reading it off the chart by eye.

---

## Page 3: Business Recommendations

Every tab here runs the same Monte Carlo engine (STL-decompose the price into trend + seasonal +
residual, extrapolate trend/seasonal forward, block-bootstrap the residuals thousands of times) —
so every result you see is a distribution, summarized into a recommendation, not a single
confident guess.

### Tab 1 — Farmer: best month to sell

**What it is:** a risk-adjusted ranking of future months by expected price.

**How to read the recommendation banner:** it names one month, its expected price, and its
**typical range (P10–P90)** — the middle 80% of simulated outcomes for that month. A tight range
means the recommendation is close to a sure thing; a wide range means treat it as a good bet, not
a guarantee.

**How to read the bar chart:** each bar is a month's expected price; the error bars are that
month's P10–P90 range; the green bar is the recommended month. **The recommended month is not
always the tallest bar** — a slightly shorter bar with much smaller error bars can out-rank a
taller, wobblier one, because the score is `expected_price − risk_aversion × std_price`, not raw
price alone.

**Example — Ginger:** recommended month **July**, expected price `76.6`, typical range
`60.6–110.3`, CV `0.30`.
**Infer:** July isn't necessarily July's *highest* average price — it's the month with the best
trade-off between price and predictability. If you raise the "Risk aversion" slider, watch the
recommendation potentially shift toward an even safer (lower-CV) month at the cost of some
expected price — that trade-off is the whole point of the tool.

### Tab 2 — Consumer: is today a good deal?

**What it is:** where a specific price falls within the distribution of "normal" prices for that
same calendar month.

**How to read the verdict banner:** three possible verdicts — "Good time to buy" (cheaper than
≥70% of typical prices for this month), "Expensive right now" (cheaper than ≤30%), or "Roughly
average pricing" in between.

**How to read the three metric cards:** *Typical median* is the simulated normal price for this
month; *typical range* is the P10–P90 band; *cheaper than* is the headline number — "cheaper than
X% of typical prices" is the plain-English version of a percentile.

**How to read the histogram:** the blue distribution is every simulated price for this commodity
in this calendar month; the red vertical line is the price being evaluated. The further left the
red line sits relative to the bulk of the blue distribution, the better the deal.

**Example — Ginger at 85.0 in May:** typical median `76.5`, typical range `63.1–120.3`, cheaper
than `38%` → verdict **"Roughly average pricing."**
**Infer:** 85.0 is actually slightly *above* the typical median (76.5) for May, but well within
the normal range — not a bad deal, just not a notable one either. Compare this to how the same
tool judged Raddish White(Local) around the same time (cheaper than 96% of typical prices → "Good
time to buy") — the verdict is genuinely commodity- and month-specific, not a generic rule.

### Tab 3 — Retailer: portfolio diversification

**What it is:** whether sourcing the same size basket from more distinct volatility clusters
actually lowers cost risk (this one requires clicking "Run diversification sweep" — it runs
several simulations, so it's opt-in rather than automatic).

**How to read the result banner:** the headline is a **relative risk reduction percentage** —
how much lower the basket's cost CV (coefficient of variation — risk per unit of expected cost)
is when diversified across the maximum number of clusters, compared to concentrating in just one.

**How to read the line chart:** x-axis is how many distinct clusters the basket is spread across;
y-axis is cost CV. **A downward-sloping curve that flattens out** is the expected pattern — each
additional cluster helps less than the last (diminishing returns). A flat or noisy curve usually
means the basket size is too small relative to the number of available clusters to see a clean
effect.

**How to read the concentrated vs. diversified comparison:** two basket examples with their actual
commodity lists and CVs, so you can see *which* commodities the tool picked, not just the summary
statistic.

**Example (from the notebook's own run):** basket cost CV dropped from `0.021` (1 cluster) to
`0.008` (spread across all available clusters) — a **62% relative risk reduction**.
**Infer:** this is risk reduction, not cost reduction — the point isn't that diversifying makes
the basket cheaper on average, it's that it makes the cost *more predictable*, which is what
actually matters for setting a procurement budget with confidence.

### Tab 4 — Buyer: buy now vs. wait

**What it is:** a comparison of three purchase strategies for acquiring a fixed quantity within a
decision window — buy immediately, wait until one fixed future date, or buy adaptively (as soon as
price drops to a threshold, or at the deadline if it never does).

**How to read the success banner + caption:** the banner names the strategy with the lowest
expected cost; the caption breaks the savings into two pieces — the *value of waiting* (fixed date
vs. buying now) and the *additional value of being adaptive* (adaptive vs. that same fixed date).
Adding both gives the *total value of flexibility*.

**How to read the bar chart:** three bars, one per strategy, mean cost with P10–P90 error bars.
**The real case for the adaptive strategy usually shows up in the error bars being tighter, not
just the bar being shorter** — it avoids the worst outcomes by not being locked into one arbitrary
date, which a mean-cost number alone can hide.

**Example — Ginger, 500 units, 90-day window:** value of waiting `4,238`, additional value of
adaptive buying `3,413`, total value of flexibility `7,651`.
**Infer:** if the "value of waiting" number were *negative* instead, that would mean prices are
expected to rise over the window — in that case buying now could beat waiting even before
factoring in risk at all. Always check the sign, not just the magnitude, before trusting the
recommendation.

---

## A few reading habits that apply everywhere in this app

- **A range is the actual answer; a single number is a summary of it.** Every tool here reports a
  P10–P90 (or CV) alongside its headline number for a reason — treat the headline as "most likely,"
  not "guaranteed."
- **"Beats the baseline" matters more than the model name.** A sophisticated-sounding model that
  doesn't clear a naive baseline in backtesting isn't actually earning its complexity.
- **Risk-reduction tools (diversification, staggered ordering) are about predictability, not
  cheapness.** Mean cost usually stays roughly flat; what changes is how confident you can be in
  that number.
- **Every recommendation assumes "more of the same."** None of these tools know about a shock that
  has no precedent in the historical data — that's what the shock stress-testing logic
  (`notebooks/06_advanced_business_simulations.ipynb`) exists to let you model deliberately instead
  of assuming away.
