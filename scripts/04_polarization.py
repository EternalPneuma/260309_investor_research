# Research 4: Sentiment Polarization & Conformity Bias (Reversal Strategy)
#
# Hypothesis: Stocks with extreme consensus bullishness (Q5 high sentconformindex)
# subsequently underperform relative to high-disagreement stocks (Q1).

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
import os
import warnings

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "sentiment_with_returns.parquet")

df = pd.read_parquet(DATA_PATH)
df["postdate"] = pd.to_datetime(df["postdate"])
print(f"Loaded {len(df):,} rows")

# ── 1. Daily Cross-Sectional Quintile Sort on sentconformindex ───────────────

work = df.dropna(subset=["sentconformindex", "fwd_ret_5d"]).copy()

def _safe_qcut(x):
    try:
        cut = pd.qcut(x, 5, labels=False, duplicates="drop")
        if cut.isna().all():
            return pd.Series(np.nan, index=x.index)
        n = int(cut.max()) + 1  # actual number of bins formed
        if n < 2:
            return pd.Series(np.nan, index=x.index)
        # Always map lowest bin → 1, highest bin → 5 regardless of how many bins formed
        mapping = {i: int(round(1 + i * 4 / (n - 1))) for i in range(n)}
        return cut.map(mapping)
    except Exception:
        return pd.Series(np.nan, index=x.index)

work["quintile"] = work.groupby("postdate")["sentconformindex"].transform(_safe_qcut)
work["quintile"] = pd.to_numeric(work["quintile"], errors="coerce")

print(f"Quintile distribution:\n{work['quintile'].value_counts().sort_index()}")

# ── 2. Portfolio Returns by Quintile (Multiple Horizons) ────────────────────

horizons = [1, 5, 10, 20]
results = {}

for h in horizons:
    col = f"fwd_ret_{h}d"
    if col not in work.columns:
        continue
    w = work.dropna(subset=[col, "quintile"])
    qret = w.groupby(["postdate", "quintile"])[col].mean().unstack()
    qret["LS"] = qret.get(1, 0) - qret.get(5, 0)
    results[h] = qret.mean()

res_df = pd.DataFrame(results).T
res_df.index.name = "horizon"
res_df.columns.name = "quintile"
print("\nMean forward return by quintile and horizon:")
print(res_df.round(4))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Sentiment Conformity Quintile Portfolio Returns", fontsize=13)

qret_5d = (
    work.dropna(subset=["fwd_ret_5d", "quintile"])
    .groupby(["postdate", "quintile"])["fwd_ret_5d"]
    .mean()
    .unstack()
    .mean()
    .reindex([1, 2, 3, 4, 5])
)
colors = ["green" if v >= 0 else "red" for v in qret_5d]
axes[0].bar(qret_5d.index.astype(str), qret_5d.values, color=colors, alpha=0.8)
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_xlabel("Conformity Quintile (Q1=high disagreement, Q5=high consensus)")
axes[0].set_ylabel("Mean 5-Day Forward Return")
axes[0].set_title("5-Day Returns by Quintile")

ls_returns = [results[h].get("LS", np.nan) for h in horizons]
axes[1].plot(horizons, ls_returns, marker="o", linewidth=2, color="purple")
axes[1].axhline(0, color="black", linestyle="--", linewidth=0.8)
axes[1].set_xlabel("Horizon (days)")
axes[1].set_ylabel("Long Q1 - Short Q5 Return")
axes[1].set_title("Reversal Strategy (Q1-Q5) Across Horizons")
axes[1].set_xticks(horizons)

plt.tight_layout()
out = os.path.join(SCRIPT_DIR, "data", "04_polarization_portfolios.png")
plt.savefig(out, dpi=150)
plt.show()
print(f"Saved {out}")

# ── 3. Statistical Significance of Long-Short Portfolio ─────────────────────

print("\nLong-Short (Q1 - Q5) portfolio significance tests:")
print(f"{'Horizon':>8} {'Mean':>8} {'Std':>8} {'t-stat':>8} {'p-val':>8} {'Sharpe':>8}")
print("-" * 55)

for h in horizons:
    col = f"fwd_ret_{h}d"
    if col not in work.columns:
        continue
    w = work.dropna(subset=[col, "quintile"])
    qret = w.groupby(["postdate", "quintile"])[col].mean().unstack()
    if 1 not in qret.columns or 5 not in qret.columns:
        continue
    ls = (qret[1] - qret[5]).dropna()
    t, p = stats.ttest_1samp(ls, 0)
    sharpe = ls.mean() / ls.std() * np.sqrt(252 / h) if ls.std() > 0 else np.nan
    print(
        f"{h:>7}d {ls.mean():>8.4f} {ls.std():>8.4f} {t:>8.3f} {p:>8.4f} {sharpe:>8.3f}"
    )

# ── 4. Fama-MacBeth Cross-Sectional Regression ──────────────────────────────

fm_df = work.dropna(
    subset=["fwd_ret_5d", "sentconformindex", "ln_mktcap", "past_ret_20d"]
).copy()

for col in ["fwd_ret_5d", "sentconformindex", "past_ret_20d"]:
    lo, hi = fm_df[col].quantile([0.01, 0.99])
    fm_df[col] = fm_df[col].clip(lo, hi)

dates = fm_df["postdate"].unique()
coefs = []
for date in dates:
    day = fm_df[fm_df["postdate"] == date]
    if len(day) < 20:
        continue
    try:
        m = smf.ols(
            "fwd_ret_5d ~ sentconformindex + ln_mktcap + past_ret_20d", data=day
        ).fit()
        coefs.append(m.params)
    except Exception:
        continue

coef_df = pd.DataFrame(coefs)
print(f"\nFM regressions run: {len(coef_df)}")

fm_means = coef_df.mean()
fm_sems = coef_df.std() / np.sqrt(len(coef_df))
fm_t = fm_means / fm_sems
fm_p = 2 * (1 - stats.t.cdf(fm_t.abs(), df=len(coef_df) - 1))

fm_summary = pd.DataFrame(
    {
        "Mean Coef": fm_means,
        "Std": coef_df.std(),
        "t-stat": fm_t,
        "p-val": fm_p,
    }
)
print("\nFama-MacBeth Results (5-day forward return):")
print(fm_summary.round(4))

# ── 5. Cumulative Long-Short Returns Over Time ───────────────────────────────

w5 = work.dropna(subset=["fwd_ret_5d", "quintile"])
qret5 = w5.groupby(["postdate", "quintile"])["fwd_ret_5d"].mean().unstack()
ls5 = (qret5[1] - qret5[5]).dropna()
cum_ls = (1 + ls5).cumprod() - 1

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(cum_ls.index, cum_ls.values * 100, linewidth=1.5, color="purple")
ax.axhline(0, color="black", linewidth=0.8)
ax.fill_between(
    cum_ls.index,
    0,
    cum_ls.values * 100,
    where=cum_ls.values > 0,
    alpha=0.3,
    color="green",
)
ax.fill_between(
    cum_ls.index,
    0,
    cum_ls.values * 100,
    where=cum_ls.values <= 0,
    alpha=0.3,
    color="red",
)
ax.set_title("Cumulative Long Q1 - Short Q5 Portfolio Return (5-day horizon)")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Return (%)")
plt.tight_layout()
out = os.path.join(SCRIPT_DIR, "data", "04_polarization_cum_ls.png")
plt.savefig(out, dpi=150)
plt.show()
print(f"Saved {out}")

ann_ret = ls5.mean() * 252 / 5
ann_vol = ls5.std() * np.sqrt(252 / 5)
print(f"\nAnnualized Return:     {ann_ret:.2%}")
print(f"Annualized Volatility: {ann_vol:.2%}")
print(f"Sharpe Ratio:          {ann_ret / ann_vol:.3f}")
