# Research 3: Overconfidence & Belief Depth
#
# Hypothesis: During sentiment-price divergence periods (bullish > 0.3
# but past 5d return < -3%), avgwords is significantly higher than normal.

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "sentiment_with_returns.parquet")

df = pd.read_parquet(DATA_PATH)
df["postdate"] = pd.to_datetime(df["postdate"])
print(f"Loaded {len(df):,} rows")

# ── Calendar Alignment: roll weekend dates forward to next trading day ───────

df = df.sort_values(["stockcode", "postdate"])

df["is_trading_day"] = df["postdate"].dt.dayofweek < 5

# For each stock, replace weekend postdates with the next Monday (bfill)
df["trade_date"] = df.groupby("stockcode")["postdate"].transform(
    lambda s: s.where(s.dt.dayofweek < 5).bfill()
)
# Drop trailing weekend rows that cannot be filled (end of series)
df = df.dropna(subset=["trade_date"])

# Aggregate weekend data into the target trading day
_agg = {
    col: func
    for col, func in {
        "bullishsentindexa": "mean",
        "past_ret_5d": "last",
        "avgwords": "mean",
        "avgnetcomments": "mean",
        "ln_mktcap": "last",
        "totalposts": "sum",
    }.items()
    if col in df.columns
}

df = (
    df.groupby(["stockcode", "trade_date"], as_index=False)
    .agg(_agg)
    .rename(columns={"trade_date": "postdate"})
)
print(f"After calendar alignment: {len(df):,} rows")

# Drop rows where any key metric is zero (no meaningful activity that day)
_zero_mask_cols = [c for c in ["bullishsentindexa", "avgwords", "avgnetcomments", "totalposts"] if c in df.columns]
df = df[~(df[_zero_mask_cols] == 0).any(axis=1)]
print(f"After dropping zero-value rows: {len(df):,} rows")

# ── 1. Define Sentiment-Price Divergence ────────────────────────────────────

df = df.sort_values(["stockcode", "postdate"])

df["divergence"] = (
    (df["bullishsentindexa"] > 0.3) & (df["past_ret_5d"] < -0.03)
).astype(int)

print(
    f"Divergence events: {df['divergence'].sum():,} ({df['divergence'].mean():.2%} of obs)"
)
print(f"Non-divergence obs: {(df['divergence'] == 0).sum():,}")

# ── 2. Compare avgwords and avgnetcomments ───────────────────────────────────

div = df[df["divergence"] == 1]
normal = df[df["divergence"] == 0]

for col in ["avgwords", "avgnetcomments"]:
    d = div[col].dropna()
    n = normal[col].dropna()
    t, p = stats.ttest_ind(d, n)
    print(f"\n{col}:")
    print(f"  Divergence: mean={d.mean():.3f}, median={d.median():.3f}, n={len(d):,}")
    print(f"  Normal:     mean={n.mean():.3f}, median={n.median():.3f}, n={len(n):,}")
    print(f"  t={t:.3f}, p={p:.4g}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Overconfidence: Belief Depth During Sentiment-Price Divergence", fontsize=13
)

for ax, col, label in [
    (axes[0], "avgwords", "Avg Words per Post"),
    (axes[1], "avgnetcomments", "Avg Net Comments"),
]:
    s_normal = normal[col].dropna().sample(min(50000, len(normal)), random_state=42)
    s_div = div[col].dropna().sample(min(50000, len(div)), random_state=42)
    ax.hist(s_normal, bins=80, alpha=0.6, density=True, label="Normal")
    ax.hist(s_div, bins=80, alpha=0.6, density=True, label="Divergence")
    ax.set_xlabel(label)
    ax.set_ylabel("Density")
    ax.set_title(label)
    ax.legend()

axes[0].set_xlim(0, 2500)
axes[1].set_xlim(0, 25)

plt.tight_layout()
out = os.path.join(SCRIPT_DIR, "data", "03_overconfidence_dist.png")
plt.savefig(out, dpi=150)
plt.show()
print(f"Saved {out}")

# ── 3. Regression: avgwords ~ bullishsentindexa × divergence + controls ──────

reg_df = df[
    ["avgwords", "bullishsentindexa", "divergence", "ln_mktcap", "totalposts"]
].dropna()

lo, hi = reg_df["avgwords"].quantile([0.01, 0.99])
reg_df["avgwords_w"] = reg_df["avgwords"].clip(lo, hi)
reg_df["ln_totalposts"] = np.log1p(reg_df["totalposts"])

model = smf.ols(
    "avgwords_w ~ bullishsentindexa * divergence + ln_mktcap + ln_totalposts",
    data=reg_df,
).fit(cov_type="HC3")

print("\n" + str(model.summary()))

# ── 4. Interaction Plot ──────────────────────────────────────────────────────

reg_df["bull_decile"] = pd.qcut(reg_df["bullishsentindexa"], 10, labels=False, duplicates="drop")
interaction = (
    reg_df.groupby(["bull_decile", "divergence"])["avgwords_w"].mean().unstack()
)

fig, ax = plt.subplots(figsize=(10, 5))
interaction[0].plot(ax=ax, marker="o", label="Normal")
interaction[1].plot(ax=ax, marker="s", label="Divergence", linestyle="--")
ax.set_xlabel("Bullish Sentiment Decile (0=most bearish, 9=most bullish)")
ax.set_ylabel("Mean Avg Words (winsorized)")
ax.set_title("Belief Depth by Sentiment Level x Divergence Status")
ax.legend()
plt.tight_layout()
out = os.path.join(SCRIPT_DIR, "data", "03_overconfidence_interaction.png")
plt.savefig(out, dpi=150)
plt.show()
print(f"Saved {out}")

# ── 5. Time Series: Divergence Frequency ────────────────────────────────────

daily_div_rate = df.groupby("postdate")["divergence"].mean()

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(daily_div_rate.index, daily_div_rate.values, linewidth=0.8, color="steelblue")
ax.fill_between(daily_div_rate.index, 0, daily_div_rate.values, alpha=0.3)
ax.set_title("Daily Fraction of Stocks in Sentiment-Price Divergence")
ax.set_xlabel("Date")
ax.set_ylabel("Fraction")
plt.tight_layout()
out = os.path.join(SCRIPT_DIR, "data", "03_overconfidence_divergence_ts.png")
plt.savefig(out, dpi=150)
plt.show()
print(f"Saved {out}")
