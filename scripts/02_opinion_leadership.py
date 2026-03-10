# Research 2: Opinion Leadership & Information Hierarchy
#
# Hypothesis: KOL bearish signals precede ordinary retail sentiment
# turning points by 3-7 days.

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "sentiment_with_returns.parquet")

df = pd.read_parquet(DATA_PATH)
df["postdate"] = pd.to_datetime(df["postdate"])
print(f"Loaded {len(df):,} rows")

# ── 1. KOL Influence Coefficient ────────────────────────────────────────────

df["kol_coeff"] = df["avgposterfans"] / (
    df["avgposterfans"] + df["avgposterattention"].fillna(0) + 1
)
print(df["kol_coeff"].describe())

# ── 2. Mark KOL-Dominated Periods (Top 25% cross-sectionally) ───────────────

daily_q75 = df.groupby("postdate")["kol_coeff"].transform(lambda x: x.quantile(0.75))
df["kol_dominated"] = df["kol_coeff"] >= daily_q75
print(f"KOL dominated: {df['kol_dominated'].mean():.1%} of obs")

# ── 3. Compare Sentiment Extremity: KOL vs Non-KOL ──────────────────────────

df["bullish_abs"] = df["bullishsentindexa"].abs()

kol = df[df["kol_dominated"]]["bullish_abs"].dropna()
non_kol = df[~df["kol_dominated"]]["bullish_abs"].dropna()

t_stat, p_val = stats.ttest_ind(kol, non_kol)
print("\n|bullishsentindexa| — KOL dominated vs non-dominated:")
print(
    f"  KOL dominated:  mean={kol.mean():.4f}, median={kol.median():.4f}, n={len(kol):,}"
)
print(
    f"  Non-dominated:  mean={non_kol.mean():.4f}, median={non_kol.median():.4f}, n={len(non_kol):,}"
)
print(f"  t={t_stat:.3f}, p={p_val:.4g}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Opinion Leadership: KOL vs Non-KOL Periods", fontsize=13)

axes[0].hist(non_kol, bins=80, alpha=0.6, label="Non-KOL", density=True)
axes[0].hist(kol, bins=80, alpha=0.6, label="KOL dominated", density=True)
axes[0].set_xlabel("|Bullish Sentiment Index|")
axes[0].set_ylabel("Density")
axes[0].set_title("Sentiment Extremity Distribution")
axes[0].legend()

box_data = [
    non_kol.sample(min(50000, len(non_kol)), random_state=42),
    kol.sample(min(50000, len(kol)), random_state=42),
]
axes[1].boxplot(box_data, labels=["Non-KOL", "KOL dominated"], showfliers=False)
axes[1].set_title("Boxplot (outliers hidden)")
axes[1].set_ylabel("|Bullish Sentiment Index|")

plt.tight_layout()
out = os.path.join(SCRIPT_DIR, "data", "02_opinion_sentiment_dist.png")
plt.savefig(out, dpi=150)
plt.show()
print(f"Saved {out}")

# ── 4. Lag Analysis: KOL Reversal vs Retail Sentiment ───────────────────────

df = df.sort_values(["stockcode", "postdate"])
kol_df = df[df["kol_dominated"]].copy()
retail_df = df[~df["kol_dominated"]].copy()

kol_df["prev_bull"] = kol_df.groupby("stockcode")["bullishsentindexa"].shift(1)
kol_reversal = kol_df[(kol_df["prev_bull"] > 0) & (kol_df["bullishsentindexa"] < 0)][
    ["stockcode", "postdate"]
].copy()
kol_reversal.columns = ["stockcode", "kol_reversal_date"]
print(f"\nKOL reversal events: {len(kol_reversal):,}")

MAX_LAG = 15
lag_means = []
for lag in range(0, MAX_LAG + 1):
    tmp = kol_reversal.copy()
    tmp["target_date"] = tmp["kol_reversal_date"] + pd.Timedelta(days=lag)
    merged = tmp.merge(
        retail_df[["stockcode", "postdate", "bullishsentindexa"]],
        left_on=["stockcode", "target_date"],
        right_on=["stockcode", "postdate"],
        how="inner",
    )
    lag_means.append(
        {
            "lag": lag,
            "retail_bull": merged["bullishsentindexa"].mean(),
            "n": len(merged),
        }
    )

lag_df = pd.DataFrame(lag_means)
print(lag_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(lag_df["lag"], lag_df["retail_bull"], color="steelblue", alpha=0.8)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Days after KOL Bearish Reversal")
ax.set_ylabel("Mean Retail Bullish Sentiment Index")
ax.set_title("Retail Sentiment Following KOL Bearish Reversal")
ax.set_xticks(lag_df["lag"])
plt.tight_layout()
out = os.path.join(SCRIPT_DIR, "data", "02_opinion_lag_analysis.png")
plt.savefig(out, dpi=150)
plt.show()
print(f"Saved {out}")

zero_crossings = lag_df[lag_df["retail_bull"] < 0]
if not zero_crossings.empty:
    print(
        f"Retail sentiment turns negative at lag: {zero_crossings['lag'].iloc[0]} days after KOL reversal"
    )
else:
    print("Retail sentiment does not turn negative within the window")

# ── 5. Cross-Correlation: KOL vs Retail Sentiment ───────────────────────────

daily_kol = kol_df.groupby("postdate")["bullishsentindexa"].mean().rename("kol_bull")
daily_ret = (
    retail_df.groupby("postdate")["bullishsentindexa"].mean().rename("retail_bull")
)
daily = pd.concat([daily_kol, daily_ret], axis=1).dropna()

max_lag = 10
xcorr = [
    daily["kol_bull"].corr(daily["retail_bull"].shift(lag))
    for lag in range(-max_lag, max_lag + 1)
]
lags = list(range(-max_lag, max_lag + 1))

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(lags, xcorr, alpha=0.8, color="steelblue")
ax.axvline(0, color="red", linestyle="--")
ax.set_xlabel("Lag (positive = KOL leads retail)")
ax.set_ylabel("Cross-Correlation")
ax.set_title("Cross-Correlation: KOL Sentiment vs Retail Sentiment")
plt.tight_layout()
out = os.path.join(SCRIPT_DIR, "data", "02_opinion_xcorr.png")
plt.savefig(out, dpi=150)
plt.show()
print(f"Saved {out}")

peak_lag = lags[int(np.argmax(xcorr))]
print(f"Peak cross-correlation at lag={peak_lag}: {max(xcorr):.4f}")
print("Positive lag means KOL leads retail")
