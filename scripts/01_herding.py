# Research 1: Herding Behavior & Attention-Driven Trading
#
# Hypothesis: After an attention burst (z-score of totalposts > 2),
# individual stock excess returns are significantly negative in the
# +5 to +20 day window (mean reversion / late retail inflow).

import os
import warnings
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

# ── 0. Seasonality Adjustment: align weekend data to next trading day ────────

df = df.sort_values(["stockcode", "postdate"])

_is_trading_day = df["postdate"].dt.dayofweek < 5
df["trade_date"] = df["postdate"].where(_is_trading_day)  # NaT on weekends
df["trade_date"] = df.groupby("stockcode")["trade_date"].transform(lambda x: x.bfill())
# Drop rows where no next trading day exists (trailing weekends)
df = df.dropna(subset=["trade_date"])

df = (
    df.groupby(["stockcode", "trade_date"], sort=False)
    .agg(
        totalposts=("totalposts", "sum"),
        totalusers=("totalusers", "sum"),
        avgrepostnum=("avgrepostnum", "mean"),
        avgthumbups=("avgthumbups", "mean"),
        ret=("ret", "last"),
    )
    .reset_index()
    .rename(columns={"trade_date": "postdate"})
)

print(f"After weekend alignment: {len(df):,} rows")

# ── 1. Rolling Z-Score of totalposts (90-day window) ────────────────────────

df = df.sort_values(["stockcode", "postdate"])

df["posts_zscore"] = df.groupby("stockcode")["totalposts"].transform(
    lambda x: (
        (x - x.rolling(90, min_periods=30).mean()) / x.rolling(90, min_periods=30).std()
    )
)
df["attention_burst"] = df["posts_zscore"] > 2

print(
    f"Attention burst events: {df['attention_burst'].sum():,} ({df['attention_burst'].mean():.2%} of obs)"
)

# ── 2. Vectorized Event Study [-10, +20] ────────────────────────────────────

METRICS = ["avgrepostnum", "avgthumbups", "totalposts", "totalusers", "ret"]


def build_event_panel(df, event_mask, metrics, window=(-10, 20)):
    df = df.sort_values(["stockcode", "postdate"]).copy()
    df["within_stock_idx"] = df.groupby("stockcode").cumcount()

    event_rows = df[event_mask][["stockcode", "postdate", "within_stock_idx"]].copy()

    all_recs = []
    for rel in range(window[0], window[1] + 1):
        shifted = event_rows.copy()
        shifted["target_within_idx"] = shifted["within_stock_idx"] + rel
        shifted["rel_day"] = rel
        all_recs.append(shifted)

    events_long = pd.concat(all_recs, ignore_index=True)
    lookup = df[["stockcode", "within_stock_idx"] + metrics].copy()

    result = events_long.merge(
        lookup,
        left_on=["stockcode", "target_within_idx"],
        right_on=["stockcode", "within_stock_idx"],
        how="inner",
    )
    return result[["rel_day"] + metrics]


event_panel = build_event_panel(df, df["attention_burst"], METRICS)
print(f"Event panel shape: {event_panel.shape}")

# ── 3. Plot Mean Paths ───────────────────────────────────────────────────────

agg = event_panel.groupby("rel_day")[METRICS].agg(["mean", "sem"])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Herding / Attention Burst Event Study\n(t=0: totalposts z-score > 2)", fontsize=14
)

for ax, col, label in [
    (axes[0, 0], "avgrepostnum", "Avg Repost Count"),
    (axes[0, 1], "avgthumbups", "Avg Thumbs Up"),
    (axes[1, 0], "totalusers", "Total Users"),
    (axes[1, 1], "ret", "Daily Return"),
]:
    means = agg[col]["mean"]
    sems = agg[col]["sem"]
    ax.plot(means.index, means.values, marker="o", markersize=3, linewidth=1.5)
    ax.fill_between(means.index, means - 1.96 * sems, means + 1.96 * sems, alpha=0.2)
    ax.axvline(0, color="red", linestyle="--", linewidth=1, label="Burst day")
    ax.axhline(means.iloc[0], color="gray", linestyle=":", linewidth=0.8)
    ax.set_title(label)
    ax.set_xlabel("Days relative to attention burst")
    ax.legend()

plt.tight_layout()
out = os.path.join(SCRIPT_DIR, "data", "01_herding_event_study.png")
plt.savefig(out, dpi=150)
plt.show()
print(f"Saved {out}")

# ── 4. Statistical Test: Post-Burst Returns ──────────────────────────────────

post_burst_ret = event_panel[event_panel["rel_day"].between(5, 20)]["ret"].dropna()
pre_burst_ret = event_panel[event_panel["rel_day"].between(-10, -1)]["ret"].dropna()

t_stat, p_val = stats.ttest_1samp(post_burst_ret, 0)
print("\nPost-burst (+5 to +20) returns:")
print(f"  Mean: {post_burst_ret.mean():.4f}  Std: {post_burst_ret.std():.4f}")
print(f"  t={t_stat:.3f}, p={p_val:.4f} (one-sample t-test vs 0)")

t2, p2 = stats.ttest_ind(post_burst_ret, pre_burst_ret)
print("\nPost-burst vs Pre-burst:")
print(f"  Pre-burst mean: {pre_burst_ret.mean():.4f}")
print(f"  t={t2:.3f}, p={p2:.4f} (two-sample t-test)")

# ── 5. Cumulative Return Around Burst ────────────────────────────────────────

cum_ret = agg["ret"]["mean"].cumsum()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(cum_ret.index, cum_ret.values, linewidth=2, color="steelblue")
ax.axvline(0, color="red", linestyle="--", label="Burst day")
ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
ax.fill_between(
    cum_ret.index,
    0,
    cum_ret.values,
    where=cum_ret.values < 0,
    alpha=0.3,
    color="red",
    label="Negative",
)
ax.fill_between(
    cum_ret.index,
    0,
    cum_ret.values,
    where=cum_ret.values >= 0,
    alpha=0.3,
    color="green",
    label="Positive",
)
ax.set_title("Cumulative Return Around Attention Burst (t=0)")
ax.set_xlabel("Days relative to attention burst")
ax.set_ylabel("Cumulative Return")
ax.legend()
plt.tight_layout()
out = os.path.join(SCRIPT_DIR, "data", "01_herding_cum_ret.png")
plt.savefig(out, dpi=150)
plt.show()
print(f"Saved {out}")

# ── 6. Summary Statistics ────────────────────────────────────────────────────

summary = pd.DataFrame(
    {
        "Window": ["Pre [-10,-1]", "Burst [0]", "Post [+1,+4]", "Post [+5,+20]"],
        "Mean Return": [
            event_panel[event_panel["rel_day"].between(-10, -1)]["ret"].mean(),
            event_panel[event_panel["rel_day"] == 0]["ret"].mean(),
            event_panel[event_panel["rel_day"].between(1, 4)]["ret"].mean(),
            event_panel[event_panel["rel_day"].between(5, 20)]["ret"].mean(),
        ],
        "N": [
            event_panel[event_panel["rel_day"].between(-10, -1)]["ret"].count(),
            event_panel[event_panel["rel_day"] == 0]["ret"].count(),
            event_panel[event_panel["rel_day"].between(1, 4)]["ret"].count(),
            event_panel[event_panel["rel_day"].between(5, 20)]["ret"].count(),
        ],
    }
)
summary["Mean Return"] = summary["Mean Return"].map("{:.4f}".format)
print("\n" + summary.to_string(index=False))
