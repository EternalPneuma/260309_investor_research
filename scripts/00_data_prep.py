# 00 Data Preparation
# - Extracts full sentiment data from quant_db.xq_investor_sentiment
# - Fetches daily stock returns via AkShare/东财 (with incremental cache)
# - Merges and saves to research/data/sentiment_with_returns.parquet

import os
import gc
import time
import warnings
import pandas as pd
import numpy as np
import akshare as ak
from sqlalchemy import create_engine

# Disable proxy — 东财 is a domestic service; proxies cause connection failures
for _var in ('HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy'):
    os.environ.pop(_var, None)

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
OUT_PATH = os.path.join(DATA_DIR, 'sentiment_with_returns.parquet')
PRICE_CACHE = os.path.join(DATA_DIR, 'price_cache.parquet')

DB_URL = 'postgresql+psycopg2://pneuma:qweasdzxc123@127.0.0.1:5432/quant_db'

# 1. Load Sentiment Data from DB

engine = create_engine(DB_URL)

query = """
SELECT
    postdate,
    stockcode,
    totalposts,
    totalusers,
    avgrepostnum,
    avgthumbups,
    avgposterfans,
    avgposterattention,
    bullishsentindexa,
    bearishposts,
    bullishposts,
    avgwords,
    avgnetcomments,
    sentconformindex,
    avgbullishthumbups,
    avgbearishthumbups
FROM xq_investor_sentiment
ORDER BY postdate, stockcode
"""

print('Loading sentiment data from DB...')
sentiment = pd.read_sql(query, engine, parse_dates=['postdate'])
print(f'Loaded {len(sentiment):,} rows, {sentiment["stockcode"].nunique():,} unique stocks')
print(f'Date range: {sentiment["postdate"].min()} to {sentiment["postdate"].max()}')

stocks = sorted(sentiment['stockcode'].unique().tolist())
date_min = sentiment['postdate'].min()
date_max = sentiment['postdate'].max()
start_str = date_min.strftime('%Y%m%d')
end_str = date_max.strftime('%Y%m%d')
print(f'{len(stocks)} stocks, {start_str} to {end_str}')

# 2. Fetch Daily Returns via AkShare (东财)

def _get_symbol(code):
    """Prepend exchange prefix for ak.stock_zh_a_daily (e.g. '000001' → 'sz000001')."""
    if code.startswith('6') or code.startswith('9'):
        return f'sh{code}'
    elif code.startswith('8') or code.startswith('4'):
        return f'bj{code}'
    else:
        return f'sz{code}'


def fetch_stock_prices(stockcodes, start, end, cache_path, batch_size=100, sleep_sec=0.2):
    """Fetch daily close + market cap for a list of stocks. Saves/loads cache."""
    if os.path.exists(cache_path):
        cached = pd.read_parquet(cache_path)
        done = set(cached['stockcode'].unique())
        print(f'Cache hit: {len(done)} stocks already fetched')
    else:
        cached = pd.DataFrame()
        done = set()

    todo = [s for s in stockcodes if s not in done]
    total = len(todo)
    total_all = len(stockcodes)
    print(f'Fetching {total} remaining stocks (skipping {total_all - total} cached)...')

    results = []
    errors = []
    for i, code in enumerate(todo):
        idx = i + 1
        try:
            df = ak.stock_zh_a_daily(
                symbol=_get_symbol(code),
                start_date=start,
                end_date=end,
                adjust='hfq',
            )
            if df is None or df.empty:
                print(f'  [{idx}/{total}] {code} | {start}~{end} | EMPTY — skipped')
                errors.append((code, 'empty'))
                continue

            df = df.rename(columns={
                'date': 'postdate',
                'amount': 'turnover_amount',
                'turnover': 'turnover_rate',  # already a decimal fraction
            })
            df['postdate'] = pd.to_datetime(df['postdate'])
            df['stockcode'] = code

            keep = [c for c in ['postdate', 'stockcode', 'close',
                                 'turnover_amount', 'turnover_rate'] if c in df.columns]
            df = df[keep]

            actual_start = df['postdate'].min().strftime('%Y%m%d')
            actual_end = df['postdate'].max().strftime('%Y%m%d')
            print(f'  [{idx}/{total}] {code} | {actual_start}~{actual_end} | {len(df)} rows')

            results.append(df)

        except Exception as e:
            print(f'  [{idx}/{total}] {code} | {start}~{end} | ERROR: {e}')
            errors.append((code, str(e)))

        time.sleep(sleep_sec)

        if idx % batch_size == 0:
            print(f'--- checkpoint {idx}/{total}: saving cache, {len(errors)} errors so far ---')
            if results:
                chunk = pd.concat(results, ignore_index=True)
                combined = pd.concat([cached, chunk], ignore_index=True) if not cached.empty else chunk
                combined.to_parquet(cache_path, index=False)
                cached = combined
                results = []
                gc.collect()

    if results:
        chunk = pd.concat(results, ignore_index=True)
        combined = pd.concat([cached, chunk], ignore_index=True) if not cached.empty else chunk
        combined.to_parquet(cache_path, index=False)
        cached = combined

    failed_codes = [c for c, _ in errors]
    print(f'Done. {len(errors)} stocks failed: {failed_codes[:10]}{"..." if len(errors) > 10 else ""}')
    return cached


prices = fetch_stock_prices(stocks, start_str, end_str, PRICE_CACHE)

# 3. Compute Returns and Market Cap Proxy

prices = prices.sort_values(['stockcode', 'postdate'])

# Compute daily return from close prices
prices['ret'] = prices.groupby('stockcode')['close'].pct_change()

for h in [1, 5, 10, 20]:
    prices[f'fwd_ret_{h}d'] = (
        prices.groupby('stockcode')['close']
        .transform(lambda x: x.shift(-h) / x - 1)
    )

for h in [5, 20]:
    prices[f'past_ret_{h}d'] = (
        prices.groupby('stockcode')['close']
        .transform(lambda x: x / x.shift(h) - 1)
    )

if 'turnover_amount' in prices.columns and 'turnover_rate' in prices.columns:
    # turnover_rate is already a decimal fraction from ak.stock_zh_a_daily
    prices['mktcap'] = prices['turnover_amount'] / prices['turnover_rate'].replace(0, np.nan)
    prices['ln_mktcap'] = np.log(prices['mktcap'].clip(lower=1))

print(f'Prices shape: {prices.shape}')

# 4. Merge Sentiment + Prices

price_cols = ['postdate', 'stockcode', 'close', 'ret',
              'fwd_ret_1d', 'fwd_ret_5d', 'fwd_ret_10d', 'fwd_ret_20d',
              'past_ret_5d', 'past_ret_20d', 'mktcap', 'ln_mktcap']
price_cols = [c for c in price_cols if c in prices.columns]

merged = sentiment.merge(prices[price_cols], on=['postdate', 'stockcode'], how='left')

print(f'Merged shape: {merged.shape}')
print(f'Return coverage: {merged["ret"].notna().mean():.1%}')
print(merged.describe())

merged.to_parquet(OUT_PATH, index=False)
print(f'\nSaved to {OUT_PATH} ({os.path.getsize(OUT_PATH)/1e6:.1f} MB)')
