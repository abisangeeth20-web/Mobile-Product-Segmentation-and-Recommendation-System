"""Data collection (CSV -> DataFrame) and cleaning.

Profiling of the raw file showed one exact identity and one strong relationship:

* price_local == price_usd x exchange_rate_to_usd   (holds on 100% of complete rows, |error| <= 0.005)
  -> a missing price_usd or price_local is recovered exactly from the other one.
* rating is the overall star rating and tracks the mean of the five aspect ratings (r = 0.76 per aspect);
  sentiment is in turn driven by rating.
  -> a missing rating / sentiment is imputed with the most frequent value among complete reviews that share
     the same aspect-rating average (and sentiment / rating). The accuracy of this rule is measured on held-out
     complete rows and logged, so the choice is evidence-based.
"""
import numpy as np
import pandas as pd

from . import config as C

TEXT_COLS = ["customer_name", "brand", "model", "price_local", "currency", "sentiment", "country", "language",
             "source"]


def load_raw(path=C.RAW_CSV) -> pd.DataFrame:
    """Step 1 - CSV to DataFrame."""
    return pd.read_csv(path)


def parse_local_price(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """'₹27996.73' / '$1,010.68 ' -> (symbol, 27996.73)."""
    s = s.astype("string").str.strip()
    symbol = s.str.extract(r"^(\D*?)\s*[0-9]")[0].str.strip()
    value = pd.to_numeric(s.str.replace(",", "", regex=False).str.extract(r"([0-9]+(?:\.[0-9]+)?)\s*$")[0],
                          errors="coerce")
    return symbol, value


def _mode_lookup(train: pd.DataFrame, keys: list, target: str) -> pd.Series:
    return train.groupby(keys)[target].agg(lambda x: x.value_counts().index[0])


def _impute_by_mode(df, mask, keys_list, target):
    """Fill df.loc[mask, target] with the conditional mode; try the most specific key set first."""
    complete = df[df[target].notna() & df[keys_list[0]].notna().all(axis=1)]
    filled = pd.Series(np.nan, index=df.index, dtype=object)
    for keys in keys_list:
        todo = mask & filled.isna() & df[keys].notna().all(axis=1)
        if not todo.any():
            continue
        lut = _mode_lookup(complete, keys, target)
        idx = pd.MultiIndex.from_frame(df.loc[todo, keys]) if len(keys) > 1 else df.loc[todo, keys[0]]
        filled.loc[todo] = lut.reindex(idx).to_numpy()
    return filled


def imputation_accuracy(df: pd.DataFrame, seed=C.RANDOM_STATE) -> dict:
    """Hold out 20% of complete rows, impute them with the same rule, and measure accuracy."""
    d = df.dropna(subset=["rating", "sentiment"]).copy()
    d["aspect_avg"] = d[C.ASPECTS].mean(axis=1).round(1)
    test = d.sample(frac=0.2, random_state=seed)
    train = d.drop(test.index)
    r_lut = _mode_lookup(train, ["aspect_avg", "sentiment"], "rating")
    r_pred = r_lut.reindex(pd.MultiIndex.from_frame(test[["aspect_avg", "sentiment"]])).to_numpy()
    s_lut = _mode_lookup(train, ["rating", "aspect_avg"], "sentiment")
    s_pred = s_lut.reindex(pd.MultiIndex.from_frame(test[["rating", "aspect_avg"]])).to_numpy()
    rating_mode = train.rating.mode()[0]
    sent_mode = train.sentiment.mode()[0]
    return {
        "rating_rule_accuracy": round(float(np.mean(r_pred == test.rating.to_numpy())), 4),
        "rating_rule_within_1_star": round(float(np.mean(np.abs(r_pred.astype(float) - test.rating) <= 1)), 4),
        "rating_global_mode_accuracy": round(float((test.rating == rating_mode).mean()), 4),
        "sentiment_rule_accuracy": round(float(np.mean(s_pred == test.sentiment.to_numpy())), 4),
        "sentiment_global_mode_accuracy": round(float((test.sentiment == sent_mode).mean()), 4),
    }


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Step 2 - handle missing values, duplicates, type conversion and validation."""
    df = raw.copy()
    log = {"raw_shape": list(df.shape), "raw_nulls": {k: int(v) for k, v in df.isna().sum().items() if v}}

    # --- text standardisation
    for c in TEXT_COLS:
        df[c] = df[c].astype("string").str.strip()
        df.loc[df[c] == "", c] = pd.NA
    df["sentiment"] = df.sentiment.str.title()

    # --- duplicates (exact rows, repeated review_id, same review content under a new id)
    n0 = len(df)
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset="review_id")
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "review_id"])
    log["duplicates_removed"] = n0 - len(df)

    # --- type conversion
    df["review_date"] = pd.to_datetime(df.review_date, format=C.RAW_DATE_FORMAT, errors="coerce")
    df["verified_purchase"] = df.verified_purchase.astype(str).str.lower().map({"true": True, "false": False})
    for c in ["age", "helpful_votes", *C.ASPECTS]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    log["unparseable_dates"] = int(df.review_date.isna().sum())

    # --- validation of reference relationships
    log["brand_model_conflicts"] = int((df.groupby("model").brand.nunique() > 1).sum())
    log["country_currency_conflicts"] = int((df.groupby("country")[["currency", "exchange_rate_to_usd"]]
                                             .nunique() > 1).any(axis=1).sum())
    log["out_of_range_ratings"] = int(((df[C.ASPECTS] < 1) | (df[C.ASPECTS] > 5)).sum().sum()
                                      + ((df.rating < 1) | (df.rating > 5)).sum())
    log["invalid_age"] = int(((df.age < 13) | (df.age > 100)).sum())

    # --- prices: parse the text column, validate and use the exact identity
    symbol, local = parse_local_price(df.price_local)
    df["price_local_value"] = local
    symbols = symbol.groupby(df.currency).agg(lambda x: x.dropna().mode().iloc[0])
    both = df.price_usd.notna() & df.price_local_value.notna()
    err = (df.price_local_value - df.price_usd * df.exchange_rate_to_usd).abs()
    bad = both & (err > 0.01 * df.exchange_rate_to_usd + 0.01)
    log["price_identity_holds_share"] = round(float(1 - bad.sum() / both.sum()), 6)
    df.loc[bad, "price_local_value"] = (df.price_usd * df.exchange_rate_to_usd).round(2)
    log["price_identity_violations_fixed"] = int(bad.sum())

    m = df.price_usd.isna() & df.price_local_value.notna()
    df.loc[m, "price_usd"] = (df.price_local_value / df.exchange_rate_to_usd).round(2)
    log["price_usd_recovered_exactly"] = int(m.sum())
    m2 = df.price_local_value.isna() & df.price_usd.notna()
    df.loc[m2, "price_local_value"] = (df.price_usd * df.exchange_rate_to_usd).round(2)
    log["price_local_recovered_exactly"] = int(m2.sum())

    both_missing = df.price_usd.isna()
    df["price_imputed"] = both_missing
    df.loc[both_missing, "price_usd"] = df.groupby("model").price_usd.transform("median")[both_missing]
    df.loc[both_missing, "price_local_value"] = (df.price_usd * df.exchange_rate_to_usd).round(2)[both_missing]
    log["price_both_missing_model_median"] = int(both_missing.sum())
    # rebuild a clean, consistently formatted text column
    df["price_local"] = df.currency.map(symbols) + df.price_local_value.map("{:.2f}".format)

    # --- rating & sentiment: conditional-mode imputation
    log["imputation_check"] = imputation_accuracy(df)
    df["aspect_avg"] = df[C.ASPECTS].mean(axis=1).round(1)
    r_mask = df.rating.isna()
    df["rating_imputed"] = r_mask
    df.loc[r_mask, "rating"] = _impute_by_mode(df, r_mask, [["aspect_avg", "sentiment"], ["aspect_avg"]],
                                               "rating")[r_mask].astype(float)
    log["rating_imputed"] = int(r_mask.sum())
    s_mask = df.sentiment.isna()
    df["sentiment_imputed"] = s_mask
    df.loc[s_mask, "sentiment"] = _impute_by_mode(df, s_mask, [["rating", "aspect_avg"], ["rating"]],
                                                  "sentiment")[s_mask]
    log["sentiment_imputed"] = int(s_mask.sum())
    df = df.drop(columns="aspect_avg")

    # --- source: no relationship with any other column (uniform across countries) -> explicit category
    log["source_filled_unknown"] = int(df.source.isna().sum())
    df["source"] = df.source.fillna("Unknown")

    # --- anything still unusable is dropped (none expected)
    before = len(df)
    df = df.dropna(subset=["brand", "model", "price_usd", "rating", "review_date"])
    log["rows_dropped_unrecoverable"] = before - len(df)

    df["rating"] = df.rating.astype(int)
    df["price_usd"] = df.price_usd.astype(float).round(2)
    df["price_local_value"] = df.price_local_value.astype(float).round(2)
    for c in ["age", "helpful_votes", *C.ASPECTS]:
        df[c] = df[c].astype(int)
    for c in ["customer_name", "brand", "model", "price_local", "currency", "sentiment", "country", "language",
              "source"]:
        df[c] = df[c].astype(str)
    df = df.sort_values("review_id").reset_index(drop=True)
    log["remaining_nulls"] = int(df.isna().sum().sum())
    log["clean_shape"] = list(df.shape)
    return df, log
