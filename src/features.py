"""Feature engineering, categorical encoding and the product-level catalog."""
import numpy as np
import pandas as pd

from . import config as C

# Fixed USD price bands used for descriptive EDA (clustering discovers its own segments independently)
PRICE_BAND_EDGES = [0, 400, 700, 1000, np.inf]
PRICE_BAND_LABELS = ["Budget (<$400)", "Mid-range ($400-700)", "Upper-mid ($700-1000)", "Premium ($1000+)"]


def add_features(clean: pd.DataFrame) -> pd.DataFrame:
    df = clean.copy()
    df["aspect_avg"] = df[C.ASPECTS].mean(axis=1).round(2)          # overall spec / performance score (1-5)
    df["rating_gap"] = (df.rating - df.aspect_avg).round(2)          # overall rating vs aspect scores
    df["sentiment_score"] = df.sentiment.map(C.SENTIMENT_SCORE).astype(int)
    df["is_positive"] = (df.sentiment == "Positive").astype(int)
    # Engagement: helpful votes, with verified-purchase reviews weighted 1.5x (more trusted by readers)
    df["engagement_score"] = (df.helpful_votes * np.where(df.verified_purchase, 1.5, 1.0)).round(1)
    # Value for money: overall rating points per $100 paid
    df["value_score"] = (df.rating / (df.price_usd / 100)).round(3)
    df["price_band"] = pd.cut(df.price_usd, PRICE_BAND_EDGES, labels=PRICE_BAND_LABELS, right=False).astype(str)
    df["age_group"] = pd.cut(df.age, C.AGE_BINS, labels=C.AGE_LABELS).astype(str)
    df["review_year"] = df.review_date.dt.year
    df["review_month"] = df.review_date.dt.to_period("M").astype(str)
    df["verified_purchase"] = df.verified_purchase.astype(int)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label codes (compact, for tree models / storage) + one-hot columns for brand and country."""
    out = df.copy()
    for c in ["brand", "model", "country", "source"]:
        out[f"{c}_code"] = out[c].astype("category").cat.codes
    out = pd.concat([out, pd.get_dummies(out.brand, prefix="brand", dtype=int),
                     pd.get_dummies(out.country, prefix="country", dtype=int)], axis=1)
    return out


def product_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """One row per phone model - the item table used for product segmentation and recommendations."""
    g = df.groupby("model")
    p = pd.DataFrame({
        "brand": g.brand.first(),
        "n_reviews": g.size(),
        "avg_price_usd": g.price_usd.mean(),
        "median_price_usd": g.price_usd.median(),
        "min_price_usd": g.price_usd.quantile(0.02),          # robust price range (2nd-98th percentile)
        "max_price_usd": g.price_usd.quantile(0.98),
        "avg_rating": g.rating.mean(),
        "positive_share": g.is_positive.mean(),
        "negative_share": g.sentiment.apply(lambda s: (s == "Negative").mean()),
        **{f"avg_{a}": g[a].mean() for a in C.ASPECTS},
        "avg_aspect": g.aspect_avg.mean(),
        "avg_value_score": g.value_score.mean(),
        "avg_engagement": g.engagement_score.mean(),
        "verified_share": g.verified_purchase.mean(),
        "avg_age": g.age.mean(),
        "top_country": g.country.agg(lambda s: s.value_counts().index[0]),
    })
    # Bayesian (shrunk) rating - fair ranking when review counts differ
    m, prior = 500, df.rating.mean()
    p["bayes_rating"] = (p.n_reviews * p.avg_rating + m * prior) / (p.n_reviews + m)
    p["popularity_rank"] = p.n_reviews.rank(ascending=False, method="min").astype(int)
    return p.reset_index().round(4)
