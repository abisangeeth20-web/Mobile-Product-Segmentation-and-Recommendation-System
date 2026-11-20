"""Exploratory data analysis: summary tables, statistical tests and figures (reports/figures/eda_*.png)."""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scipy import stats

from . import config as C
from .plotting import apply_style, save, usd

BLUE = C.SEQ_BLUE[4]


def _save_table(df, name):
    C.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.TABLES_DIR / f"{name}.csv")
    return df


def summary_tables(df: pd.DataFrame, products: pd.DataFrame) -> dict:
    t = {}
    t["numeric_summary"] = df[["price_usd", "rating", "aspect_avg", *C.ASPECTS, "age", "helpful_votes",
                               "engagement_score", "value_score"]].describe().T.round(3)
    t["brand_summary"] = df.groupby("brand").agg(
        reviews=("rating", "size"), models=("model", "nunique"), avg_price_usd=("price_usd", "mean"),
        min_price_usd=("price_usd", "min"), max_price_usd=("price_usd", "max"), avg_rating=("rating", "mean"),
        positive_share=("is_positive", "mean"), avg_helpful=("helpful_votes", "mean"),
        avg_value_score=("value_score", "mean")).sort_values("avg_price_usd", ascending=False).round(3)
    t["country_summary"] = df.groupby("country").agg(
        reviews=("rating", "size"), currency=("currency", "first"), avg_price_usd=("price_usd", "mean"),
        avg_rating=("rating", "mean"), positive_share=("is_positive", "mean"),
        top_brand=("brand", lambda s: s.value_counts().index[0])).round(3)
    t["brand_by_country"] = pd.crosstab(df.brand, df.country)
    t["brand_share_by_age_group_%"] = (pd.crosstab(df.age_group, df.brand, normalize="index") * 100).round(2)
    t["price_band_summary"] = df.groupby("price_band").agg(
        reviews=("rating", "size"), avg_rating=("rating", "mean"), positive_share=("is_positive", "mean"),
        **{C.ASPECT_LABELS[a]: (a, "mean") for a in C.ASPECTS}).round(3)
    t["source_summary"] = df.groupby("source").agg(reviews=("rating", "size"), avg_rating=("rating", "mean"),
                                                   positive_share=("is_positive", "mean")).round(3)
    rank = products.sort_values("bayes_rating", ascending=False)[
        ["model", "brand", "n_reviews", "avg_price_usd", "avg_rating", "bayes_rating", "positive_share",
         "negative_share"]].round(4)
    t["top5_products"] = rank.head(5).set_index("model")
    t["bottom5_products"] = rank.tail(5).set_index("model")
    t["product_catalog"] = products.set_index("model")
    corr_cols = ["price_usd", "rating", *C.ASPECTS, "age", "helpful_votes", "verified_purchase", "sentiment_score"]
    t["correlation"] = df[corr_cols].corr().round(3)
    for k, v in t.items():
        _save_table(v, k)
    return t


def statistical_tests(df: pd.DataFrame) -> dict:
    """Do brands / countries / price really differ in satisfaction? (alpha = 0.05)"""
    res = {}
    groups = [g.rating.to_numpy() for _, g in df.groupby("brand")]
    f, p = stats.f_oneway(*groups)
    res["anova_rating_by_brand"] = {"F": f, "p": p}
    h, p = stats.kruskal(*groups)
    res["kruskal_rating_by_brand"] = {"H": h, "p": p}
    groups = [g.rating.to_numpy() for _, g in df.groupby("model")]
    f, p = stats.f_oneway(*groups)
    res["anova_rating_by_model"] = {"F": f, "p": p}
    f, p = stats.f_oneway(*[g.price_usd.to_numpy() for _, g in df.groupby("brand")])
    res["anova_price_by_brand"] = {"F": f, "p": p}
    for col in ["brand", "country", "source"]:
        chi2, p, dof, _ = stats.chi2_contingency(pd.crosstab(df[col], df.sentiment))
        res[f"chi2_sentiment_vs_{col}"] = {"chi2": chi2, "dof": dof, "p": p}
    chi2, p, dof, _ = stats.chi2_contingency(pd.crosstab(df.age_group, df.brand))
    res["chi2_brand_vs_age_group"] = {"chi2": chi2, "dof": dof, "p": p}
    chi2, p, dof, _ = stats.chi2_contingency(pd.crosstab(df.country, df.brand))
    res["chi2_brand_vs_country"] = {"chi2": chi2, "dof": dof, "p": p}
    r, p = stats.pearsonr(df.price_usd, df.rating)
    res["pearson_price_vs_rating"] = {"r": r, "p": p}
    r, p = stats.spearmanr(df.price_usd, df.aspect_avg)
    res["spearman_price_vs_aspect_avg"] = {"rho": r, "p": p}
    r, p = stats.pearsonr(df.helpful_votes, df.rating)
    res["pearson_helpful_votes_vs_rating"] = {"r": r, "p": p}
    t, p = stats.ttest_ind(df[df.verified_purchase == 1].rating, df[df.verified_purchase == 0].rating,
                           equal_var=False)
    res["welch_t_rating_verified_vs_not"] = {"t": t, "p": p}
    out = {k: {kk: float(vv) for kk, vv in v.items()} for k, v in res.items()}
    for v in out.values():
        v["significant"] = bool(v["p"] < 0.05)
    C.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (C.TABLES_DIR / "statistical_tests.json").write_text(json.dumps(out, indent=2))
    return out


# --------------------------------------------------------------------------- figures
def fig_brand_country(df):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1, 1.5]})
    b = df.brand.value_counts().reindex(C.BRANDS)
    ax[0].barh(b.index[::-1], b.values[::-1], color=BLUE, height=0.6)
    for y, v in enumerate(b.values[::-1]):
        ax[0].text(v, y, f" {v:,}", va="center", color=C.INK["secondary"], fontsize=9)
    ax[0].set_title("Reviews per brand")
    ax[0].grid(axis="y", visible=False)
    ct = pd.crosstab(df.brand, df.country).reindex(C.BRANDS)
    sns.heatmap(ct, annot=True, fmt=",d", cmap=sns.color_palette(C.SEQ_BLUE, as_cmap=True), cbar=False,
                linewidths=1, linecolor=C.INK["surface"], ax=ax[1], annot_kws={"fontsize": 8})
    ax[1].set_title("Reviews by brand x country (colour range is narrow: 837-961)")
    ax[1].tick_params(axis="y", rotation=0)
    ax[1].set_xlabel("")
    ax[1].set_ylabel("")
    return save(fig, "eda_01_brand_country_distribution.png")


def fig_price_by_brand(df):
    fig, ax = plt.subplots(figsize=(12, 4.6))
    order = df.groupby("brand").price_usd.median().sort_values().index
    sns.boxplot(data=df, x="brand", y="price_usd", order=order, hue="brand", palette=C.BRAND_COLORS,
                width=0.55, fliersize=0, linewidth=1, ax=ax, legend=False)
    ax.yaxis.set_major_formatter(FuncFormatter(usd))
    ax.set_title("Price paid (USD) by brand - each brand sells in its own price band")
    ax.set_xlabel("")
    ax.set_ylabel("Price (USD)")
    return save(fig, "eda_02_price_by_brand.png")


def fig_rating_sentiment(df):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    r = df.rating.value_counts().sort_index()
    ax[0].bar(r.index, r.values, color=BLUE, width=0.6)
    ax[0].set_title("Overall rating distribution")
    ax[0].set_xlabel("Rating (stars)")
    ax[0].set_ylabel("Reviews")
    s = (pd.crosstab(df.brand, df.sentiment, normalize="index") * 100).reindex(C.BRANDS)[
        ["Negative", "Neutral", "Positive"]]
    left = np.zeros(len(s))
    for col in s.columns:
        ax[1].barh(s.index, s[col], left=left, color=C.SENTIMENT_COLORS[col], label=col, height=0.6,
                   edgecolor=C.INK["surface"], linewidth=2)
        for y, (l, v) in enumerate(zip(left, s[col])):
            ax[1].text(l + v / 2, y, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                       color="#ffffff" if col != "Neutral" else C.INK["primary"])
        left += s[col].to_numpy()
    ax[1].invert_yaxis()
    ax[1].set_xlim(0, 100)
    ax[1].set_title("Sentiment mix by brand (%) - nearly identical")
    ax[1].legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.22))
    ax[1].grid(axis="y", visible=False)
    return save(fig, "eda_03_rating_sentiment.png")


def fig_top_low_products(products):
    p = products.sort_values("bayes_rating")
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.barh(p.model, p.bayes_rating, color=[C.BRAND_COLORS[b] for b in p.brand], height=0.65)
    for y, (v, n) in enumerate(zip(p.bayes_rating, p.n_reviews)):
        ax.text(v, y, f"  {v:.3f}  (n={n:,})", va="center", fontsize=8, color=C.INK["secondary"])
    lo, hi = p.bayes_rating.min(), p.bayes_rating.max()
    ax.set_xlim(lo - 0.03, hi + 0.04)
    ax.set_title("Products ranked by Bayesian-average rating (axis zoomed: total spread is only "
                 f"{hi - lo:.2f} stars)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=C.BRAND_COLORS[b]) for b in C.BRANDS]
    ax.legend(handles, C.BRANDS, ncol=7, loc="lower center", bbox_to_anchor=(0.5, -0.13), fontsize=8)
    ax.grid(axis="y", visible=False)
    return save(fig, "eda_04_top_low_products.png")


def fig_price_vs_rating(df):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    bins = pd.cut(df.price_usd, np.arange(150, 1551, 100))
    g = df.groupby(bins, observed=True).agg(rating=("rating", "mean"), aspect=("aspect_avg", "mean"),
                                           n=("rating", "size"))
    x = [i.mid for i in g.index]
    ax[0].plot(x, g.rating, marker="o", color=C.CATEGORICAL[0], label="Overall rating")
    ax[0].plot(x, g.aspect, marker="o", color=C.CATEGORICAL[1], label="Aspect average")
    ax[0].set_ylim(1, 5)
    ax[0].xaxis.set_major_formatter(FuncFormatter(usd))
    ax[0].set_title("Average rating by price ($100 bins) - flat")
    ax[0].set_xlabel("Price paid (USD)")
    ax[0].legend()
    pv = df.groupby("price_band").agg(**{C.ASPECT_LABELS[a]: (a, "mean") for a in C.ASPECTS})
    from .features import PRICE_BAND_LABELS
    pv = pv.reindex(PRICE_BAND_LABELS)
    w = 0.16
    for i, col in enumerate(pv.columns):
        ax[1].bar(np.arange(len(pv)) + (i - 2) * w, pv[col], width=w * 0.9, color=C.CATEGORICAL[i], label=col)
    ax[1].set_xticks(range(len(pv)), [s.split(" (")[0] for s in pv.index])
    ax[1].set_ylim(1, 3.2)
    ax[1].set_title("Aspect ratings by price band (axis from 1)")
    ax[1].legend(ncol=5, fontsize=8, loc="upper center")
    return save(fig, "eda_05_price_vs_rating.png")


def fig_correlation(df):
    cols = ["price_usd", "rating", *C.ASPECTS, "age", "helpful_votes", "verified_purchase", "sentiment_score"]
    corr = df[cols].corr()
    labels = [C.ASPECT_LABELS.get(c, c) for c in cols]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=sns.diverging_palette(10, 250, s=75, l=50, as_cmap=True),
                vmin=-1, vmax=1, center=0, linewidths=1, linecolor=C.INK["surface"], ax=ax,
                xticklabels=labels, yticklabels=labels, annot_kws={"fontsize": 8})
    ax.set_title("Correlation matrix")
    return save(fig, "eda_06_correlation.png")


def fig_trend(df):
    m = df.groupby("review_month").agg(reviews=("rating", "size"), rating=("rating", "mean"),
                                       positive=("is_positive", "mean"))
    m.index = pd.PeriodIndex(m.index, freq="M").to_timestamp()
    m = m.iloc[1:-1]   # drop partial first / last months
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
    ax[0].plot(m.index, m.reviews, color=C.CATEGORICAL[0])
    ax[0].set_ylim(0, m.reviews.max() * 1.15)
    ax[0].set_title("Reviews per month")
    ax[1].plot(m.index, m.rating, color=C.CATEGORICAL[0])
    ax[1].set_ylim(2.5, 3.7)
    ax[1].set_title("Average rating per month - stable")
    return save(fig, "eda_07_monthly_trend.png")


def fig_preferences(df):
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    a = (pd.crosstab(df.age_group, df.brand, normalize="index") * 100)[C.BRANDS]
    sns.heatmap(a, annot=True, fmt=".1f", cmap=sns.color_palette(C.SEQ_BLUE, as_cmap=True), cbar=False,
                linewidths=1, linecolor=C.INK["surface"], ax=ax[0], annot_kws={"fontsize": 8})
    ax[0].set_title("Brand share within each age group (%)")
    ax[0].tick_params(axis="y", rotation=0)
    ax[0].set_xlabel("")
    ax[0].set_ylabel("Age group")
    pr = df.groupby("country").agg(**{C.ASPECT_LABELS[a]: (a, "mean") for a in C.ASPECTS},
                                   Overall=("rating", "mean"))
    sns.heatmap(pr, annot=True, fmt=".2f", cmap=sns.color_palette(C.SEQ_BLUE, as_cmap=True), cbar=False,
                linewidths=1, linecolor=C.INK["surface"], ax=ax[1], annot_kws={"fontsize": 8})
    ax[1].set_title("Average aspect ratings by country")
    ax[1].set_ylabel("")
    return save(fig, "eda_08_customer_preferences.png")


def fig_engagement(df):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.2))
    h = df.groupby("rating").helpful_votes.mean()
    ax[0].bar(h.index, h.values, color=BLUE, width=0.6)
    ax[0].set_title("Helpful votes per review by rating")
    ax[0].set_xlabel("Rating (stars)")
    s = df.groupby("source").agg(rating=("rating", "mean"), n=("rating", "size")).sort_values("rating")
    ax[1].barh(s.index, s.rating, color=BLUE, height=0.6)
    ax[1].set_xlim(2.9, 3.25)
    for y, (v, n) in enumerate(zip(s.rating, s.n)):
        ax[1].text(v, y, f"  {v:.3f} (n={n:,})", va="center", fontsize=8, color=C.INK["secondary"])
    ax[1].set_title("Average rating by review source (axis zoomed)")
    ax[1].grid(axis="y", visible=False)
    return save(fig, "eda_09_engagement_source.png")


def fig_missing(raw):
    n = raw.isna().sum()
    n = n[n > 0].sort_values()
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.barh(n.index, n.values, color=BLUE, height=0.55)
    for y, v in enumerate(n.values):
        ax.text(v, y, f" {v:,} ({v / len(raw):.1%})", va="center", fontsize=9, color=C.INK["secondary"])
    ax.set_xlim(0, n.max() * 1.3)
    ax.set_title("Missing values in the raw file")
    ax.grid(axis="y", visible=False)
    return save(fig, "eda_00_missing_values.png")


def run_eda(df: pd.DataFrame, products: pd.DataFrame, raw: pd.DataFrame | None = None) -> dict:
    apply_style()
    tables = summary_tables(df, products)
    tables["tests"] = statistical_tests(df)
    if raw is not None:
        fig_missing(raw)
    fig_brand_country(df)
    fig_price_by_brand(df)
    fig_rating_sentiment(df)
    fig_top_low_products(products)
    fig_price_vs_rating(df)
    fig_correlation(df)
    fig_trend(df)
    fig_preferences(df)
    fig_engagement(df)
    return tables
