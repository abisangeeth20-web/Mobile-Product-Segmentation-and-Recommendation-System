"""Figures for the cluster-analysis and recommendation reports (reports/figures/seg_*.png, rec_*.png)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

from . import config as C
from .plotting import apply_style, save, usd
from .segmentation import CUSTOMER_SEGMENTS, PRODUCT_SEGMENT_COLORS, PRODUCT_SEGMENTS, SEGMENT_COLORS

MARKERS = dict(zip(CUSTOMER_SEGMENTS, ["o", "s", "^", "D"]))


def fig_k_selection(scores: pd.DataFrame, chosen: int, name: str, title: str):
    apply_style()
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    specs = [("inertia", "Elbow: within-cluster sum of squares", None),
             ("silhouette", "Silhouette (higher = better)", None),
             ("davies_bouldin", "Davies-Bouldin (lower = better)", None)]
    for a, (col, t, _) in zip(ax, specs):
        a.plot(scores.index, scores[col], marker="o", color=C.CATEGORICAL[0], markersize=6)
        a.scatter([chosen], [scores.loc[chosen, col]], s=120, color=C.CATEGORICAL[1], zorder=3,
                  label=f"chosen k = {chosen}")
        a.set_title(t, fontsize=10)
        a.set_xlabel("k")
        a.set_xticks(list(scores.index))
    ax[0].legend()
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    return save(fig, name)


def fig_customer_clusters(df: pd.DataFrame, pca_df: pd.DataFrame, var_ratio):
    apply_style()
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.4))
    s = df.sample(6000, random_state=C.RANDOM_STATE)
    jitter = np.random.RandomState(0).uniform(-0.25, 0.25, len(s))
    for seg in CUSTOMER_SEGMENTS:
        d = s[s.customer_segment == seg]
        j = jitter[s.customer_segment.to_numpy() == seg]
        ax[0].scatter(d.price_usd, d.rating + j, s=9, alpha=0.45, color=SEGMENT_COLORS[seg],
                      marker=MARKERS[seg], label=seg, edgecolors="none")
    ax[0].xaxis.set_major_formatter(FuncFormatter(usd))
    ax[0].set_xlabel("Price paid (USD)")
    ax[0].set_ylabel("Overall rating (jittered)")
    ax[0].set_title("Customer segments: price paid x satisfaction (sample of 6,000)")
    ax[0].legend(markerscale=2.2, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    for seg in CUSTOMER_SEGMENTS:
        d = pca_df[pca_df.customer_segment == seg]
        ax[1].scatter(d.PC1, d.PC2, s=9, alpha=0.45, color=SEGMENT_COLORS[seg], marker=MARKERS[seg],
                      edgecolors="none")
    ax[1].set_xlabel(f"PC1 ({var_ratio[0]:.0%} of variance)")
    ax[1].set_ylabel(f"PC2 ({var_ratio[1]:.0%} of variance)")
    ax[1].set_title("Same segments in 2-D PCA space")
    return save(fig, "seg_03_customer_clusters.png")


def fig_customer_profiles(profile: pd.DataFrame):
    apply_style()
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    cols = [("share_%", "Share of reviews (%)", "{:.1f}%"), ("avg_price_usd", "Avg price paid (USD)", "${:,.0f}"),
            ("avg_rating", "Avg overall rating", "{:.2f}"), ("positive_%", "Positive reviews (%)", "{:.0f}%")]
    colors = [SEGMENT_COLORS[s] for s in profile.index]
    labels = [s.replace(" · ", "\n") for s in profile.index]
    for a, (col, t, fmt) in zip(ax, cols):
        a.bar(labels, profile[col], color=colors, width=0.6)
        for x, v in enumerate(profile[col]):
            a.text(x, v, fmt.format(v), ha="center", va="bottom", fontsize=8, color=C.INK["secondary"])
        a.set_title(t, fontsize=10)
        a.tick_params(axis="x", labelsize=7.5)
        a.set_ylim(0, profile[col].max() * 1.18)
    fig.suptitle("Customer segment profiles", x=0.01, ha="left", fontweight="bold")
    return save(fig, "seg_04_customer_segment_profiles.png")


def fig_product_segments(products: pd.DataFrame):
    apply_style()
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.4, 1]})
    p = products.sort_values("avg_price_usd")
    for seg in PRODUCT_SEGMENTS:
        d = p[p.product_segment == seg]
        ax[0].hlines(d.model, d.min_price_usd, d.max_price_usd, color=PRODUCT_SEGMENT_COLORS[seg], linewidth=5,
                     alpha=0.35)
        ax[0].scatter(d.avg_price_usd, d.model, color=PRODUCT_SEGMENT_COLORS[seg], s=50, zorder=3, label=seg)
    ax[0].xaxis.set_major_formatter(FuncFormatter(usd))
    ax[0].set_title("Product segments: average price (dot) and 2nd-98th percentile price range (bar)")
    ax[0].legend(loc="lower right")
    ax[0].grid(axis="y", visible=False)
    for seg in PRODUCT_SEGMENTS:
        d = p[p.product_segment == seg]
        ax[1].scatter(d.avg_price_usd, d.bayes_rating, color=PRODUCT_SEGMENT_COLORS[seg], s=60, label=seg,
                      edgecolors=C.INK["surface"], linewidths=1.5)
    for _, r in p.iterrows():
        ax[1].annotate(r.model, (r.avg_price_usd, r.bayes_rating), fontsize=7, color=C.INK["secondary"],
                       xytext=(4, 3), textcoords="offset points")
    ax[1].xaxis.set_major_formatter(FuncFormatter(usd))
    ax[1].set_xlabel("Average price (USD)")
    ax[1].set_ylabel("Bayesian-average rating")
    ax[1].set_title("Price vs rating per model - no premium for paying more")
    return save(fig, "seg_05_product_segments.png")


def fig_segment_brand_mix(df: pd.DataFrame):
    apply_style()
    s = (pd.crosstab(df.customer_segment, df.brand, normalize="index") * 100).reindex(CUSTOMER_SEGMENTS)[C.BRANDS]
    fig, ax = plt.subplots(figsize=(13, 3.8))
    left = np.zeros(len(s))
    for b in C.BRANDS:
        ax.barh(s.index, s[b], left=left, color=C.BRAND_COLORS[b], label=b, height=0.6,
                edgecolor=C.INK["surface"], linewidth=2)
        for y, (l, v) in enumerate(zip(left, s[b])):
            if v >= 4:
                ax.text(l + v / 2, y, f"{v:.0f}%", ha="center", va="center", fontsize=8,
                        color=C.INK["primary"] if b == "OnePlus" else "#ffffff")
        left += s[b].to_numpy()
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_title("Brand mix inside each customer segment (%)")
    ax.legend(ncol=7, loc="lower center", bbox_to_anchor=(0.5, -0.3))
    ax.grid(axis="y", visible=False)
    return save(fig, "seg_06_segment_brand_mix.png")


def fig_similarity(sim: pd.DataFrame, products: pd.DataFrame):
    apply_style()
    order = products.sort_values("avg_price_usd").model
    m = sim.loc[order, order]
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(m, cmap=sns.color_palette(C.SEQ_BLUE, as_cmap=True), vmin=0, vmax=1, ax=ax, square=True,
                linewidths=0.5, linecolor=C.INK["surface"], cbar_kws={"shrink": 0.7, "label": "similarity"})
    ax.set_title("Product similarity (1 / (1 + Euclidean distance)), models ordered by price")
    ax.set_xlabel("")
    ax.set_ylabel("")
    return save(fig, "rec_01_similarity_heatmap.png")


def fig_rec_evaluation(ev: pd.DataFrame, cf: dict):
    apply_style()
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    col_seg = [c for c in ev.columns if c.startswith("same_segment")][0]
    colors = [C.CATEGORICAL[0], C.CATEGORICAL[1], C.INK["muted"]]
    for a, (col, t, fmt) in zip(ax[:2], [(col_seg, "Recommendations in same segment (%)", "{:.0f}%"),
                                         ("mean_abs_price_diff_usd", "Mean |price gap| to selected phone",
                                          "${:,.0f}")]):
        a.bar(ev.index, ev[col], color=colors, width=0.55)
        for x, v in enumerate(ev[col]):
            a.text(x, v, fmt.format(v), ha="center", va="bottom", fontsize=9, color=C.INK["secondary"])
        a.set_title(t, fontsize=10)
        a.set_ylim(0, ev[col].max() * 1.18)
    names = ["global", "item", "group", "cf"]
    vals = [cf[f"rmse_{n}"] for n in names]
    ax[2].bar(["Global mean", "Item mean", "Group mean", "Item-item CF"], vals,
              color=[C.INK["muted"]] * 3 + [C.CATEGORICAL[0]], width=0.55)
    for x, v in enumerate(vals):
        ax[2].text(x, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9, color=C.INK["secondary"])
    ax[2].set_title("Held-out rating RMSE (lower = better)", fontsize=10)
    ax[2].set_ylim(0, max(vals) * 1.18)
    ax[2].tick_params(axis="x", labelsize=8)
    fig.suptitle("Recommendation validation", x=0.01, ha="left", fontweight="bold")
    return save(fig, "rec_02_recommender_evaluation.png")
