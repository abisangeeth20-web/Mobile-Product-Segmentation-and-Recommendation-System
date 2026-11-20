"""Similarity-based product recommendation.

Three pieces, all built only from structured data:

1. Content-based "similar products" - each phone model is a feature vector (price structure, overall rating,
   positive-review share, five aspect ratings). Similarity = cosine similarity or a distance metric
   (Euclidean -> 1 / (1 + d)). Both are evaluated; the better one is the default.
2. Preference-based recommendations - a user gives a budget, the aspects they care about (battery, camera...)
   and optionally their country / age group. Each model in budget is scored on the aspects that matter to the
   user, as rated by buyers like them.
3. Collaborative filtering (item-item) - "users" are customer groups (country x age group), because 93% of
   customer names occur only once. Adjusted-cosine similarity between items on the group x item rating matrix
   predicts how a group would rate a phone it has not been scored on.
"""
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from . import config as C

PRICE_COLS = ["avg_price_usd", "min_price_usd", "max_price_usd"]
QUALITY_COLS = ["avg_rating", "positive_share"] + [f"avg_{a}" for a in C.ASPECTS]
GROUP_COLS = ["country", "age_group"]
DEFAULT_WEIGHTS = {"aspect": 0.50, "cf": 0.10, "positive": 0.20, "budget": 0.20}


class ProductRecommender:
    def __init__(self, products: pd.DataFrame, reviews: pd.DataFrame, group_cols=GROUP_COLS):
        self.products = products.set_index("model").copy()
        cols = ["model", "price_usd", "rating", "is_positive", *C.ASPECTS, *group_cols]
        self.reviews = reviews[cols].copy()
        self.group_cols = group_cols
        self.vectors = self._content_vectors()
        self.sim = {"cosine": pd.DataFrame(cosine_similarity(self.vectors), self.vectors.index, self.vectors.index),
                    "euclidean": pd.DataFrame(1 / (1 + euclidean_distances(self.vectors)),
                                              self.vectors.index, self.vectors.index)}
        self.R, self.R_counts = self._group_matrix(self.reviews)
        self.item_sim = self._item_similarity(self.R)

    # ----------------------------------------------------------------------- content-based
    def _content_vectors(self) -> pd.DataFrame:
        """Price columns are z-scored across models. Quality columns are centred and divided by the
        review-level standard deviation, so they weigh in proportion to how much models really differ
        (a 0.05-star gap between models is small next to the 1.3-star spread between reviews)."""
        p = self.products
        price = (p[PRICE_COLS] - p[PRICE_COLS].mean()) / p[PRICE_COLS].std()
        rv = self.reviews
        review_std = {"avg_rating": rv.rating.std(), "positive_share": rv.is_positive.std(),
                      **{f"avg_{a}": rv[a].std() for a in C.ASPECTS}}
        quality = (p[QUALITY_COLS] - p[QUALITY_COLS].mean()) / pd.Series(review_std)
        return pd.concat([price, quality], axis=1)

    def recommend_similar(self, model: str, k=5, metric="euclidean", exclude_same_brand=False) -> pd.DataFrame:
        s = self.sim[metric][model].drop(model)
        cand = self.products.loc[s.index].assign(similarity=s)
        if exclude_same_brand:
            cand = cand[cand.brand != self.products.loc[model, "brand"]]
        out = cand.sort_values("similarity", ascending=False).head(k)
        base = self.products.loc[model]
        out["price_diff_usd"] = out.avg_price_usd - base.avg_price_usd
        out["rating_diff"] = out.avg_rating - base.avg_rating
        out["reason"] = [self._reason(base, r) for _, r in out.iterrows()]
        return out.reset_index()

    @staticmethod
    def _reason(base, r) -> str:
        parts = []
        if "product_segment" in r and r.product_segment == base.get("product_segment"):
            parts.append(f"same {r.product_segment} segment")
        d = r.avg_price_usd - base.avg_price_usd
        parts.append(f"avg price {'+' if d >= 0 else '-'}${abs(d):,.0f}")
        best = max(C.ASPECTS, key=lambda a: r[f"avg_{a}"])
        parts.append(f"strongest aspect: {C.ASPECT_LABELS[best]}")
        return " · ".join(parts)

    # ----------------------------------------------------------------------- collaborative filtering
    def _group_key(self, df):
        return df[self.group_cols].astype(str).agg(" | ".join, axis=1)

    def _group_matrix(self, reviews):
        g = reviews.assign(group=self._group_key(reviews))
        R = g.pivot_table(index="group", columns="model", values="rating", aggfunc="mean")
        N = g.pivot_table(index="group", columns="model", values="rating", aggfunc="size").fillna(0)
        return R, N

    @staticmethod
    def _item_similarity(R: pd.DataFrame) -> pd.DataFrame:
        """Adjusted cosine: centre each group's ratings on that group's mean, then cosine between item columns."""
        centred = R.sub(R.mean(axis=1), axis=0).fillna(0)
        return pd.DataFrame(cosine_similarity(centred.T), R.columns, R.columns)

    def cf_predict(self, group: str, model: str, n_neighbors=5, R=None, item_sim=None) -> float:
        R = self.R if R is None else R
        item_sim = self.item_sim if item_sim is None else item_sim
        if group not in R.index:
            return float(R[model].mean())
        row = R.loc[group]
        mu = row.mean()
        sims = item_sim[model].drop(model)
        rated = row.drop(model).dropna()
        sims = sims[rated.index]
        sims = sims[sims > 0].nlargest(n_neighbors)
        if sims.empty:
            return float(mu)
        return float(np.clip(mu + (sims * (rated[sims.index] - mu)).sum() / sims.abs().sum(), 1, 5))

    def similar_by_ratings(self, model: str, k=5) -> pd.DataFrame:
        """Collaborative 'customers who rated X highly also rated Y highly' neighbours."""
        s = self.item_sim[model].drop(model).sort_values(ascending=False).head(k)
        return self.products.loc[s.index, ["brand", "product_segment", "avg_price_usd", "avg_rating"]].assign(
            cf_similarity=s.round(3)).reset_index()

    # ----------------------------------------------------------------------- preference-based (hybrid)
    def recommend_for_user(self, budget_min=0.0, budget_max=1e9, priorities=None, country=None, age_group=None,
                           brands=None, min_positive_share=0.0, k=5, weights=None) -> pd.DataFrame:
        """Score every phone that is sold within the budget.

        score = 0.50 * aspect match (priority-weighted aspect ratings from buyers like the user, /5)
              + 0.10 * collaborative-filtering predicted rating for the user's group (/5)
              + 0.20 * positive-review share
              + 0.20 * share of that model's observed prices inside the budget
        """
        # CF gets a small weight: on this dataset it does not beat the item-mean baseline (see evaluate_cf)
        w = {**DEFAULT_WEIGHTS, **(weights or {})}
        priorities = priorities or {a: 1 for a in C.ASPECTS}
        tot = sum(priorities.values()) or 1
        rv = self.reviews
        peer = rv
        if country:
            peer = peer[peer.country == country]
        if age_group:
            peer = peer[peer.age_group == age_group]
        rows = []
        for model, p in self.products.iterrows():
            if brands and p.brand not in brands:
                continue
            prices = rv.loc[rv.model == model, "price_usd"]
            budget_fit = float(prices.between(budget_min, budget_max).mean())
            if budget_fit == 0 or p.positive_share < min_positive_share:
                continue
            pr = peer[peer.model == model]
            src = pr if len(pr) >= 30 else rv[rv.model == model]
            aspect = sum(src[a].mean() * wt for a, wt in priorities.items()) / tot
            group = " | ".join([country or "", age_group or ""]) if country and age_group else None
            cf = self.cf_predict(group, model) if group else float(src.rating.mean())
            score = (w["aspect"] * aspect / 5 + w["cf"] * cf / 5 + w["positive"] * p.positive_share
                     + w["budget"] * budget_fit)
            in_budget = prices[prices.between(budget_min, budget_max)]
            rows.append({"model": model, "brand": p.brand, "product_segment": p.get("product_segment", ""),
                         "typical_price_in_budget_usd": float(in_budget.median()),
                         "budget_fit": budget_fit, "aspect_match": aspect, "cf_predicted_rating": cf,
                         "positive_share": p.positive_share, "peer_reviews": int(len(pr)), "score": score})
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("score", ascending=False).head(k).reset_index(drop=True)


# =========================================================================== validation
def evaluate_content(rec: ProductRecommender, k=3) -> pd.DataFrame:
    """Relevance of 'similar products' for every model in the catalog, per similarity metric,
    against a random-recommendation baseline."""
    P = rec.products
    rng = np.random.RandomState(C.RANDOM_STATE)
    rows = []
    for metric in ["euclidean", "cosine", "random"]:
        seg, pdiff, rdiff, brand_div, recs = [], [], [], [], set()
        for m in P.index:
            if metric == "random":
                picks = list(rng.choice(P.index.drop(m), k, replace=False))
            else:
                picks = list(rec.recommend_similar(m, k, metric).model)
            recs.update(picks)
            seg.append(np.mean(P.loc[picks, "product_segment"] == P.loc[m, "product_segment"]))
            pdiff.append(np.mean(np.abs(P.loc[picks, "avg_price_usd"] - P.loc[m, "avg_price_usd"])))
            rdiff.append(np.mean(np.abs(P.loc[picks, "avg_rating"] - P.loc[m, "avg_rating"])))
            brand_div.append(np.mean(P.loc[picks, "brand"] != P.loc[m, "brand"]))
        rows.append({"metric": metric, f"same_segment@{k}_%": np.mean(seg) * 100,
                     "mean_abs_price_diff_usd": np.mean(pdiff), "mean_abs_rating_diff": np.mean(rdiff),
                     "other_brand_share_%": np.mean(brand_div) * 100,
                     "catalog_coverage_%": len(recs) / len(P) * 100})
    return pd.DataFrame(rows).set_index("metric").round(3)


def evaluate_cf(reviews: pd.DataFrame, products: pd.DataFrame, test_frac=0.2, k=3) -> dict:
    """Hold out 20% of reviews; predict each held-out (group, model) mean rating from the training matrix."""
    test = reviews.sample(frac=test_frac, random_state=C.RANDOM_STATE)
    train = reviews.drop(test.index)
    rec = ProductRecommender(products, train)
    test = test.assign(group=rec._group_key(test))
    cell = test.groupby(["group", "model"]).rating.agg(["mean", "size"]).reset_index()
    gmean = train.rating.mean()
    item_mean = train.groupby("model").rating.mean()
    group_mean = rec.R.mean(axis=1)
    cell["pred_global"] = gmean
    cell["pred_item"] = cell.model.map(item_mean)
    cell["pred_group"] = cell.group.map(group_mean).fillna(gmean)
    # CF: predict each cell with that cell itself hidden from the group's row
    preds = []
    for g, m in zip(cell.group, cell.model):
        R = rec.R.copy()
        if g in R.index:
            R.loc[g, m] = np.nan
        preds.append(rec.cf_predict(g, m, R=R))
    cell["pred_cf"] = preds
    out = {}
    w = cell["size"]
    for c in ["pred_global", "pred_item", "pred_group", "pred_cf"]:
        err = cell[c] - cell["mean"]
        out[c.replace("pred_", "rmse_")] = float(np.sqrt(np.average(err ** 2, weights=w)))
        out[c.replace("pred_", "mae_")] = float(np.average(err.abs(), weights=w))
    # ranking quality: does CF's top-k per group overlap the group's actual top-k in the held-out data?
    hits, rand = [], k / len(products)
    for g, sub in cell.groupby("group"):
        if len(sub) < 10:
            continue
        actual = set(sub.nlargest(k, "mean").model)
        pred = set(sub.nlargest(k, "pred_cf").model)
        hits.append(len(actual & pred) / k)
    out[f"precision@{k}_cf"] = float(np.mean(hits))
    out[f"precision@{k}_random_expected"] = float(rand)
    out["groups"] = int(rec.R.shape[0])
    out["test_cells"] = int(len(cell))
    return {k_: round(v, 4) if isinstance(v, float) else v for k_, v in out.items()}


PERSONAS = [
    {"name": "Student on a budget", "budget_min": 150, "budget_max": 450,
     "priorities": {"battery_life_rating": 3, "performance_rating": 2}, "country": "India", "age_group": "18-24"},
    {"name": "Camera-first creator", "budget_min": 600, "budget_max": 1100,
     "priorities": {"camera_rating": 3, "display_rating": 2}, "country": "USA", "age_group": "25-34"},
    {"name": "Premium professional", "budget_min": 1000, "budget_max": 1600,
     "priorities": {"performance_rating": 2, "design_rating": 2}, "country": "UK", "age_group": "35-44"},
    {"name": "Mid-range all-rounder", "budget_min": 400, "budget_max": 800,
     "priorities": {a: 1 for a in C.ASPECTS}, "country": "Germany", "age_group": "45-54"},
]


def evaluate_personas(rec: ProductRecommender, k=5) -> pd.DataFrame:
    """Constraint checks for preference-based recommendations on example user profiles."""
    rows = []
    for p in PERSONAS:
        r = rec.recommend_for_user(p["budget_min"], p["budget_max"], p["priorities"], p["country"],
                                   p["age_group"], k=k)
        in_budget = r.typical_price_in_budget_usd.between(p["budget_min"], p["budget_max"]).mean()
        catalog_aspect = np.mean([rec.reviews[a].mean() for a in p["priorities"]])
        rows.append({"persona": p["name"], "budget_usd": f"{p['budget_min']}-{p['budget_max']}",
                     "priorities": ", ".join(C.ASPECT_LABELS[a] for a in p["priorities"]),
                     "top_recommendations": ", ".join(r.model.head(3)),
                     "in_budget_%": in_budget * 100,
                     "avg_priority_score_recs": r.aspect_match.mean(),
                     "catalog_priority_score": catalog_aspect})
    return pd.DataFrame(rows).round(3)
