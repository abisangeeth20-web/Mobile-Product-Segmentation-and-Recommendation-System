"""K-Means segmentation at two levels.

* Customer / review level (50,000 purchases): what the buyer paid x how satisfied they are
  (price_usd, rating, aspect_avg) -> 4 segments: Premium/Budget-mid x Satisfied/Dissatisfied.
* Product level (22 phone models): price structure (average, low end, high end of the observed price range)
  -> 4 segments: Budget, Mid-range, Upper mid-range, Premium.

Number of clusters is chosen from the elbow curve, silhouette, Davies-Bouldin and Calinski-Harabasz scores
(k = 2..10), together with business interpretability (the spec suggests 4).
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C

PRODUCT_SEGMENTS = ["Budget", "Mid-range", "Upper mid-range", "Premium"]
CUSTOMER_SEGMENTS = ["Premium · Satisfied", "Premium · Dissatisfied",
                     "Budget/Mid · Satisfied", "Budget/Mid · Dissatisfied"]
SEGMENT_COLORS = dict(zip(CUSTOMER_SEGMENTS, C.CATEGORICAL[:4]))
PRODUCT_SEGMENT_COLORS = dict(zip(PRODUCT_SEGMENTS, C.CATEGORICAL[:4]))
PRODUCT_FEATURES = ["avg_price_usd", "min_price_usd", "max_price_usd"]


def kmeans_pipeline(k: int, n_init=10) -> Pipeline:
    return Pipeline([("scale", StandardScaler()),
                     ("kmeans", KMeans(n_clusters=k, n_init=n_init, random_state=C.RANDOM_STATE))])


def k_selection(X: pd.DataFrame, k_range=C.K_RANGE, sample=10_000) -> pd.DataFrame:
    """Elbow + internal validity scores for each k (silhouette on a fixed random sample for speed)."""
    Xs = StandardScaler().fit_transform(X)
    rng = np.random.RandomState(C.RANDOM_STATE)
    idx = rng.choice(len(Xs), min(sample, len(Xs)), replace=False)
    rows = []
    for k in k_range:
        if k >= len(Xs):
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=C.RANDOM_STATE).fit(Xs)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(Xs[idx], km.labels_[idx]),
                     "davies_bouldin": davies_bouldin_score(Xs, km.labels_),
                     "calinski_harabasz": calinski_harabasz_score(Xs, km.labels_)})
    return pd.DataFrame(rows).set_index("k")


# --------------------------------------------------------------------------- customer segments
def _name_customer_clusters(centers: pd.DataFrame) -> dict:
    """Two highest-price centroids = Premium; within each price pair the higher rating = Satisfied."""
    order = centers.sort_values("price_usd", ascending=False).index
    names = {}
    for tier, pair in [("Premium", order[:2]), ("Budget/Mid", order[2:])]:
        hi, lo = sorted(pair, key=lambda i: -centers.loc[i, "rating"])
        names[hi], names[lo] = f"{tier} · Satisfied", f"{tier} · Dissatisfied"
    return names


def fit_customer_segments(df: pd.DataFrame, k=C.N_CLUSTERS):
    X = df[C.CUSTOMER_CLUSTER_FEATURES]
    pipe = kmeans_pipeline(k).fit(X)
    labels = pipe.predict(X)
    centers = pd.DataFrame(pipe.named_steps["scale"].inverse_transform(pipe.named_steps["kmeans"].cluster_centers_),
                           columns=C.CUSTOMER_CLUSTER_FEATURES)
    names = _name_customer_clusters(centers) if k == 4 else {i: f"Segment {i}" for i in range(k)}
    Xs = pipe.named_steps["scale"].transform(X)
    idx = np.random.RandomState(C.RANDOM_STATE).choice(len(Xs), min(10_000, len(Xs)), replace=False)
    metrics = {"silhouette_sample10k": float(silhouette_score(Xs[idx], labels[idx])),
               "davies_bouldin": float(davies_bouldin_score(Xs, labels)),
               "calinski_harabasz": float(calinski_harabasz_score(Xs, labels)),
               "inertia": float(pipe.named_steps["kmeans"].inertia_)}
    centers.index = [names[i] for i in centers.index]
    return pipe, pd.Series(labels).map(names).to_numpy(), names, centers.round(3), metrics


def alternative_encoded_clustering(df: pd.DataFrame, k=C.N_CLUSTERS) -> dict:
    """Comparison: also one-hot encode brand + country into the K-Means input.
    Used to justify the feature selection (clusters then split on categories, not on behaviour)."""
    num = C.CUSTOMER_CLUSTER_FEATURES + ["age", "engagement_score"]
    pre = ColumnTransformer([("num", StandardScaler(), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                              ["brand", "country"])])
    X = pre.fit_transform(df)
    km = KMeans(n_clusters=k, n_init=10, random_state=C.RANDOM_STATE).fit(X)
    idx = np.random.RandomState(C.RANDOM_STATE).choice(len(X), 10_000, replace=False)
    prof = df.assign(c=km.labels_).groupby("c").agg(price=("price_usd", "mean"), rating=("rating", "mean"),
                                                    n=("rating", "size"))
    return {"features": num + ["brand (one-hot)", "country (one-hot)"],
            "silhouette_sample10k": float(silhouette_score(X[idx], km.labels_[idx])),
            "davies_bouldin": float(davies_bouldin_score(X, km.labels_)),
            "cluster_rating_spread": float(prof.rating.max() - prof.rating.min()),
            "cluster_price_spread": float(prof.price.max() - prof.price.min())}


# --------------------------------------------------------------------------- product segments
def fit_product_segments(products: pd.DataFrame, k=C.N_CLUSTERS):
    X = products[PRODUCT_FEATURES]
    pipe = kmeans_pipeline(k, n_init=50).fit(X)
    labels = pipe.predict(X)
    price_by_cluster = products.groupby(labels).avg_price_usd.mean().sort_values()
    names = dict(zip(price_by_cluster.index, PRODUCT_SEGMENTS if k == 4 else [f"Tier {i + 1}" for i in range(k)]))
    Xs = pipe.named_steps["scale"].transform(X)
    metrics = {"silhouette": float(silhouette_score(Xs, labels)),
               "davies_bouldin": float(davies_bouldin_score(Xs, labels)),
               "calinski_harabasz": float(calinski_harabasz_score(Xs, labels))}
    return pipe, pd.Series(labels).map(names).to_numpy(), names, metrics


def product_feature_variation(products: pd.DataFrame) -> pd.DataFrame:
    """Coefficient of variation across the 22 models - shows why rating features are not clustering inputs."""
    cols = ["avg_price_usd", "avg_rating", "positive_share", "avg_aspect", "avg_engagement", "avg_age"]
    s = products[cols]
    return pd.DataFrame({"mean": s.mean(), "std": s.std(), "cv_%": s.std() / s.mean() * 100}).round(4)


def pca_2d(df: pd.DataFrame, cols: list, n=6000):
    """2-D PCA projection (sample) for plotting clusters."""
    d = df.sample(min(n, len(df)), random_state=C.RANDOM_STATE)
    Xs = StandardScaler().fit_transform(d[cols])
    pca = PCA(n_components=2, random_state=C.RANDOM_STATE).fit(Xs)
    xy = pca.transform(Xs)
    return d.assign(PC1=xy[:, 0], PC2=xy[:, 1]), pca.explained_variance_ratio_


def customer_segment_profile(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("customer_segment")
    prof = pd.DataFrame({
        "reviews": g.size(),
        "share_%": g.size() / len(df) * 100,
        "avg_price_usd": g.price_usd.mean(),
        "avg_rating": g.rating.mean(),
        "avg_aspect": g.aspect_avg.mean(),
        "positive_%": g.is_positive.mean() * 100,
        "negative_%": g.sentiment.apply(lambda s: (s == "Negative").mean() * 100),
        "avg_helpful_votes": g.helpful_votes.mean(),
        "verified_%": g.verified_purchase.mean() * 100,
        "avg_age": g.age.mean(),
        "top_brand": g.brand.agg(lambda s: s.value_counts().index[0]),
        "top_model": g.model.agg(lambda s: s.value_counts().index[0]),
        **{C.ASPECT_LABELS[a]: g[a].mean() for a in C.ASPECTS},
    })
    return prof.reindex([s for s in CUSTOMER_SEGMENTS if s in prof.index]).round(2)


def product_segment_profile(products: pd.DataFrame) -> pd.DataFrame:
    g = products.groupby("product_segment")
    prof = pd.DataFrame({
        "models": g.size(),
        "brands": g.brand.agg(lambda s: ", ".join(sorted(s.unique()))),
        "reviews": g.n_reviews.sum(),
        "avg_price_usd": g.avg_price_usd.mean(),
        "price_range_usd": g.apply(lambda x: f"${x.min_price_usd.min():,.0f} - ${x.max_price_usd.max():,.0f}",
                                   include_groups=False),
        "avg_rating": g.avg_rating.mean(),
        "positive_%": g.positive_share.mean() * 100,
        "value_score": g.avg_value_score.mean(),
    })
    return prof.reindex([s for s in PRODUCT_SEGMENTS if s in prof.index]).round(3)
