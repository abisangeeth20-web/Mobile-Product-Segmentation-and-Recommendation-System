"""End-to-end pipeline:
collect -> clean -> feature engineering & encoding -> EDA -> K-Means segmentation -> recommender -> validate -> save.

Usage:  python run_pipeline.py
"""
import json
import time

import joblib
import pandas as pd

from src import config as C
from src import eda, features, preprocessing, recommender, report_figures, segmentation


def log(msg):
    print(msg, flush=True)


def main():
    t0 = time.time()
    for d in [C.PROCESSED_DIR, C.MODELS_DIR, C.REPORTS_DIR, C.FIGURES_DIR, C.TABLES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    log("1. Data collection (CSV -> DataFrame)")
    raw = preprocessing.load_raw()
    log(f"   {raw.shape[0]:,} rows x {raw.shape[1]} columns, {int(raw.isna().sum().sum()):,} missing cells")

    log("2. Cleaning & preprocessing")
    clean, clog = preprocessing.clean(raw)
    clean.to_csv(C.CLEANED_CSV, index=False, date_format=C.DATE_FORMAT)
    for k, v in clog.items():
        log(f"   {k}: {v}")

    log("3. Feature engineering, encoding & product catalog")
    feat = features.add_features(clean)
    features.encode_categoricals(feat).to_csv(C.FEATURED_CSV, index=False, date_format=C.DATE_FORMAT)
    products = features.product_catalog(feat)
    log(f"   review-level features: {feat.shape}; product catalog: {products.shape}")

    log("4. Clustering - customer (review) level")
    cust_k = segmentation.k_selection(feat[C.CUSTOMER_CLUSTER_FEATURES])
    cust_k.round(4).to_csv(C.REPORTS_DIR / "customer_k_selection.csv")
    cpipe, clabels, cnames, ccenters, cmetrics = segmentation.fit_customer_segments(feat)
    feat["customer_segment"] = clabels
    alt = segmentation.alternative_encoded_clustering(feat)
    cprof = segmentation.customer_segment_profile(feat)
    cprof.to_csv(C.REPORTS_DIR / "customer_segment_profile.csv")
    log(f"   silhouette {cmetrics['silhouette_sample10k']:.3f} | with one-hot brand+country: "
        f"{alt['silhouette_sample10k']:.3f}")

    log("5. Clustering - product (model) level")
    prod_k = segmentation.k_selection(products[segmentation.PRODUCT_FEATURES], k_range=range(2, 9))
    prod_k.round(4).to_csv(C.REPORTS_DIR / "product_k_selection.csv")
    ppipe, plabels, pnames, pmetrics = segmentation.fit_product_segments(products)
    products["product_segment"] = plabels
    feat["product_segment"] = feat.model.map(products.set_index("model").product_segment)
    pprof = segmentation.product_segment_profile(products)
    pprof.to_csv(C.REPORTS_DIR / "product_segment_profile.csv")
    variation = segmentation.product_feature_variation(products)
    variation.to_csv(C.REPORTS_DIR / "product_feature_variation.csv")
    products.to_csv(C.PRODUCTS_CSV, index=False)
    log(f"   product silhouette {pmetrics['silhouette']:.3f}")

    log("6. Exploratory data analysis")
    tables = eda.run_eda(feat, products, raw)

    log("7. Recommendation system")
    rec = recommender.ProductRecommender(products, feat)
    rec.sim["euclidean"].round(4).to_csv(C.MODELS_DIR / "similarity_euclidean.csv")
    rec.sim["cosine"].round(4).to_csv(C.MODELS_DIR / "similarity_cosine.csv")
    rec.item_sim.round(4).to_csv(C.MODELS_DIR / "similarity_item_cf.csv")
    ev = recommender.evaluate_content(rec)
    ev.to_csv(C.REPORTS_DIR / "recommender_content_evaluation.csv")
    cf = recommender.evaluate_cf(feat, products)
    personas = recommender.evaluate_personas(rec)
    personas.to_csv(C.REPORTS_DIR / "recommender_persona_checks.csv", index=False)
    examples = pd.concat([rec.recommend_similar(m, 3).assign(selected=m)
                          for m in ["iPhone 14", "Galaxy S24", "Pixel 8", "OnePlus 12", "Realme 12 Pro"]])
    examples[["selected", "model", "brand", "product_segment", "avg_price_usd", "avg_rating", "similarity",
              "reason"]].round(3).to_csv(C.REPORTS_DIR / "recommendation_examples.csv", index=False)
    log(f"   euclidean same-segment@3 {ev.iloc[0, 0]:.1f}% | cosine {ev.iloc[1, 0]:.1f}% | random {ev.iloc[2, 0]:.1f}%")
    log(f"   CF RMSE {cf['rmse_cf']:.3f} vs item-mean {cf['rmse_item']:.3f}")

    log("8. Report figures")
    report_figures.fig_k_selection(cust_k, C.N_CLUSTERS, "seg_01_k_selection_customer.png",
                                   "Choosing k - customer segmentation (price, rating, aspect average)")
    report_figures.fig_k_selection(prod_k, C.N_CLUSTERS, "seg_02_k_selection_product.png",
                                   "Choosing k - product segmentation (price structure of 22 models)")
    pca_df, var = segmentation.pca_2d(feat, C.CUSTOMER_CLUSTER_FEATURES)
    report_figures.fig_customer_clusters(feat, pca_df, var)
    report_figures.fig_customer_profiles(cprof)
    report_figures.fig_product_segments(products)
    report_figures.fig_segment_brand_mix(feat)
    report_figures.fig_similarity(rec.sim["euclidean"], products)
    report_figures.fig_rec_evaluation(ev, cf)

    log("9. Saving models & app data")
    joblib.dump(cpipe, C.MODELS_DIR / "kmeans_customer_segments.joblib", compress=3)
    joblib.dump(ppipe, C.MODELS_DIR / "kmeans_product_segments.joblib", compress=3)
    app_cols = ["review_id", "brand", "model", "price_usd", "price_local", "currency", "exchange_rate_to_usd",
                "rating", "sentiment", "country", "language", "review_date", "verified_purchase", *C.ASPECTS,
                "helpful_votes", "source", "age", "age_group", "aspect_avg", "is_positive", "sentiment_score",
                "engagement_score", "value_score", "price_band", "review_month", "customer_segment",
                "product_segment"]
    feat[app_cols].to_csv(C.SEGMENTED_GZ, index=False, date_format=C.DATE_FORMAT, compression="gzip")

    meta = {
        "rows": len(feat), "n_clusters": C.N_CLUSTERS,
        "customer_segmentation": {"features": C.CUSTOMER_CLUSTER_FEATURES, "metrics": cmetrics,
                                  "cluster_names": {str(k): v for k, v in cnames.items()},
                                  "centers": ccenters.to_dict(orient="index"),
                                  "alternative_with_encoded_categoricals": alt,
                                  "feature_ranges": {c: [float(feat[c].min()), float(feat[c].max())]
                                                     for c in C.CUSTOMER_CLUSTER_FEATURES}},
        "product_segmentation": {"features": segmentation.PRODUCT_FEATURES, "metrics": pmetrics,
                                 "cluster_names": {str(k): v for k, v in pnames.items()},
                                 "segments": products.groupby("product_segment").model.apply(list).to_dict()},
        "recommender": {"default_metric": "euclidean", "hybrid_weights": recommender.DEFAULT_WEIGHTS,
                        "content_evaluation": ev.reset_index().to_dict(orient="records"),
                        "cf_evaluation": cf, "cf_groups": recommender.GROUP_COLS},
        "imputation_check": clog["imputation_check"],
        "categories": {c: sorted(feat[c].unique().tolist()) for c in ["brand", "model", "country", "age_group",
                                                                       "source"]},
        "exchange_rates": feat.groupby("country")[["currency", "exchange_rate_to_usd"]].first()
                              .to_dict(orient="index"),
    }
    (C.MODELS_DIR / "model_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    summary = {"cleaning_log": clog, "customer_segment_profile": cprof.reset_index().to_dict(orient="records"),
               "product_segment_profile": pprof.reset_index().to_dict(orient="records"),
               "statistical_tests": tables["tests"], "cf_evaluation": cf}
    (C.REPORTS_DIR / "pipeline_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
