"""Streamlit app - Mobile Product Segmentation & Recommendation System.

Run:  streamlit run app.py
"""
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import config as C
from src.recommender import ProductRecommender
from src.segmentation import CUSTOMER_SEGMENTS, PRODUCT_SEGMENT_COLORS, PRODUCT_SEGMENTS, SEGMENT_COLORS

st.set_page_config(page_title="Mobile Segmentation & Recommender", page_icon="📱", layout="wide")

GRID = "rgba(137,135,129,0.25)"
SEGMENT_TEXT = {
    "Premium · Satisfied": "Paid ~$1,030 and loved it (≈90% positive). Brand advocates - ideal for referrals, "
                           "upgrades and accessory upsell.",
    "Premium · Dissatisfied": "Paid ~$1,040 but rated it ~2.1★. The highest-risk group: expensive phone, unmet "
                              "expectations - priority for service recovery and retention offers.",
    "Budget/Mid · Satisfied": "Paid ~$490 and are happy (≈90% positive). Value seekers - promote value bundles "
                              "and trade-up paths to mid-range.",
    "Budget/Mid · Dissatisfied": "Paid ~$500 and rated it ~2.1★. The largest group - quality fixes and better "
                                 "expectation-setting at the low end reach the most customers.",
}


# --------------------------------------------------------------------------- loaders
@st.cache_data
def load_data():
    df = pd.read_csv(C.SEGMENTED_GZ, parse_dates=["review_date"])
    products = pd.read_csv(C.PRODUCTS_CSV)
    return df, products


@st.cache_resource
def load_models():
    cust = joblib.load(C.MODELS_DIR / "kmeans_customer_segments.joblib")
    meta = json.loads((C.MODELS_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    return cust, meta


@st.cache_resource
def load_recommender():
    df, products = load_data()
    return ProductRecommender(products, df)


def style(fig, height=360):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend_title_text="",
                      font=dict(family="system-ui, Segoe UI, sans-serif"), title_x=0)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def usd(v):
    return f"${v:,.0f}"


# --------------------------------------------------------------------------- pages
def page_recommend(df, products, rec, meta):
    st.title("📱 Phone Recommendations")
    st.caption("Similarity-based recommendations built only from structured data: price structure, overall "
               "rating, positive-review share and the five aspect ratings of 50,000 reviews.")
    tab1, tab2 = st.tabs(["🔁 Similar to a phone", "🎯 Find my phone (preferences)"])

    with tab1:
        c1, c2, c3, c4 = st.columns([2, 1.3, 1, 1.2])
        model = c1.selectbox("Pick a phone", sorted(products.model), index=sorted(products.model).index("Pixel 8"))
        metric = c2.radio("Similarity metric", ["euclidean", "cosine"], horizontal=True,
                          help="Euclidean (distance-based) scored best in validation: 100% of its top-3 "
                               "recommendations fall in the same product segment vs 80% for cosine.")
        k = c3.slider("How many", 3, 10, 5)
        other = c4.toggle("Other brands only", False)

        base = products.set_index("model").loc[model]
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Brand", base.brand)
        k2.metric("Product segment", base.product_segment)
        k3.metric("Average price", usd(base.avg_price_usd),
                  help=f"Observed range {usd(base.min_price_usd)} – {usd(base.max_price_usd)}")
        k4.metric("Average rating", f"{base.avg_rating:.2f} ★")
        k5.metric("Positive reviews", f"{base.positive_share:.1%}")

        r = rec.recommend_similar(model, k, metric, exclude_same_brand=other)
        c1, c2 = st.columns([3, 2])
        show = r[["model", "brand", "product_segment", "avg_price_usd", "avg_rating", "positive_share",
                  "similarity", "reason"]].rename(columns={
                      "model": "Model", "brand": "Brand", "product_segment": "Segment", "avg_price_usd": "Avg price",
                      "avg_rating": "Rating", "positive_share": "Positive", "similarity": "Similarity",
                      "reason": "Why"})
        c1.markdown(f"**Phones most similar to {model}**")
        c1.dataframe(show.style.format({"Avg price": "${:,.0f}", "Rating": "{:.2f}", "Positive": "{:.1%}",
                                        "Similarity": "{:.3f}"}), hide_index=True, width="stretch")
        fig = px.bar(r.iloc[::-1], x="similarity", y="model", orientation="h", color="product_segment",
                     color_discrete_map=PRODUCT_SEGMENT_COLORS, text_auto=".3f",
                     title=f"Similarity to {model} ({metric})", range_x=[0, 1.05])
        fig.update_yaxes(title="")
        c2.plotly_chart(style(fig, 330), width="stretch")

        st.markdown("**Price position of the recommendations**")
        cand = products.assign(role=np.where(products.model == model, "Selected",
                                             np.where(products.model.isin(r.model), "Recommended", "Other")))
        fig = px.scatter(cand, x="avg_price_usd", y="avg_rating", color="role", text="model",
                         color_discrete_map={"Selected": C.CATEGORICAL[1], "Recommended": C.CATEGORICAL[0],
                                             "Other": "#c3c2b7"},
                         labels={"avg_price_usd": "Average price (USD)", "avg_rating": "Average rating"})
        fig.update_traces(textposition="top center", marker=dict(size=11), textfont_size=9)
        st.plotly_chart(style(fig, 380), width="stretch")
        st.caption("Rating differences between models are tiny (≈0.1★ total spread, not statistically "
                   "significant), so similarity is driven mostly by price structure - by design the quality "
                   "features are scaled by review-level variation and only count as much as models really differ.")

        st.markdown("**Collaborative filtering view - customer groups that rate this phone above their "
                    "average also rate these above average**")
        st.dataframe(rec.similar_by_ratings(model, 5).rename(columns={
            "model": "Model", "brand": "Brand", "product_segment": "Segment", "avg_price_usd": "Avg price",
            "avg_rating": "Rating", "cf_similarity": "CF similarity"}).style.format(
            {"Avg price": "${:,.0f}", "Rating": "{:.2f}", "CF similarity": "{:.3f}"}), hide_index=True,
            width="stretch")
        st.caption("Item-item adjusted-cosine similarity on a 40 customer-group (country × age group) × 22 phone "
                   "rating matrix. On this dataset it does not beat the item-average baseline (see Model "
                   "performance), so treat it as illustrative.")

    with tab2:
        rates = meta["exchange_rates"]
        with st.form("prefs"):
            c1, c2, c3 = st.columns(3)
            country = c1.selectbox("Your country", sorted(rates), index=sorted(rates).index("India"))
            cur, fx = rates[country]["currency"], rates[country]["exchange_rate_to_usd"]
            age = c2.selectbox("Your age group", C.AGE_LABELS, index=1)
            brands = c3.multiselect("Brands (optional)", C.BRANDS)
            lo_usd, hi_usd = 150, 1500
            budget = st.slider(f"Budget ({cur})", int(lo_usd * fx), int(hi_usd * fx),
                               (int(250 * fx), int(700 * fx)), step=max(1, int(10 * fx)))
            st.markdown("**How much does each aspect matter to you?** (0 = don't care, 5 = critical)")
            cols = st.columns(5)
            pr = {a: cols[i].slider(C.ASPECT_LABELS[a], 0, 5, 3 if a in ("camera_rating", "battery_life_rating")
                                    else 1) for i, a in enumerate(C.ASPECTS)}
            k = st.slider("Number of recommendations", 3, 10, 5)
            go_btn = st.form_submit_button("Recommend", type="primary", width="stretch")

        if not go_btn:
            st.info("Set your budget and priorities, then press **Recommend**.")
            return
        if sum(pr.values()) == 0:
            st.error("Give at least one aspect a weight above 0.")
            return
        bmin, bmax = budget[0] / fx, budget[1] / fx
        res = rec.recommend_for_user(bmin, bmax, {a: w for a, w in pr.items() if w > 0}, country, age,
                                     brands or None, k=k)
        if res.empty:
            st.warning("No phone in the catalog sells within this budget / brand filter. Widen the budget.")
            return
        st.success(f"Top {len(res)} phones for a {age} buyer in {country} with a budget of "
                   f"{cur} {budget[0]:,} – {budget[1]:,} (≈ {usd(bmin)} – {usd(bmax)})")
        out = res.assign(local_price=res.typical_price_in_budget_usd * fx)
        st.dataframe(out[["model", "brand", "product_segment", "local_price", "typical_price_in_budget_usd",
                          "budget_fit", "aspect_match", "cf_predicted_rating", "positive_share", "peer_reviews",
                          "score"]].rename(columns={
            "model": "Model", "brand": "Brand", "product_segment": "Segment",
            "local_price": f"Typical price ({cur})", "typical_price_in_budget_usd": "Typical price (USD)",
            "budget_fit": "Budget fit", "aspect_match": "Your-priority score (/5)",
            "cf_predicted_rating": "CF predicted ★", "positive_share": "Positive", "peer_reviews": "Peer reviews",
            "score": "Score"}).style.format({
            f"Typical price ({cur})": "{:,.0f}", "Typical price (USD)": "${:,.0f}", "Budget fit": "{:.0%}",
            "Your-priority score (/5)": "{:.2f}", "CF predicted ★": "{:.2f}", "Positive": "{:.1%}",
            "Score": "{:.3f}"}), hide_index=True, width="stretch")

        w = meta["recommender"]["hybrid_weights"]
        parts = pd.DataFrame({
            "Model": res.model,
            "Priority aspects": w["aspect"] * res.aspect_match / 5,
            "Collaborative filtering": w["cf"] * res.cf_predicted_rating / 5,
            "Positive share": w["positive"] * res.positive_share,
            "Budget fit": w["budget"] * res.budget_fit}).melt("Model", var_name="Component", value_name="Points")
        fig = px.bar(parts, x="Points", y="Model", color="Component", orientation="h",
                     color_discrete_sequence=C.CATEGORICAL[:4], title="Score breakdown")
        fig.update_yaxes(categoryorder="total ascending", title="")
        fig.update_traces(marker_line_color="#fcfcfb", marker_line_width=2)
        st.plotly_chart(style(fig, 340), width="stretch")
        st.caption(f"Score = {w['aspect']:.2f}×priority-aspect rating from buyers in your country and age group "
                   f"(≥30 peer reviews, else all buyers) + {w['cf']:.2f}×collaborative-filtering predicted rating "
                   f"+ {w['positive']:.2f}×positive-review share + {w['budget']:.2f}×share of the phone's observed "
                   "prices inside your budget.")


def page_segments(df, products, cust_pipe, meta):
    st.title("🧩 Segmentation (K-Means)")
    tab1, tab2, tab3 = st.tabs(["👥 Customer segments", "📦 Product segments", "🧭 Which segment am I?"])
    cm = meta["customer_segmentation"]

    with tab1:
        st.caption("K-Means (k = 4) on standardised price paid, overall rating and aspect-rating average of all "
                   f"50,000 reviews · silhouette {cm['metrics']['silhouette_sample10k']:.3f}")
        prof = df.groupby("customer_segment").agg(
            Reviews=("rating", "size"), Avg_price=("price_usd", "mean"), Avg_rating=("rating", "mean"),
            Positive=("is_positive", "mean"), Helpful=("helpful_votes", "mean"),
            Top_brand=("brand", lambda s: s.value_counts().index[0])).reindex(CUSTOMER_SEGMENTS)
        cols = st.columns(4)
        for col, seg in zip(cols, CUSTOMER_SEGMENTS):
            p = prof.loc[seg]
            col.markdown(f"<div style='border-left:6px solid {SEGMENT_COLORS[seg]};padding-left:10px'>"
                         f"<b>{seg}</b><br>{p.Reviews:,.0f} reviews ({p.Reviews / len(df):.0%})<br>"
                         f"avg {usd(p.Avg_price)} · {p.Avg_rating:.2f}★ · {p.Positive:.0%} positive</div>",
                         unsafe_allow_html=True)
            col.caption(SEGMENT_TEXT[seg])
        c1, c2 = st.columns(2)
        s = df.sample(5000, random_state=1)
        s = s.assign(rating_j=s.rating + np.random.RandomState(0).uniform(-0.3, 0.3, len(s)))
        fig = px.scatter(s, x="price_usd", y="rating_j", color="customer_segment", symbol="customer_segment",
                         color_discrete_map=SEGMENT_COLORS, opacity=0.5,
                         category_orders={"customer_segment": CUSTOMER_SEGMENTS},
                         labels={"price_usd": "Price paid (USD)", "rating_j": "Rating (jittered)"},
                         hover_data=["model", "country", "rating"], title="Price paid × satisfaction (5,000 sample)")
        fig.update_traces(marker_size=5)
        c1.plotly_chart(style(fig, 420), width="stretch")
        bm = (pd.crosstab(df.customer_segment, df.brand, normalize="index") * 100).reindex(CUSTOMER_SEGMENTS)
        bm = bm[C.BRANDS].reset_index().melt("customer_segment", var_name="Brand", value_name="Share %")
        fig = px.bar(bm, x="Share %", y="customer_segment", color="Brand", orientation="h",
                     color_discrete_map=C.BRAND_COLORS, title="Brand mix per segment (%)")
        fig.update_yaxes(title="", categoryorder="array", categoryarray=CUSTOMER_SEGMENTS[::-1])
        fig.update_traces(marker_line_color="#fcfcfb", marker_line_width=2)
        c2.plotly_chart(style(fig, 420), width="stretch")
        asp = df.groupby("customer_segment")[C.ASPECTS].mean().reindex(CUSTOMER_SEGMENTS)
        asp.columns = [C.ASPECT_LABELS[a] for a in asp.columns]
        st.markdown("**Segment profile table**")
        prof2 = prof.join(asp)
        st.dataframe(prof2.style.format({"Avg_price": "${:,.0f}", "Avg_rating": "{:.2f}", "Positive": "{:.1%}",
                                         "Helpful": "{:.2f}", **{c: "{:.2f}" for c in asp.columns}}),
                     width="stretch")

    with tab2:
        pm = meta["product_segmentation"]["metrics"]
        st.caption("K-Means (k = 4) on each model's average price and observed low / high price "
                   f"(2nd / 98th percentile) · silhouette {pm['silhouette']:.3f}. Rating features vary by "
                   "<2.1% across models, so they would only add noise to the clustering.")
        seg = products.groupby("product_segment").agg(
            Models=("model", lambda s: ", ".join(s)), Avg_price=("avg_price_usd", "mean"),
            Reviews=("n_reviews", "sum"), Avg_rating=("avg_rating", "mean"), Positive=("positive_share", "mean"),
            Value=("avg_value_score", "mean")).reindex(PRODUCT_SEGMENTS)
        st.dataframe(seg.style.format({"Avg_price": "${:,.0f}", "Reviews": "{:,}", "Avg_rating": "{:.3f}",
                                       "Positive": "{:.1%}", "Value": "{:.3f}"}), width="stretch")
        p = products.sort_values("avg_price_usd")
        fig = go.Figure()
        for sname in PRODUCT_SEGMENTS:
            d = p[p.product_segment == sname]
            fig.add_trace(go.Scatter(x=d.avg_price_usd, y=d.model, mode="markers", name=sname,
                                     marker=dict(size=12, color=PRODUCT_SEGMENT_COLORS[sname]),
                                     error_x=dict(type="data", symmetric=False,
                                                  array=d.max_price_usd - d.avg_price_usd,
                                                  arrayminus=d.avg_price_usd - d.min_price_usd,
                                                  color=PRODUCT_SEGMENT_COLORS[sname], thickness=2, width=0)))
        fig.update_layout(title="Average price and observed price range per model", xaxis_title="USD")
        st.plotly_chart(style(fig, 600), width="stretch")

    with tab3:
        st.caption("Uses the saved K-Means pipeline (same scaler + centroids as training).")
        with st.form("seg"):
            c1, c2 = st.columns(2)
            price = c1.number_input("Price paid (USD)", 50.0, 3000.0, 650.0, step=10.0)
            rating = c2.slider("Overall rating", 1, 5, 4)
            cols = st.columns(5)
            asp = [cols[i].slider(C.ASPECT_LABELS[a], 1, 5, 3) for i, a in enumerate(C.ASPECTS)]
            ok = st.form_submit_button("Assign segment", type="primary", width="stretch")
        if ok:
            X = pd.DataFrame([{"price_usd": price, "rating": rating, "aspect_avg": float(np.mean(asp))}])
            label = int(cust_pipe.predict(X)[0])
            seg = cm["cluster_names"][str(label)]
            dist = cust_pipe.transform(X)[0]
            st.markdown(f"### {seg}")
            st.write(SEGMENT_TEXT[seg])
            lo, hi = cm["feature_ranges"]["price_usd"]
            if not lo <= price <= hi:
                st.warning(f"Price {usd(price)} is outside the training range ({usd(lo)} – {usd(hi)}); the "
                           "assignment extrapolates.")
            d = pd.DataFrame({"Segment": [cm["cluster_names"][str(i)] for i in range(len(dist))],
                              "Distance": dist}).sort_values("Distance")
            fig = px.bar(d, x="Distance", y="Segment", orientation="h", color="Segment",
                         color_discrete_map=SEGMENT_COLORS, title="Distance to each segment centre (scaled)")
            fig.update_layout(showlegend=False)
            st.plotly_chart(style(fig, 280), width="stretch")


def page_dashboard(df, products):
    st.title("📊 Market Dashboard")
    f1, f2, f3, f4 = st.columns([1.6, 1.6, 1.2, 1.4])
    brands = f1.multiselect("Brand", C.BRANDS, default=C.BRANDS)
    countries = f2.multiselect("Country", sorted(df.country.unique()), default=sorted(df.country.unique()))
    sources = f3.multiselect("Source", sorted(df.source.unique()), default=sorted(df.source.unique()))
    dr = f4.date_input("Review date", (df.review_date.min().date(), df.review_date.max().date()))
    d = df[df.brand.isin(brands) & df.country.isin(countries) & df.source.isin(sources)]
    if isinstance(dr, tuple) and len(dr) == 2:
        d = d[(d.review_date >= pd.Timestamp(dr[0])) & (d.review_date <= pd.Timestamp(dr[1]))]
    if d.empty:
        st.warning("No reviews match these filters.")
        return

    k = st.columns(5)
    k[0].metric("Reviews", f"{len(d):,}")
    k[1].metric("Average rating", f"{d.rating.mean():.2f} ★")
    k[2].metric("Positive reviews", f"{d.is_positive.mean():.1%}")
    k[3].metric("Average price", usd(d.price_usd.mean()))
    k[4].metric("Verified purchases", f"{d.verified_purchase.mean():.1%}")

    c1, c2 = st.columns(2)
    b = d.groupby("brand").agg(reviews=("rating", "size"), price=("price_usd", "mean")).reindex(
        [x for x in C.BRANDS if x in d.brand.unique()]).reset_index()
    fig = px.bar(b, x="brand", y="reviews", color="brand", color_discrete_map=C.BRAND_COLORS,
                 title="Reviews by brand", text_auto=",")
    fig.update_layout(showlegend=False)
    c1.plotly_chart(style(fig), width="stretch")
    fig = px.box(d.sample(min(len(d), 15000), random_state=0), x="brand", y="price_usd", color="brand",
                 color_discrete_map=C.BRAND_COLORS, points=False, title="Price paid by brand (USD)",
                 category_orders={"brand": C.BRANDS})
    fig.update_layout(showlegend=False)
    c2.plotly_chart(style(fig), width="stretch")

    c1, c2 = st.columns(2)
    s = (pd.crosstab(d.brand, d.sentiment, normalize="index") * 100).reset_index().melt(
        "brand", var_name="Sentiment", value_name="Share %")
    fig = px.bar(s, x="Share %", y="brand", color="Sentiment", orientation="h",
                 color_discrete_map=C.SENTIMENT_COLORS, title="Sentiment mix by brand (%)",
                 category_orders={"Sentiment": ["Negative", "Neutral", "Positive"], "brand": C.BRANDS})
    fig.update_traces(marker_line_color="#fcfcfb", marker_line_width=2)
    fig.update_yaxes(title="")
    c1.plotly_chart(style(fig), width="stretch")
    m = d.groupby("review_month").agg(reviews=("rating", "size"), rating=("rating", "mean")).reset_index()
    fig = px.line(m, x="review_month", y="reviews", markers=True, title="Reviews per month",
                  color_discrete_sequence=[C.CATEGORICAL[0]])
    fig.update_xaxes(title="")
    c2.plotly_chart(style(fig), width="stretch")

    c1, c2 = st.columns(2)
    ct = pd.crosstab(d.brand, d.country)
    fig = px.imshow(ct, text_auto=",", color_continuous_scale=C.SEQ_BLUE, aspect="auto",
                    title="Reviews: brand × country")
    fig.update_coloraxes(showscale=False)
    c1.plotly_chart(style(fig, 380), width="stretch")
    asp = d.groupby("brand")[C.ASPECTS].mean()
    asp.columns = [C.ASPECT_LABELS[a] for a in asp.columns]
    fig = px.imshow(asp.round(2), text_auto=".2f", color_continuous_scale=C.SEQ_BLUE, aspect="auto",
                    title="Average aspect rating by brand")
    fig.update_coloraxes(showscale=False)
    c2.plotly_chart(style(fig, 380), width="stretch")

    p = d.groupby(["model", "brand"]).agg(Reviews=("rating", "size"), Avg_price=("price_usd", "mean"),
                                          Rating=("rating", "mean"), Positive=("is_positive", "mean")).reset_index()
    p = p[p.Reviews >= 30]
    prior, mm = d.rating.mean(), 500
    p["Bayes rating"] = (p.Reviews * p.Rating + mm * prior) / (p.Reviews + mm)
    fmt = {"Avg_price": "${:,.0f}", "Rating": "{:.3f}", "Positive": "{:.1%}", "Bayes rating": "{:.3f}"}
    c1, c2 = st.columns(2)
    c1.markdown("**Top-rated products** (Bayesian average, ≥30 reviews)")
    c1.dataframe(p.nlargest(5, "Bayes rating").style.format(fmt), hide_index=True, width="stretch")
    c2.markdown("**Lowest-rated products**")
    c2.dataframe(p.nsmallest(5, "Bayes rating").style.format(fmt), hide_index=True, width="stretch")


def page_insights(df, products, meta):
    st.title("💡 Key Insights & Recommendations")
    st.caption(f"Computed live from {len(df):,} cleaned reviews of {products.shape[0]} phone models · 7 brands · "
               f"8 countries · {df.review_date.min():%b %Y} – {df.review_date.max():%b %Y}.")
    from scipy import stats

    k = st.columns(4)
    k[0].metric("Reviews", f"{len(df):,}")
    k[1].metric("Average rating", f"{df.rating.mean():.2f} ★")
    k[2].metric("Positive reviews", f"{df.is_positive.mean():.1%}")
    k[3].metric("Dissatisfied customers", f"{df.customer_segment.str.contains('Dissatisfied').mean():.1%}")

    # 1. segments
    st.divider()
    st.subheader("1 · The market splits cleanly into four price tiers")
    seg = products.groupby("product_segment").agg(models=("model", "size"), reviews=("n_reviews", "sum"),
                                                  price=("avg_price_usd", "mean"),
                                                  value=("avg_value_score", "mean")).reindex(PRODUCT_SEGMENTS)
    seg["share"] = seg.reviews / seg.reviews.sum() * 100
    c1, c2 = st.columns([3, 2])
    fig = px.bar(seg.reset_index(), x="product_segment", y="share", color="product_segment", text_auto=".1f",
                 color_discrete_map=PRODUCT_SEGMENT_COLORS, title="Share of all reviews by product segment (%)",
                 labels={"product_segment": "", "share": "% of reviews"})
    fig.update_layout(showlegend=False)
    c1.plotly_chart(style(fig, 320), width="stretch")
    brands_by = products.groupby("product_segment").brand.agg(lambda s: ", ".join(sorted(s.unique())))
    c2.markdown("\n".join(f"- **{s}** (avg {usd(seg.loc[s, 'price'])}): {brands_by[s]}" for s in PRODUCT_SEGMENTS))
    c2.markdown(f"- Budget phones generate **{seg.loc['Budget', 'share']:.0f}%** of reviews - the volume "
                f"segment - while Premium (Apple) is **{seg.loc['Premium', 'share']:.0f}%**.")

    # 2. price vs performance
    st.subheader("2 · Paying more does not buy more satisfaction")
    r, p = stats.pearsonr(df.price_usd, df.rating)
    bins = pd.cut(df.price_usd, np.arange(150, 1551, 100))
    g = df.groupby(bins, observed=True).agg(Rating=("rating", "mean"), Aspects=("aspect_avg", "mean")).reset_index()
    g["Price"] = g.price_usd.apply(lambda i: i.mid)
    c1, c2 = st.columns([3, 2])
    fig = px.line(g.melt("Price", ["Rating", "Aspects"], var_name="Measure", value_name="Average"), x="Price",
                  y="Average", color="Measure", markers=True, range_y=[1, 5],
                  color_discrete_sequence=C.CATEGORICAL[:2], title="Average rating by price paid ($100 bins)")
    c1.plotly_chart(style(fig, 320), width="stretch")
    val = products.groupby("product_segment").avg_value_score.mean()
    c2.markdown(
        f"- Price vs rating correlation **r = {r:.3f}** (p = {p:.2f}) - no relationship.\n"
        f"- Premium phones (~{usd(seg.loc['Premium', 'price'])}) score the same as Budget phones "
        f"(~{usd(seg.loc['Budget', 'price'])}) on battery, camera, performance, design and display.\n"
        f"- Value for money (rating points per $100): Budget **{val['Budget']:.2f}** vs Premium "
        f"**{val['Premium']:.2f}** - budget phones deliver **{val['Budget'] / val['Premium']:.1f}×** the "
        "satisfaction per dollar.")

    # 3. high / low performers
    st.subheader("3 · High- and low-performing products")
    f, pa = stats.f_oneway(*[x.rating.to_numpy() for _, x in df.groupby("model")])
    rk = products.sort_values("bayes_rating", ascending=False)
    c1, c2 = st.columns([3, 2])
    fig = px.bar(rk, x="bayes_rating", y="model", color="brand", orientation="h", color_discrete_map=C.BRAND_COLORS,
                 title="Bayesian-average rating per model (axis zoomed)",
                 range_x=[rk.bayes_rating.min() - 0.02, rk.bayes_rating.max() + 0.02])
    fig.update_yaxes(categoryorder="total ascending", title="")
    c1.plotly_chart(style(fig, 520), width="stretch")
    c2.markdown(
        f"- **Top 3:** {', '.join(rk.model.head(3))}.\n"
        f"- **Bottom 3:** {', '.join(rk.model.tail(3)[::-1])}.\n"
        f"- But the whole spread is only **{rk.bayes_rating.max() - rk.bayes_rating.min():.2f}★**, and a one-way "
        f"ANOVA across the 22 models gives **p = {pa:.2f}**: the differences are **not statistically "
        "significant**. No model is a true outlier - every product has ~55% positive and ~20% negative reviews.\n"
        f"- Biggest *volume* winners: {', '.join(products.nlargest(2, 'n_reviews').model)} (Realme) - "
        "the most-reviewed models are budget phones.")

    # 4. customer segments
    st.subheader("4 · Customer segments: satisfaction, not brand, separates buyers")
    cs = df.groupby("customer_segment").agg(share=("rating", "size"), rating=("rating", "mean"),
                                            helpful=("helpful_votes", "mean"),
                                            positive=("is_positive", "mean")).reindex(CUSTOMER_SEGMENTS)
    cs["share"] = cs.share / len(df) * 100
    c1, c2 = st.columns([3, 2])
    fig = px.bar(cs.reset_index(), x="customer_segment", y="share", color="customer_segment", text_auto=".1f",
                 color_discrete_map=SEGMENT_COLORS, title="Customer segment sizes (% of reviews)",
                 labels={"customer_segment": "", "share": "%"})
    fig.update_layout(showlegend=False)
    c1.plotly_chart(style(fig, 320), width="stretch")
    rh, _ = stats.pearsonr(df.helpful_votes, df.rating)
    c2.markdown(
        f"- **{cs.loc[[s for s in CUSTOMER_SEGMENTS if 'Dissatisfied' in s], 'share'].sum():.0f}%** of buyers are "
        "in a *Dissatisfied* segment (≈2.1★) - in every price tier.\n"
        f"- **Premium · Dissatisfied** ({cs.loc['Premium · Dissatisfied', 'share']:.0f}%) paid ~$1,040 for a "
        "≈2.1★ experience - the most valuable group to win back.\n"
        f"- Satisfied reviews collect **{cs.loc['Premium · Satisfied', 'helpful']:.1f}** helpful votes vs "
        f"**{cs.loc['Premium · Dissatisfied', 'helpful']:.1f}** for dissatisfied ones (r = {rh:.2f} between "
        "helpful votes and rating): shoppers find positive reviews more useful.")

    # 5. preferences
    st.subheader("5 · Customer preference patterns")
    chi_c = stats.chi2_contingency(pd.crosstab(df.country, df.brand))[1]
    chi_a = stats.chi2_contingency(pd.crosstab(df.age_group, df.brand))[1]
    aspect_r = df[C.ASPECTS].corrwith(df.rating).rename(index=C.ASPECT_LABELS).sort_values()
    c1, c2 = st.columns([3, 2])
    fig = px.bar(aspect_r.rename("r").rename_axis("Aspect").reset_index(), x="r", y="Aspect", orientation="h",
                 text_auto=".3f", range_x=[0, 1], labels={"r": "Correlation with overall rating", "Aspect": ""},
                 title="Which aspect drives the overall rating?", color_discrete_sequence=[C.SEQ_BLUE[4]])
    c1.plotly_chart(style(fig, 300), width="stretch")
    c2.markdown(
        f"- All five aspects drive the overall rating **equally** (r ≈ {aspect_r.mean():.2f} each): buyers "
        "judge the whole package, not one spec.\n"
        f"- Brand choice is the same across countries (χ² p = {chi_c:.2f}) and age groups (p = {chi_a:.2f}) - "
        "each brand holds ~14% everywhere.\n"
        "- Review source and verified status don't change ratings either, so the reviews can be trusted as one "
        "pool.")

    prem = df[df.customer_segment.str.startswith("Premium")]
    prem_dis = (prem.customer_segment == "Premium · Dissatisfied").mean()
    st.divider()
    st.subheader("Recommendations")
    st.markdown(
        f"1. **Lead with value in the Budget tier.** It carries ~{seg.loc['Budget', 'share']:.0f}% of review "
        f"volume and gives {val['Budget'] / val['Premium']:.1f}× more satisfaction per dollar; stock depth and "
        "promotions here move the most customers.\n"
        f"2. **Fix the premium experience.** {prem_dis:.0%} of premium buyers rate their ~$1,000+ phone ≈2★. "
        "Target the "
        "*Premium · Dissatisfied* segment with service recovery, extended warranty and trade-in offers.\n"
        "3. **Don't price on quality claims.** Ratings don't rise with price, so premium pricing must be justified "
        "by brand, ecosystem or service - not by spec scores.\n"
        "4. **Recommend by price tier first, then by the aspects a shopper cares about** - exactly what the "
        "Recommend page does; its similar-phone results stay inside the same segment 100% of the time.\n"
        "5. **Surface positive, highly-voted reviews** on product pages - they are the ones shoppers rate as "
        "helpful.\n"
        "6. **One global playbook works.** Preferences don't differ by country or age group, so the same "
        "assortment and messaging can be used in all eight markets.")


def page_models(df, meta):
    st.title("🧪 Model Performance & Validation")
    cm, pm = meta["customer_segmentation"], meta["product_segmentation"]
    st.subheader("Customer segmentation - K-Means (k = 4)")
    k = st.columns(4)
    k[0].metric("Silhouette (10k sample)", f"{cm['metrics']['silhouette_sample10k']:.3f}")
    k[1].metric("Davies-Bouldin", f"{cm['metrics']['davies_bouldin']:.3f}")
    k[2].metric("Calinski-Harabasz", f"{cm['metrics']['calinski_harabasz']:,.0f}")
    alt = cm["alternative_with_encoded_categoricals"]
    k[3].metric("Silhouette with one-hot brand+country", f"{alt['silhouette_sample10k']:.3f}",
                delta=f"{alt['silhouette_sample10k'] - cm['metrics']['silhouette_sample10k']:.3f}")
    ks = pd.read_csv(C.REPORTS_DIR / "customer_k_selection.csv")
    kp = pd.read_csv(C.REPORTS_DIR / "product_k_selection.csv")
    c1, c2, c3 = st.columns(3)
    for col, (y, t) in zip([c1, c2, c3], [("inertia", "Elbow (inertia)"), ("silhouette", "Silhouette ↑"),
                                          ("davies_bouldin", "Davies-Bouldin ↓")]):
        fig = px.line(ks, x="k", y=y, markers=True, title=t, color_discrete_sequence=[C.CATEGORICAL[0]])
        fig.add_vline(x=4, line_dash="dash", line_color=C.CATEGORICAL[1])
        col.plotly_chart(style(fig, 260), width="stretch")
    st.markdown(
        "- **Why k = 4:** the elbow bends at 4, and 4 is the smallest k that separates *both* price tier and "
        "satisfaction. k = 2 splits on satisfaction only (both clusters average ~$690); k = 5 merely splits the "
        "budget side into a third 'neutral' group. Silhouette is flat between k = 4 and 10 (0.34–0.36).\n"
        "- **Feature selection check:** adding one-hot brand + country + age + engagement drops silhouette to "
        f"{alt['silhouette_sample10k']:.3f} - categorical dummies fragment the clusters without adding behavioural "
        "signal (brand and country have no relationship with satisfaction).")

    st.subheader("Product segmentation - K-Means (k = 4) on 22 models")
    k = st.columns(3)
    k[0].metric("Silhouette", f"{pm['metrics']['silhouette']:.3f}")
    k[1].metric("Davies-Bouldin", f"{pm['metrics']['davies_bouldin']:.3f}")
    k[2].metric("Calinski-Harabasz", f"{pm['metrics']['calinski_harabasz']:,.1f}")
    st.dataframe(kp.style.format({"inertia": "{:.3f}", "silhouette": "{:.3f}", "davies_bouldin": "{:.3f}",
                                  "calinski_harabasz": "{:,.1f}"}), hide_index=True, width="stretch")
    st.caption("Higher k scores better only because each brand sells all its models in one shared price band; "
               "k = 7 is simply one cluster per brand. k = 4 gives the business tiers Budget / Mid-range / "
               "Upper mid-range / Premium.")

    st.subheader("Recommendation relevance")
    ev = pd.read_csv(C.REPORTS_DIR / "recommender_content_evaluation.csv")
    st.dataframe(ev.style.format({c: "{:.2f}" for c in ev.columns if c != "metric"}), hide_index=True,
                 width="stretch")
    st.caption("For every one of the 22 phones, its top-3 'similar' phones are checked: share in the same product "
               "segment, average absolute price gap, rating gap, other-brand share and catalog coverage - against "
               "random recommendations.")
    cf = meta["recommender"]["cf_evaluation"]
    c1, c2 = st.columns([3, 2])
    d = pd.DataFrame({"Predictor": ["Global mean", "Item mean", "Group mean", "Item-item CF"],
                      "RMSE": [cf["rmse_global"], cf["rmse_item"], cf["rmse_group"], cf["rmse_cf"]]})
    fig = px.bar(d, x="Predictor", y="RMSE", text_auto=".3f", title="Held-out (group, phone) rating RMSE",
                 color_discrete_sequence=[C.SEQ_BLUE[4]])
    c1.plotly_chart(style(fig, 300), width="stretch")
    c2.markdown(
        f"- 20% of reviews held out; {cf['test_cells']} (customer group, phone) cells predicted.\n"
        f"- Item-item CF RMSE **{cf['rmse_cf']:.3f}** vs item-mean **{cf['rmse_item']:.3f}**; precision@3 "
        f"{cf['precision@3_cf']:.2f} vs random {cf['precision@3_random_expected']:.2f}.\n"
        "- CF does **not** beat the baselines here because ratings do not vary by country or age group (χ² and "
        "ANOVA tests). It is kept as a 10% component of the hybrid score and would add value on data with real "
        "taste differences.")
    st.markdown("**Preference-based recommendations - example profiles**")
    st.dataframe(pd.read_csv(C.REPORTS_DIR / "recommender_persona_checks.csv"), hide_index=True, width="stretch")

    st.subheader("Data-cleaning validation")
    ic = meta["imputation_check"]
    k = st.columns(4)
    k[0].metric("Rating imputation accuracy", f"{ic['rating_rule_accuracy']:.1%}",
                delta=f"+{ic['rating_rule_accuracy'] - ic['rating_global_mode_accuracy']:.1%} vs mode")
    k[1].metric("Rating within ±1 star", f"{ic['rating_rule_within_1_star']:.1%}")
    k[2].metric("Sentiment imputation accuracy", f"{ic['sentiment_rule_accuracy']:.1%}",
                delta=f"+{ic['sentiment_rule_accuracy'] - ic['sentiment_global_mode_accuracy']:.1%} vs mode")
    k[3].metric("Prices recovered exactly", "4,635", help="price_local = price_usd × exchange rate (100% of rows)")
    for img in ["seg_01_k_selection_customer.png", "seg_03_customer_clusters.png", "rec_01_similarity_heatmap.png"]:
        p = C.FIGURES_DIR / img
        if p.exists():
            st.image(str(p), width="stretch")


# --------------------------------------------------------------------------- main
def main():
    try:
        cust_pipe, meta = load_models()
        df, products = load_data()
    except FileNotFoundError:
        st.error("Models / data not found. Run `python run_pipeline.py` first.")
        st.stop()
    rec = load_recommender()
    page = st.sidebar.radio("Navigate", ["🔎 Recommend", "🧩 Segmentation", "📊 Dashboard", "💡 Insights",
                                         "🧪 Model performance"])
    st.sidebar.markdown("---")
    st.sidebar.caption(f"{len(df):,} reviews · {products.shape[0]} phone models · 7 brands · 8 countries")
    if page.endswith("Recommend"):
        page_recommend(df, products, rec, meta)
    elif page.endswith("Segmentation"):
        page_segments(df, products, cust_pipe, meta)
    elif page.endswith("Dashboard"):
        page_dashboard(df, products)
    elif page.endswith("Insights"):
        page_insights(df, products, meta)
    else:
        page_models(df, meta)


main()
