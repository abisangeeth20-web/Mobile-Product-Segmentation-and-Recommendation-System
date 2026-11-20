# Final Project Documentation
## Mobile Product Segmentation and Recommendation System Using Python and Machine Learning

| | |
|---|---|
| **Domain** | E-Commerce Analytics & Intelligent Product Recommendation Systems |
| **Dataset** | Global Mobile Reviews: `Mobile Reviews Sentiment null.csv`, 50,000 rows × 22 columns |
| **Repository** | https://github.com/abisangeeth20-web/Mobile-Product-Segmentation-and-Recommendation-System |
| **Live app** | https://mobile-segmentation-recommender.streamlit.app |
| **Stack** | Python, Pandas, NumPy, scikit-learn, SciPy, Matplotlib, Seaborn, Plotly, Streamlit |

---

## 1. Problem statement

Smartphone data covers specifications, pricing, ratings and user information, but in raw form it is unorganised and hard to analyse. This project:

* converts the data into a clean, structured form;
* segments **customers and products** with K-Means to identify premium, mid-range and budget segments;
* builds a **similarity-based recommendation system**, using structured data only, that suggests phones based on a selected phone or on a user's preferences.

## 2. Objectives and how each was met

| Lifecycle step | Implementation | Where |
|---|---|---|
| Data collection | CSV loaded into a Pandas DataFrame; structure, columns and data types profiled | `src/preprocessing.py` → `load_raw()`, notebook §1 |
| Data cleaning & preprocessing | Missing values, duplicates, type conversion, validation | `src/preprocessing.py` → `clean()` |
| EDA | Distributions, top/low products, price vs ratings vs specs, trends, correlations, statistical tests | `src/eda.py`, `reports/EDA_Report.md` |
| Feature engineering & scaling | 10 engineered features, label and one-hot encoding, `StandardScaler` | `src/features.py`, `src/segmentation.py` |
| Clustering | K-Means at customer and product level, k = 4 chosen with 4 validity scores | `src/segmentation.py`, `reports/Cluster_Analysis_Report.md` |
| Cluster analysis & insights | Segment profiles, naming, business actions | `reports/Cluster_Analysis_Report.md`, app → Insights |
| Recommendation system | Cosine / Euclidean similarity, preference-based scoring, item-item collaborative filtering, validation | `src/recommender.py`, `reports/Recommendation_System_Report.md` |
| Streamlit application | 5 pages: Recommend, Segmentation, Dashboard, Insights, Model performance | `app.py` (deployed) |

## 3. Data cleaning and preprocessing

| Problem | Count | Treatment | Evidence |
|---|---|---|---|
| `price_usd` missing | 2,450 | Recovered **exactly** as `price_local / exchange_rate` | The identity holds on 100% of complete rows (max error $0.005) |
| `price_local` missing | 2,431 | Recovered **exactly** as `price_usd × exchange_rate` | Same identity |
| Both prices missing | 123 | Model median price | Every model sells within a fixed price band |
| `rating` missing | 2,453 | Most frequent rating for the same aspect average and sentiment | 77.8% exact (99.97% within ±1★) on held-out rows, vs 28.1% for the overall mode |
| `sentiment` missing | 2,445 | Most frequent sentiment for the same rating and aspect average | 74.5% on held-out rows, vs 56.1% for the overall mode |
| `source` missing | 2,448 | `Unknown` category | Source has no relationship with any other column |
| Duplicates | 0 | Checked exact rows, repeated IDs, and the same content under a new ID | — |
| Type conversion | — | Date → datetime; `₹27996.73` / `$1,010.68 ` text → numeric; bool; int | — |
| Validation | 0 conflicts | brand↔model, country↔currency↔rate, ratings 1–5, age 18–65 | — |

**Output:** `data/processed/cleaned_mobile_reviews.csv` with 0 nulls. Imputed rows are flagged in `price_imputed`, `rating_imputed` and `sentiment_imputed`.

## 4. Feature engineering, encoding and scaling

| Feature | Definition | Purpose |
|---|---|---|
| `aspect_avg` | Mean of battery, camera, performance, design and display ratings | Overall specification / performance score |
| `rating_gap` | rating − aspect_avg | Overall rating vs aspect scores |
| `sentiment_score`, `is_positive` | −1 / 0 / 1; positive flag | Numeric sentiment |
| `engagement_score` | helpful votes × 1.5 if verified purchase | Engagement score (named in the spec) |
| `value_score` | rating per $100 paid | Price vs performance |
| `price_band`, `age_group`, year, month | Bins and date parts | EDA and preference analysis |
| Encoding | Label codes for brand, model, country and source; one-hot brand and country | Encoding of the categorical variables the spec names |
| Product catalog | One row per model: price statistics, ratings, aspect means, positive share, Bayesian rating | Item table for product clustering and recommendations |
| Scaling | `StandardScaler` inside every saved K-Means pipeline | Equal feature weight; the app reuses the same scaling |

**Output:** `data/processed/feature_engineered_mobile_reviews.csv` and `data/processed/product_catalog.csv`.

## 5. Exploratory data analysis: key findings

1. **Brands occupy distinct price tiers.** Apple averages $1,104, Samsung $899, Google $805, OnePlus $673, Motorola $506, Xiaomi $450 and Realme $393 (ANOVA p < 0.001).
2. **Price does not raise satisfaction.** Price vs rating r = 0.001 (p = 0.74), and every aspect rating is flat across price bands.
3. **Top and low products** are ranked by Bayesian rating. Top: iPhone 14, iPhone 13, Galaxy Note 20. Bottom: Galaxy S24, iPhone 15 Pro, Edge 50. The spread is only 0.08★ and not significant (ANOVA p = 0.78).
4. **All five aspects drive the overall rating equally** (r ≈ 0.76 each).
5. **Helpful votes favour positive reviews** (r = 0.46).
6. **Preferences are global.** Brand share and sentiment do not vary by country, age group, source or verified status (all p > 0.3).

The full analysis with 10 figures and an 11-test statistical table is in `reports/EDA_Report.md`.

## 6. Clustering (K-Means)

### Choosing the number of clusters
k = 2 to 10 were compared using inertia (elbow), silhouette, Davies-Bouldin and Calinski-Harabasz.

* **Customer level:** k = 4 is the smallest k that separates both price tier and satisfaction. k = 2 ignores price, and k = 5 or more only splits rating levels further.
* **Product level:** k = 4 matches the market tiers. A larger k just isolates single brands.

### Feature selection check
Adding one-hot brand and country to the customer clustering lowered silhouette from 0.348 to 0.135, so behavioural features were kept. Rating features vary by less than 2.1% across models (price varies by 34%), so product clusters use price structure.

### Customer segments (50,000 reviews; silhouette 0.348)

| Segment | Share | Avg price | Avg rating | Positive | Business action |
|---|---|---|---|---|---|
| Premium · Satisfied | 17.5% | $1,029 | 4.21 | 90% | Referral, upsell, early-access programmes |
| Premium · Dissatisfied | 18.8% | $1,037 | 2.14 | 25% | **Priority:** service recovery, warranty, trade-in |
| Budget/Mid · Satisfied | 30.0% | $492 | 4.21 | 90% | Value bundles, trade-up path |
| Budget/Mid · Dissatisfied | 33.8% | $496 | 2.14 | 25% | Quality fixes, clearer expectation-setting |

### Product segments (22 models; silhouette 0.791)

| Segment | Brands | Avg price | Share of reviews | Value (★ per $100) |
|---|---|---|---|---|
| Budget | Realme, Xiaomi, Motorola | $457 | 43% | 0.78 |
| Mid-range | OnePlus, Google | $739 | 29% | 0.46 |
| Upper mid-range | Samsung | $899 | 14% | 0.39 |
| Premium | Apple | $1,104 | 14% | 0.30 |

The clusters are visualised with scatter plots, 2-D PCA, profile bars, brand mix and price-range charts. See `reports/figures/seg_*.png`.

## 7. Recommendation system

| Component | Method | Validation result |
|---|---|---|
| **Similar phones** | Product vectors (price structure + quality features scaled by review-level variation). Similarity is cosine or `1/(1+Euclidean distance)`. | Euclidean: **100%** of top-3 recommendations in the same segment, mean price gap $23. Cosine: 80%, $57. Random: 24%, $274. **Euclidean is the default.** |
| **Find my phone** | Budget (in local currency), aspect priorities, country and age group. Score = 0.50 priority aspects + 0.10 CF + 0.20 positive share + 0.20 budget fit | 100% of recommendations within budget for all 4 test profiles. With specific priorities, recommended phones score above the catalog average on those aspects. |
| **Collaborative filtering basics** | Item-item adjusted cosine on a 40 customer-group (country × age group) × 22 phone rating matrix | Held-out RMSE 0.355 vs 0.342 for the item mean. CF does not beat the baseline because ratings do not differ by group, so it carries a small 10% weight. |

Every recommendation comes with a reason, for example: *"same Mid-range segment · avg price +$8 · strongest aspect: Display"*.

## 8. Streamlit application

**Live:** https://mobile-segmentation-recommender.streamlit.app

| Page | Features |
|---|---|
| 🔎 Recommend | *Similar to a phone*: choose the metric and whether to show other brands only; similarity chart; price-position chart; CF view. *Find my phone*: budget in local currency, aspect sliders, score breakdown. |
| 🧩 Segmentation | Customer segment cards, scatter plot, brand mix, profile table; product tiers with price ranges; *Which segment am I?* using the saved K-Means model, with a warning when the price is outside the training range |
| 📊 Dashboard | Brand, country, source and date filters; KPIs; brand, price, sentiment, trend and heatmap charts; top and bottom products |
| 💡 Insights | Findings computed live (segments, price vs performance, high and low performers, customer segments, preference patterns) and 6 business recommendations |
| 🧪 Model performance | k-selection curves, clustering metrics, encoding comparison, recommender relevance, CF evaluation, test profiles, imputation accuracy |

## 9. Business insights and recommendations

1. **Lead with value in the Budget tier.** It has 43% of the volume and gives 2.6× more satisfaction per dollar.
2. **Fix the premium experience.** About half of premium buyers rate their roughly $1,000 phone ≈2★. Target the *Premium · Dissatisfied* segment.
3. **Don't justify premium pricing with spec quality.** Ratings do not rise with price, so brand, ecosystem and service must carry the premium.
4. **Recommend by price tier first, then by the aspects the user cares about.** This is what the app does.
5. **Show positive, highly-voted reviews** on product pages, since shoppers find them most helpful.
6. **Use one global playbook.** Preferences do not differ by country or age group.

## 10. Project deliverables checklist

| Spec deliverable | File(s) |
|---|---|
| CSV dataset (raw data file) | `Mobile Reviews Sentiment null.csv` |
| Cleaned and preprocessed dataset | `data/processed/cleaned_mobile_reviews.csv`, `feature_engineered_mobile_reviews.csv`, `product_catalog.csv` |
| Python script for preprocessing & feature engineering | `src/preprocessing.py`, `src/features.py`, `run_pipeline.py` |
| Implementation of clustering model (K-Means) | `src/segmentation.py`, `models/kmeans_customer_segments.joblib`, `models/kmeans_product_segments.joblib` |
| Cluster analysis report with insights (4 segments) | `reports/Cluster_Analysis_Report.md` |
| Recommendation system implementation (similarity-based) | `src/recommender.py`, `models/similarity_*.csv` |
| Python script for recommendation logic | `src/recommender.py` |
| Data visualisation (Plotly / Seaborn / Matplotlib) | `reports/figures/*.png` (18 figures) and Plotly charts in the app |
| Interactive Streamlit application | `app.py` + live link |
| EDA report with key findings and patterns | `reports/EDA_Report.md` |
| Final project documentation file | `PROJECT_DOCUMENTATION.md` (this file) + `README.md` |
| End-to-end walk-through | `notebooks/Mobile_Segmentation_Recommendation.ipynb` (executed) |

## 11. Evaluation metrics coverage

| Evaluation metric | Section of this document |
|---|---|
| Proper data preprocessing (missing values & duplicates) | §3 |
| Accurate CSV-to-DataFrame conversion | §2, §3 |
| Effective EDA | §5 |
| Correct feature selection, encoding and scaling | §4, §6 (feature selection check) |
| Proper implementation of K-Means | §6 |
| Appropriate selection of the number of clusters | §6 (choosing k) |
| Clear cluster interpretation and business insights | §6, §9 |
| Effective visualisation of clusters and patterns | §6, `reports/figures` |
| Correct similarity-based recommendation system | §7 |
| Relevance and accuracy of recommended products | §7 (validation) |
| Integration of clustering and recommendation logic | §7: segments label recommendations, and same-segment rate is the relevance metric |
| Visualisation quality (Plotly / Seaborn / Matplotlib) | §8, figures |
| Streamlit functionality and UI | §8 |
| Clean, modular Python code | `src/` package: 8 modules shared by the pipeline, notebook and app |
| Insight generation and interpretation | §5, §6, §9 |

## 12. How to run

```bash
pip install -r requirements-dev.txt
python run_pipeline.py        # cleans data, engineers features, clusters, builds recommender, writes reports (~1 min)
streamlit run app.py          # http://localhost:8501
```

## 13. Limitations and future work

* **The dataset appears synthetic.** Ratings are independent of price, brand, country and age. Because of this:
  * product quality differences are not significant;
  * customer segments separate on price and satisfaction only;
  * collaborative filtering has no group-level taste to learn.

  These limits are measured and reported, not hidden.
* **No repeat-user history.** 93% of names appear only once, so true user-item collaborative filtering is not possible.
* **Future work:**
  * Add product specifications such as RAM, storage, battery mAh and camera MP to give richer item vectors.
  * Use real user IDs to enable user-based collaborative filtering.
  * Run A/B tests on recommendation click-through rates.

## 14. Conclusion

The project delivers a complete, reproducible pipeline (`python run_pipeline.py`) that:

* cleans 50,000 reviews and recovers 4,635 prices exactly;
* segments the market into 4 customer segments and 4 product tiers with K-Means;
* recommends phones with a validated similarity model (100% same-segment relevance);
* serves everything in a deployed Streamlit app with live insights.
