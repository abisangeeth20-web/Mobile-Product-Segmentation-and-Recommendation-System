# Mobile Product Segmentation & Recommendation System

An end-to-end Python and machine-learning pipeline built on the **Global Mobile Reviews** dataset (50,000 reviews · 7 brands · 22 phone models · 8 countries). It covers:

* CSV ingestion and cleaning
* Feature engineering, encoding and scaling
* EDA with statistical tests
* **K-Means segmentation** of customers and products
* A **similarity-based recommendation system**: cosine and distance similarity, preference-based scoring, and item-item collaborative filtering
* An interactive **Streamlit** app

**Domain:** E-Commerce Analytics & Intelligent Product Recommendation Systems
**Stack:** Python, Pandas, NumPy, scikit-learn, SciPy, Matplotlib, Seaborn, Plotly, Streamlit

**Live app:** _add your Streamlit link here after deploying_

---

## Results at a glance

| Component | Method | Result |
|---|---|---|
| Data cleaning | Exact price identity, conditional-mode imputation, validation | 12,227 missing cells handled · 4,635 prices **recovered exactly** · 0 nulls and 0 duplicates left |
| Customer segmentation | K-Means k = 4 on price paid, rating, aspect average | Premium · Satisfied 17.5% · Premium · Dissatisfied 18.8% · Budget/Mid · Satisfied 30.0% · Budget/Mid · Dissatisfied 33.8% (silhouette 0.35) |
| Product segmentation | K-Means k = 4 on the price structure of 22 models | Budget (Realme, Xiaomi, Motorola) · Mid-range (OnePlus, Google) · Upper mid-range (Samsung) · Premium (Apple) (silhouette 0.79) |
| Similar-phone recommender | Euclidean-distance similarity (cosine compared) | **100%** of top-3 recommendations in the same segment (cosine 80%, random 24%) · mean price gap $23 (random $274) |
| Preference recommender | Budget + aspect priorities + country/age group | 100% of recommendations within budget for every test profile |
| Collaborative filtering | Item-item adjusted cosine on customer groups × phones | Held-out RMSE 0.355 vs 0.342 item mean. It does not beat the baseline, because there is no group-level taste signal (reported honestly). |

### Main insights

1. **The market is four price tiers.** Budget phones carry 43% of review volume and give 2.6× more rating per dollar than Premium.
2. **Paying more does not buy satisfaction.** Price vs rating r = 0.001 (p = 0.74), and all aspect ratings are flat across price.
3. **Product quality differences are not significant.** The best and worst model differ by only 0.08★ (ANOVA p = 0.78).
4. **About 53% of buyers are dissatisfied (≈2.1★) in every price tier.** The *Premium · Dissatisfied* segment (≈$1,040, 18.8%) is the priority to win back.
5. **Preferences are global.** Brand choice and sentiment do not vary by country, age group, source or verified status.

Full details are in the reports:

* [reports/EDA_Report.md](reports/EDA_Report.md)
* [reports/Cluster_Analysis_Report.md](reports/Cluster_Analysis_Report.md)
* [reports/Recommendation_System_Report.md](reports/Recommendation_System_Report.md)

## Project structure

```
├── Mobile Reviews Sentiment null.csv          raw data (as delivered)
├── run_pipeline.py                            one command: clean → features → clustering → EDA → recommender → save
├── app.py                                     Streamlit app (Recommend · Segmentation · Dashboard · Insights · Model performance)
├── requirements.txt / requirements-dev.txt
├── src/
│   ├── config.py                              paths, column groups, palette
│   ├── preprocessing.py                       CSV → DataFrame, cleaning, identity-based recovery, validated imputation
│   ├── features.py                            engineered features, encoding, product catalog
│   ├── eda.py                                 EDA tables, statistical tests, figures
│   ├── segmentation.py                        K-Means (k selection, customer and product segments, profiles, PCA)
│   ├── recommender.py                         content-based, preference-based and collaborative-filtering recommenders + validation
│   ├── report_figures.py                      cluster and recommendation figures
│   └── plotting.py                            shared chart style
├── notebooks/
│   ├── Mobile_Segmentation_Recommendation.ipynb   executed walk-through notebook
│   └── build_notebook.py                          regenerates the notebook
├── data/processed/
│   ├── cleaned_mobile_reviews.csv             cleaned dataset (0 nulls)
│   ├── feature_engineered_mobile_reviews.csv  + engineered features, label codes and one-hot brand/country
│   ├── segmented_mobile_reviews.csv.gz        + customer and product segment (used by the app)
│   └── product_catalog.csv                    one row per model with its product segment
├── models/
│   ├── kmeans_customer_segments.joblib        StandardScaler + KMeans pipeline
│   ├── kmeans_product_segments.joblib
│   ├── similarity_euclidean.csv / similarity_cosine.csv / similarity_item_cf.csv
│   └── model_metadata.json                    features, metrics, segment names, evaluation results
└── reports/
    ├── EDA_Report.md                          key findings and patterns
    ├── Cluster_Analysis_Report.md             segment interpretation and business actions
    ├── Recommendation_System_Report.md        recommender design and validation
    ├── *_k_selection.csv, *_segment_profile.csv, recommender_*.csv, recommendation_examples.csv
    ├── eda_tables/*.csv, statistical_tests.json
    └── figures/*.png
```

## How to run

```bash
pip install -r requirements-dev.txt     # requirements.txt alone is enough for the app
python run_pipeline.py                  # ~1 min: writes processed data, models, tables and figures
streamlit run app.py                    # opens http://localhost:8501
```

To rebuild and execute the notebook:

```bash
python notebooks/build_notebook.py
python -m nbconvert --to notebook --execute --inplace notebooks/Mobile_Segmentation_Recommendation.ipynb
```

## Deploy for a permanent public link (Streamlit Community Cloud)

1. Push this folder to a public GitHub repository. Every file is under 25 MB, so the GitHub web upload works.
2. Sign in at https://share.streamlit.io with GitHub and click **Create app**. Choose the repo, branch `main` and main file `app.py`.
3. Under **Advanced settings**, choose **Python 3.13**. `requirements.txt` pins the exact library versions.
4. Click **Deploy**. The app gets a permanent `https://<name>.streamlit.app` URL.

Free apps sleep after a period without visitors. When you open a sleeping app, click **"Yes, get this app back up"** and it wakes in about 30–60 s. The URL never changes.

---

## Approach

### 1. Data collection
`pd.read_csv` loads the raw file into a DataFrame of 50,000 rows × 22 columns. The structure, columns and data types are profiled in the notebook.

### 2. Data preprocessing
* **Text standardisation and duplicate checks:** exact rows, repeated `review_id`, and the same content under a new id. None were found.
* **Type conversion:**
  * `review_date` → datetime
  * `price_local` text such as `₹27996.73` or `$1,010.68 ` → numeric `price_local_value`
  * booleans and integers
* **Validation:** brand↔model and country↔currency↔rate mappings, rating range 1–5, age range. No conflicts were found.
* **Missing values:**
  * **`price_usd` / `price_local` (≈2,450 each):** recovered exactly through `price_local = price_usd × exchange_rate_to_usd`, which holds on 100% of complete rows. The 123 rows with both prices missing get the model's median price.
  * **`rating` (2,453):** filled with the most frequent rating among reviews with the same aspect average and sentiment. This is 77.8% exact on held-out rows, versus 28.1% for the overall mode.
  * **`sentiment` (2,445):** filled with the most frequent sentiment for the same rating and aspect average. Held-out accuracy is 74.5%, versus 56.1% for the overall mode.
  * **`source` (2,448):** set to `Unknown`.
* Every imputed value is flagged (`price_imputed`, `rating_imputed`, `sentiment_imputed`).

### 3. Feature engineering, encoding and scaling
* **New features:**
  * `aspect_avg`, `rating_gap`
  * `sentiment_score` (−1 / 0 / 1), `is_positive`
  * `engagement_score` = helpful votes × 1.5 when the purchase is verified
  * `value_score` = rating per $100
  * `price_band`, `age_group`, review year and month
* **Encoding:** label codes for brand, model, country and source, plus one-hot brand and country columns.
* **Product catalog:** per-model price statistics, ratings, aspect means, positive share and a Bayesian-average rating.
* **Scaling:** `StandardScaler` inside each saved K-Means pipeline.

### 4. EDA
* Distribution across brands and countries
* Top- and low-rated products
* Price vs ratings vs specifications
* Correlations and monthly trends
* Customer preference patterns
* ANOVA, Kruskal-Wallis, χ², Pearson / Spearman and Welch t-tests

See the [EDA report](reports/EDA_Report.md).

### 5. Clustering (K-Means)
* **Customer level:** price paid, rating and aspect average. k was chosen with the elbow, silhouette, Davies-Bouldin and Calinski-Harabasz scores (k = 2–10). k = 4 is the smallest k that separates both price tier and satisfaction. Adding one-hot brand and country lowered silhouette from 0.35 to 0.14, so those dummies were left out.
* **Product level:** price structure of the 22 models gives 4 tiers. Rating features vary by less than 2.1% across models, so they would only add noise.
* Visualised with scatter plots, 2-D PCA, segment profiles and brand mix.

See the [cluster analysis report](reports/Cluster_Analysis_Report.md).

### 6. Recommendation system
* **Similar phones:** product vectors compared with cosine and Euclidean similarity. Euclidean was validated as better and is the default.
* **Find my phone:** budget in local currency, aspect priorities, country and age group. The score combines priority-aspect ratings from similar buyers, collaborative filtering, positive share and budget fit.
* **Collaborative filtering basics:** item-item adjusted cosine on a 40 customer-group × 22 phone rating matrix, evaluated on held-out data.

See the [recommendation report](reports/Recommendation_System_Report.md).

### 7. Streamlit application
* **🔎 Recommend:**
  * *Similar to a phone:* choose the metric and whether to show other brands only; see why each phone was recommended and its price position.
  * *Find my phone:* enter preferences and see a score breakdown chart.
* **🧩 Segmentation:** customer segment cards, scatter plot and brand mix; product tiers with price ranges; and *Which segment am I?*, which uses the saved K-Means pipeline and warns when the input is outside the training range.
* **📊 Dashboard:** brand, country, source and date filters; KPIs; Plotly charts; top- and bottom-rated product tables.
* **💡 Insights:** findings computed live from the data (segments, price vs performance, high and low performers, customer segments, preference patterns) plus business recommendations.
* **🧪 Model performance:** k-selection curves, clustering metrics, the encoding comparison, recommender relevance, the CF evaluation, test-profile checks and imputation accuracy.

### 8. Limitations
The dataset appears to be synthetic. Ratings do not depend on price, brand, country or age. As a result, product quality differences are not significant, and collaborative filtering has no group-level taste signal to learn from. The pipeline measures and reports this instead of hiding it.
