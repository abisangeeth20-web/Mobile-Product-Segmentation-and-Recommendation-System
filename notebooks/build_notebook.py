"""Generates notebooks/Mobile_Segmentation_Recommendation.ipynb (walk-through of the pipeline).

python notebooks/build_notebook.py
python -m nbconvert --to notebook --execute --inplace notebooks/Mobile_Segmentation_Recommendation.ipynb
"""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# Mobile Product Segmentation & Recommendation System\n"
       "Global Mobile Reviews dataset (50,000 reviews · 7 brands · 22 models · 8 countries) - end-to-end "
       "walk-through.\n\nAll logic lives in the `src/` package, so this notebook, `run_pipeline.py` and the "
       "Streamlit app share exactly the same code."),
    code("import sys, json, warnings\nsys.path.insert(0, '..')\nwarnings.filterwarnings('ignore')\n"
         "import pandas as pd, numpy as np, joblib\nfrom IPython.display import Image, display\n"
         "from src import config as C, preprocessing, features, segmentation, recommender, eda\n"
         "pd.set_option('display.max_columns', 40); pd.set_option('display.width', 200)\n"
         "show = lambda f: display(Image(str(C.FIGURES_DIR / f)))"),

    md("## 1. Data collection - CSV ➜ DataFrame"),
    code("raw = preprocessing.load_raw()\nprint(raw.shape)\nraw.head()"),
    code("raw.info()"),
    code("nulls = raw.isna().sum().to_frame('nulls')\nnulls['%'] = (nulls.nulls / len(raw) * 100).round(2)\n"
         "nulls[nulls.nulls > 0].T"),
    code("show('eda_00_missing_values.png')"),

    md("### Discovering the data's structure\n"
       "* **price_local = price_usd × exchange_rate_to_usd** holds on 100% of complete rows → a missing price is "
       "recovered exactly from the other price.\n"
       "* `price_local` is text with a currency symbol (`₹27996.73`, `$1,010.68 `) → parsed to a number.\n"
       "* `rating` follows the mean of the five aspect ratings, and `sentiment` follows `rating`."),
    code("sym, val = preprocessing.parse_local_price(raw.price_local)\nok = raw.price_usd.notna() & val.notna()\n"
         "err = (val[ok] - raw.price_usd[ok] * raw.exchange_rate_to_usd[ok]).abs()\n"
         "print('identity holds:', (err < 0.01).mean(), '| max error:', err.max())\n"
         "pd.crosstab(raw.rating, raw.sentiment)"),

    md("## 2. Data cleaning & preprocessing\n"
       "1. Text standardisation, duplicate removal (exact rows, repeated `review_id`, same content under a new id)\n"
       "2. Type conversion: dates, booleans, integers; `price_local` parsed to `price_local_value`\n"
       "3. Validation: brand↔model and country↔currency mappings, rating ranges, age range, price identity\n"
       "4. Prices: exact recovery via the identity; 123 rows with both prices missing → model median\n"
       "5. `rating` → most frequent rating among reviews with the same aspect average and sentiment; `sentiment` "
       "→ most frequent sentiment for the same rating and aspect average (accuracy measured on held-out rows)\n"
       "6. `source` → `Unknown` (it has no relationship with any other column)"),
    code("clean, log = preprocessing.clean(raw)\nfor k, v in log.items():\n"
         "    if k != 'raw_nulls': print(f'{k:<36} {v}')"),
    code("print('Remaining nulls:', clean.isna().sum().sum(), '| duplicates:', clean.duplicated().sum())\n"
         "clean.dtypes.to_frame('dtype').T"),

    md("## 3. Feature engineering & encoding\n"
       "* `aspect_avg` (mean of 5 aspect ratings), `rating_gap`, `sentiment_score` (-1/0/1), `is_positive`\n"
       "* `engagement_score` = helpful votes × 1.5 if verified purchase\n"
       "* `value_score` = rating per $100 paid (price vs performance)\n"
       "* `price_band`, `age_group`, review year / month\n"
       "* Encoding: label codes for brand / model / country / source and one-hot brand + country\n"
       "* Scaling: `StandardScaler` inside each K-Means pipeline"),
    code("feat = features.add_features(clean)\nenc = features.encode_categoricals(feat)\nprint(enc.shape)\n"
         "enc[['brand','model','country','brand_code','model_code','country_code','brand_Apple','country_India',"
         "'aspect_avg','engagement_score','value_score','price_band']].head()"),
    code("products = features.product_catalog(feat)\nproducts.head()"),

    md("## 4. Exploratory data analysis"),
    code("tables = eda.run_eda(feat, products)\ntables['brand_summary']"),
    code("show('eda_01_brand_country_distribution.png'); show('eda_02_price_by_brand.png')"),
    code("show('eda_03_rating_sentiment.png'); show('eda_04_top_low_products.png')"),
    code("display(tables['top5_products']); display(tables['bottom5_products'])"),
    md("### Price vs ratings vs specifications"),
    code("show('eda_05_price_vs_rating.png'); show('eda_06_correlation.png')"),
    code("show('eda_07_monthly_trend.png'); show('eda_08_customer_preferences.png'); "
         "show('eda_09_engagement_source.png')"),
    md("### Statistical tests (α = 0.05)"),
    code("pd.DataFrame(tables['tests']).T"),

    md("## 5. Clustering - customer (review) level\n"
       "K-Means on standardised **price paid, overall rating and aspect average**. k chosen with elbow, "
       "silhouette, Davies-Bouldin and Calinski-Harabasz for k = 2..10."),
    code("pd.read_csv(C.REPORTS_DIR / 'customer_k_selection.csv', index_col=0).round(3)"),
    code("show('seg_01_k_selection_customer.png')"),
    code("pipe, labels, names, centers, metrics = segmentation.fit_customer_segments(feat)\n"
         "feat['customer_segment'] = labels\nprint(metrics)\ncenters"),
    code("segmentation.customer_segment_profile(feat)"),
    code("show('seg_03_customer_clusters.png'); show('seg_04_customer_segment_profiles.png'); "
         "show('seg_06_segment_brand_mix.png')"),
    md("Feature-selection check - adding one-hot brand and country (plus age and engagement) to the clustering:"),
    code("segmentation.alternative_encoded_clustering(feat)"),

    md("## 6. Clustering - product (model) level\n"
       "Rating features hardly vary across the 22 models (coefficient of variation < 2.1%), so product "
       "segments are formed on price structure."),
    code("segmentation.product_feature_variation(products)"),
    code("ppipe, plabels, pnames, pmetrics = segmentation.fit_product_segments(products)\n"
         "products['product_segment'] = plabels\nfeat['product_segment'] = feat.model.map("
         "products.set_index('model').product_segment)\nprint(pmetrics)\nsegmentation.product_segment_profile(products)"),
    code("show('seg_02_k_selection_product.png'); show('seg_05_product_segments.png')"),

    md("## 7. Recommendation system\n"
       "* **Content-based**: product vectors → cosine similarity or Euclidean-distance similarity\n"
       "* **Preference-based (hybrid)**: budget + aspect priorities + country / age group\n"
       "* **Collaborative filtering basics**: item-item adjusted cosine on a customer-group × phone matrix"),
    code("rec = recommender.ProductRecommender(products, feat)\n"
         "rec.recommend_similar('iPhone 14', 5)[['model','brand','product_segment','avg_price_usd','avg_rating',"
         "'similarity','reason']]"),
    code("rec.recommend_similar('Realme 12 Pro', 5, metric='cosine')[['model','product_segment','avg_price_usd',"
         "'similarity']]"),
    code("rec.recommend_for_user(budget_min=150, budget_max=450, priorities={'battery_life_rating': 3, "
         "'performance_rating': 2}, country='India', age_group='18-24')"),
    code("rec.similar_by_ratings('Pixel 8')"),
    md("### Validating recommendation relevance"),
    code("recommender.evaluate_content(rec)"),
    code("recommender.evaluate_cf(feat, products)"),
    code("recommender.evaluate_personas(rec)"),
    code("show('rec_01_similarity_heatmap.png'); show('rec_02_recommender_evaluation.png')"),

    md("## 8. Using the saved models"),
    code("km = joblib.load(C.MODELS_DIR / 'kmeans_customer_segments.joblib')\n"
         "meta = json.loads((C.MODELS_DIR / 'model_metadata.json').read_text())\n"
         "new = pd.DataFrame([{'price_usd': 1150, 'rating': 2, 'aspect_avg': 1.8},\n"
         "                    {'price_usd': 350, 'rating': 5, 'aspect_avg': 4.2}])\n"
         "[meta['customer_segmentation']['cluster_names'][str(i)] for i in km.predict(new)]"),
    md("See `reports/EDA_Report.md`, `reports/Cluster_Analysis_Report.md` and "
       "`reports/Recommendation_System_Report.md` for insights and recommendations."),
]
nb["cells"] = cells
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = Path(__file__).parent / "Mobile_Segmentation_Recommendation.ipynb"
nbf.write(nb, out)
print("wrote", out)
