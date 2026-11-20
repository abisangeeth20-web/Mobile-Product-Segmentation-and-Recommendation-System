"""Central configuration: paths, column groups and shared constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "Mobile Reviews Sentiment null.csv"          # raw data (as delivered)
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "eda_tables"

CLEANED_CSV = PROCESSED_DIR / "cleaned_mobile_reviews.csv"
FEATURED_CSV = PROCESSED_DIR / "feature_engineered_mobile_reviews.csv"
SEGMENTED_GZ = PROCESSED_DIR / "segmented_mobile_reviews.csv.gz"   # what the app reads
PRODUCTS_CSV = PROCESSED_DIR / "product_catalog.csv"               # one row per phone model

DATE_FORMAT = "%Y-%m-%d"
RAW_DATE_FORMAT = "%m/%d/%Y"
RANDOM_STATE = 42
N_CLUSTERS = 4            # spec: "assume and create 4 clusters" - confirmed by elbow / silhouette (see report)
K_RANGE = range(2, 11)

ASPECTS = ["battery_life_rating", "camera_rating", "performance_rating", "design_rating", "display_rating"]
ASPECT_LABELS = {"battery_life_rating": "Battery", "camera_rating": "Camera", "performance_rating": "Performance",
                 "design_rating": "Design", "display_rating": "Display"}
CATEGORICAL_COLS = ["brand", "model", "country", "currency", "language", "source", "sentiment"]

# Customer (review-level) clustering inputs: what was paid and how satisfied the buyer is
CUSTOMER_CLUSTER_FEATURES = ["price_usd", "rating", "aspect_avg"]
# Product (model-level) clustering inputs
PRODUCT_CLUSTER_FEATURES = ["avg_price_usd", "min_price_usd", "max_price_usd", "avg_rating", "positive_share"]

SENTIMENT_SCORE = {"Negative": -1, "Neutral": 0, "Positive": 1}
AGE_BINS = [17, 24, 34, 44, 54, 120]
AGE_LABELS = ["18-24", "25-34", "35-44", "45-54", "55+"]

# Chart palette (validated reference palette - categorical slots in fixed order, never cycled)
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BRANDS = ["Apple", "Samsung", "Google", "OnePlus", "Motorola", "Xiaomi", "Realme"]
BRAND_COLORS = dict(zip(BRANDS, CATEGORICAL))
SENTIMENT_COLORS = {"Positive": "#2a78d6", "Neutral": "#c3c2b7", "Negative": "#e34948"}   # diverging pair + gray
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
       "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb"}
