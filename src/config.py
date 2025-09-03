# src/config.py

# --- Data & Output Paths ---
DATA_DIR = "data/"
MODEL_DIR = "models/"
REPORTS_DIR = "reports/"
FIGURES_DIR = f"{REPORTS_DIR}figures/"
OUTPUT_DIR = "output/"

# Input data
DATA_PATH = f"{DATA_DIR}HillstromEmailMarketing.csv"

# Model output
TWO_MODELS_PATH = f"{MODEL_DIR}uplift_two_models.joblib"

# Report outputs
UPLIFT_CURVE_PATH = f"{FIGURES_DIR}uplift_curve.png"
PROFIT_CURVE_PATH = f"{FIGURES_DIR}profit_curve.png"

# --- Business Parameters ---
# These can be adjusted for cost-benefit analysis
COST_OF_TREATMENT = 0.15  # Cost to send one promotional email (e.g., in dollars)
VALUE_OF_CONVERSION = 5.00  # Profit from a single successful conversion

# --- Model & Feature Parameters ---
CAMPAIGN_SEGMENT = 'Womens E-Mail'
CONTROL_SEGMENT = 'No E-Mail'
EXCLUDE_SEGMENT = 'Mens E-Mail'

OUTCOME = 'visit'
TREATMENT = 'treatment'

FEATURES = [
    'recency', 'history', 'mens', 'womens', 'newbie',
]
CATEGORICAL_FEATURES = ['channel', 'zip_code']