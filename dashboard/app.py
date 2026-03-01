import streamlit as st
import pandas as pd
import joblib
import sys
from pathlib import Path

# -------------------------------------------------
# Fix Python path so Streamlit Cloud can find /src
# -------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# Now safe to import
from src.anomaly_detection import detect_anomalies, explain_anomalies

# -------------------------------------------------
# Paths
# -------------------------------------------------
DATA_PATH = ROOT / "data" / "bioreactor_synthetic_1000rows.csv"
MODEL_PATH = ROOT / "src" / "feed_model.joblib"

# -------------------------------------------------
# Load data and model (with safety checks)
# -------------------------------------------------
@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        st.error(f"Data file not found: {DATA_PATH}")
        st.stop()
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

df = load_data()
model = load_model()

# -------------------------------------------------
# Features used for feed prediction
# -------------------------------------------------
features = [
    "time_hr",
    "vcd_e6_per_ml",
    "glucose_g_per_L",
    "lactate_g_per_L",
    "ph",
    "do_pct",
    "temperature_C",
    "agitation_rpm",
    "airflow_slpm"
]

# Check if required columns exist
missing_cols = [col for col in features if col not in df.columns]
if missing_cols:
    st.error(f"Missing required columns in dataset: {missing_cols}")
    st.stop()

# -------------------------------------------------
# Feed Recommendation Logic
# -------------------------------------------------
def recommend_feed(row, model):
    x = row[features].values.reshape(1, -1)
    pred_feed = model.predict(x)[0]

    messages = []

    if row["glucose_g_per_L"] < 2:
        messages.append("Glucose is low; consider increasing feed to avoid depletion.")
    if row["do_pct"] < 30:
        messages.append("DO is low; check aeration and agitation before increasing feed.")
    if row["vcd_e6_per_ml"] > 8:
        messages.append("High cell density; ensure sufficient nutrients and oxygen.")

    if not messages:
        messages.append("Process conditions are stable; maintain current feed strategy.")

    return float(pred_feed), " ".join(messages)

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.title("AI-Generated Bioreactor Optimization Assistant")

if "batch_id" not in df.columns:
    st.error("Column 'batch_id' not found in dataset.")
    st.stop()

batch_ids = df["batch_id"].unique()
selected_batch = st.selectbox("Select batch", batch_ids)

df_batch = df[df["batch_id"] == selected_batch].sort_values("time_hr")

# -------------------------------------------------
# Time-Series Plots
# -------------------------------------------------
st.subheader("Bioreactor Time-Series")

df_plot = df_batch.set_index("time_hr")

st.line_chart(df_plot[["vcd_e6_per_ml", "glucose_g_per_L", "lactate_g_per_L"]])
st.line_chart(df_plot[["do_pct", "temperature_C"]])

# -------------------------------------------------
# Feed Recommendations & Anomalies
# -------------------------------------------------
st.subheader("Feed Recommendations & Anomalies")

df_batch_flags = detect_anomalies(df_batch)

recs = []
for _, row in df_batch_flags.iterrows():
    pred_feed, feed_msg = recommend_feed(row, model)
    anomaly_msg = explain_anomalies(row)

    recs.append({
        "time_hr": row["time_hr"],
        "predicted_feed_ml_per_hr": round(pred_feed, 2),
        "feed_recommendation": feed_msg,
        "anomaly_explanation": anomaly_msg
    })

df_recs = pd.DataFrame(recs)
st.dataframe(df_recs)