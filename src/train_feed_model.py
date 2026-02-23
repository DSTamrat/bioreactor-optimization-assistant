import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# -----------------------------
# Load dataset
# -----------------------------
data_path = Path(__file__).resolve().parents[1] / "data" / "bioreactor_synthetic_1000rows.csv"
df = pd.read_csv(data_path)

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
target = "feed_rate_ml_per_hr"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

print("Train R²:", model.score(X_train, y_train))
print("Test R²:", model.score(X_test, y_test))

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

# Quick test
sample_row = df.sample(1, random_state=1).iloc[0]
pred_feed, explanation = recommend_feed(sample_row, model)
print("\nSample Prediction:")
print("Predicted feed rate (mL/hr):", round(pred_feed, 2))
print("Explanation:", explanation)

# Save model
models_dir = Path(__file__).resolve().parents[1] / "src" / "models"
models_dir.mkdir(exist_ok=True)
model_path = models_dir / "feed_model.joblib"
joblib.dump(model, model_path)
print(f"\nModel saved to: {model_path}")
