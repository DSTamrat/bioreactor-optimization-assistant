import pandas as pd

def detect_anomalies(df_batch: pd.DataFrame) -> pd.DataFrame:
    df_batch = df_batch.sort_values("time_hr").copy()
    df_batch["do_drop"] = df_batch["do_pct"].diff() < -15
    df_batch["lactate_spike"] = df_batch["lactate_g_per_L"].diff() > 0.8
    df_batch["ph_drift"] = df_batch["ph"].diff().abs() > 0.1
    return df_batch

def explain_anomalies(row) -> str:
    msgs = []
    if row.get("do_drop", False):
        msgs.append("DO dropped rapidly; likely due to increased oxygen demand or insufficient aeration.")
    if row.get("lactate_spike", False):
        msgs.append("Lactate spiked; this may indicate overfeeding or metabolic stress.")
    if row.get("ph_drift", False):
        msgs.append("pH drift observed; possible CO₂ accumulation or buffer issues.")
    if not msgs:
        msgs.append("No significant anomalies detected at this time point.")
    return " ".join(msgs)
