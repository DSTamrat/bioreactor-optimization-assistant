import numpy as np
import pandas as pd
from pathlib import Path

def generate_bioreactor_data(n_batches=20, rows_per_batch=50, seed=42):
    np.random.seed(seed)

    batch_ids = [f"B{str(i).zfill(3)}" for i in range(1, n_batches + 1)]
    data = []

    for b in batch_ids:
        times = np.linspace(0, 120, rows_per_batch)  # 0–120 hours
        
        vcd = 0.3 + 10 / (1 + np.exp(-0.08 * (times - 50))) + np.random.normal(0, 0.3, rows_per_batch)
        glucose = 6.5 - 0.04 * times + np.random.normal(0, 0.2, rows_per_batch)
        glucose = np.clip(glucose, 0.1, None)
        
        lactate = 0.1 + 0.03 * times + np.random.normal(0, 0.1, rows_per_batch)
        lactate = np.clip(lactate, 0.05, None)
        
        ph = 7.1 - 0.003 * times + np.random.normal(0, 0.03, rows_per_batch)
        
        do = 95 - 0.4 * times + np.random.normal(0, 2, rows_per_batch)
        do = np.clip(do, 5, 100)
        
        temp = 36.8 + np.random.normal(0, 0.1, rows_per_batch)
        
        agitation = 150 + 0.5 * times + np.random.normal(0, 5, rows_per_batch)
        
        airflow = 0.5 + 0.02 * times + np.random.normal(0, 0.05, rows_per_batch)
        
        base_feed = 5 + 0.8 * (vcd - vcd.min()) + 3 * (glucose < 2).astype(int)
        feed_rate = base_feed + np.random.normal(0, 1, rows_per_batch)
        feed_rate = np.clip(feed_rate, 0, None)
        
        for i in range(rows_per_batch):
            data.append({
                "batch_id": b,
                "time_hr": times[i],
                "vcd_e6_per_ml": vcd[i],
                "glucose_g_per_L": glucose[i],
                "lactate_g_per_L": lactate[i],
                "ph": ph[i],
                "do_pct": do[i],
                "temperature_C": temp[i],
                "agitation_rpm": agitation[i],
                "airflow_slpm": airflow[i],
                "feed_rate_ml_per_hr": feed_rate[i]
            })

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    df = generate_bioreactor_data()
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(exist_ok=True)
    out_path = data_dir / "bioreactor_synthetic_1000rows.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved synthetic dataset to: {out_path}")
