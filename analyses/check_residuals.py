
from pathlib import Path
import json
import os

def load_metrics(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return {}

results_dir = Path("../results/strong_scaling")
latest_files = {}

# Find latest file for each J for fixed-point
for metrics_path in results_dir.rglob("metrics.json"):
    data = load_metrics(metrics_path)
    if data.get("algorithm") == "fixed-point" and data.get("mesh_size") == 128 and data.get("wavenumber") == 16.0:
        J = data.get("subdomains")
        mtime = metrics_path.stat().st_mtime
        if J not in latest_files or mtime > latest_files[J][0]:
            latest_files[J] = (mtime, metrics_path, data)

print("Latest Fixed-Point Strong Scaling Results:")
for J in sorted(latest_files.keys()):
    time, path, data = latest_files[J]
    resid = data.get("final_residual")
    iters = data.get("iterations")
    print(f"J={J}: Residual={resid}, Iterations={iters}, Path={path}")
    
    # Check history last element
    hist = data.get("residual_history", [])
    if hist:
        print(f"     History Last: {hist[-1]}")
