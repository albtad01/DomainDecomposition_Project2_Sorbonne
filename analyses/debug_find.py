
from pathlib import Path
import sys
import json

# Mock load_metrics
def load_metrics(metrics_path):
    with open(metrics_path, "r") as f:
        return json.load(f)

def find_metrics_by_subdomains_strong(results_dir, algorithm, m, kappa, omega=None):
    print(f"Searching in {results_dir} for algo={algorithm}, m={m}, kappa={kappa}, omega={omega}")
    latest_by_J = {}
    count = 0 
    for metrics_path in Path(results_dir).rglob("metrics.json"):
        count += 1
        try:
            data = load_metrics(metrics_path)
            # print(f"Checking {metrics_path}: {data.get('algorithm')}, m={data.get('mesh_size')}, k={data.get('wavenumber')}, o={data.get('omega')}")
        except Exception as e:
            print(f"Error reading {metrics_path}: {e}")
            continue

        if data.get("algorithm") != algorithm:
            continue
        if data.get("mesh_size") != m:
            continue
        if float(data.get("wavenumber")) != float(kappa):
            continue
        if algorithm == "fixed-point" and omega is not None:
             if abs(data.get("omega") - omega) > 1e-9: # Safe float comparison
                continue
        
        J = data.get("subdomains")
        print(f"Found match: J={J} in {metrics_path}")
        if J is None:
            continue

        mtime = metrics_path.stat().st_mtime
        if J not in latest_by_J or mtime > latest_by_J[J][0]:
            latest_by_J[J] = (mtime, metrics_path)
    
    print(f"Total files checked: {count}")
    return {J: path for J, (_, path) in latest_by_J.items()}

results_dir = "../results/strong_scaling"
print("--- GMRES ---")
found_gmres = find_metrics_by_subdomains_strong(results_dir, "gmres", 128, 16.0)
print("GMRES found Js:", sorted(found_gmres.keys()))

print("\n--- Fixed-Point ---")
found_fp = find_metrics_by_subdomains_strong(results_dir, "fixed-point", 128, 16.0, omega=0.1)
print("Fixed-Point found Js:", sorted(found_fp.keys()))
