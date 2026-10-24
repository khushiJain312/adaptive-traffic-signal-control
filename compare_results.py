import pandas as pd
import os

# ============================================================
# FILE PATHS
# ============================================================

RL_FILE = "results/rl_evaluation.csv"
BASELINE_FILE = "results/baseline_results.csv"

# ============================================================
# LOAD RESULTS
# ============================================================

rl = pd.read_csv(RL_FILE)
baseline = pd.read_csv(BASELINE_FILE)

print("\n" + "=" * 60)
print("RL VS FIXED-TIME COMPARISON")
print("=" * 60)

# ============================================================
# AVERAGE METRICS
# ============================================================

rl_waiting = rl["Average_Waiting_Time"].mean()
baseline_waiting = baseline["Average_Waiting_Time"].mean()

rl_queue = rl["Average_Queue"].mean()
baseline_queue = baseline["Average_Queue"].mean()

rl_vehicles = rl["Average_Vehicles"].mean()
baseline_vehicles = baseline["Average_Vehicles"].mean()

# IMPORTANT:
# RL CSV uses "Throughput"
# Baseline CSV uses "Average_Throughput"

rl_throughput = rl["Throughput"].mean()
baseline_throughput = baseline["Average_Throughput"].mean()

# ============================================================
# IMPROVEMENT CALCULATIONS
# ============================================================

waiting_improvement = (
    (baseline_waiting - rl_waiting) / baseline_waiting
) * 100

queue_improvement = (
    (baseline_queue - rl_queue) / baseline_queue
) * 100

vehicles_improvement = (
    (baseline_vehicles - rl_vehicles) / baseline_vehicles
) * 100

throughput_improvement = (
    (rl_throughput - baseline_throughput) / baseline_throughput
) * 100

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nAVERAGE RESULTS")
print("-" * 60)

print(f"Waiting Time:")
print(f"  Fixed-Time : {baseline_waiting:.3f}")
print(f"  RL         : {rl_waiting:.3f}")
print(f"  Improvement: {waiting_improvement:.2f}%")

print(f"\nQueue:")
print(f"  Fixed-Time : {baseline_queue:.3f}")
print(f"  RL         : {rl_queue:.3f}")
print(f"  Improvement: {queue_improvement:.2f}%")

print(f"\nAverage Vehicles:")
print(f"  Fixed-Time : {baseline_vehicles:.3f}")
print(f"  RL         : {rl_vehicles:.3f}")
print(f"  Improvement: {vehicles_improvement:.2f}%")

print(f"\nThroughput:")
print(f"  Fixed-Time : {baseline_throughput:.3f}")
print(f"  RL         : {rl_throughput:.3f}")
print(f"  Improvement: {throughput_improvement:.2f}%")

print("\n" + "=" * 60)
print("COMPARISON COMPLETE")
print("=" * 60)

# ============================================================
# SAVE SUMMARY
# ============================================================

comparison = pd.DataFrame({
    "Metric": [
        "Average Waiting Time",
        "Average Queue",
        "Average Vehicles",
        "Throughput"
    ],
    "Fixed_Time": [
        baseline_waiting,
        baseline_queue,
        baseline_vehicles,
        baseline_throughput
    ],
    "RL": [
        rl_waiting,
        rl_queue,
        rl_vehicles,
        rl_throughput
    ],
    "Improvement_Percent": [
        waiting_improvement,
        queue_improvement,
        vehicles_improvement,
        throughput_improvement
    ]
})

output_file = "results/comparison_results.csv"
comparison.to_csv(output_file, index=False)

print(f"\nComparison saved to:")
print(os.path.abspath(output_file))