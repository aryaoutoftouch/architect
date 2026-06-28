"""
Architect — quickstart examples
Run from the repository root: python examples/quickstart.py
"""

import architect as ar

# 1. Basic usage
print("=== Basic usage ===")
asset = ar.create(n=1000, k=3, random_state=42).generate()
print(asset)
print(asset.series.head(), "\n")

# 2. Custom regime parameters
print("=== Custom regimes ===")
asset = ar.create(
    n=500,
    k=2,
    returns=[-0.001, 0.002],
    volatilities=[0.01, 0.025],
    persistence=0.95,
    random_state=0,
).generate()
print("Regimes:", asset.generated_regimes, "\n")

# 3. Heavy-tailed noise
print("=== Student-t noise ===")
asset = ar.create(n=2000, k=3, noise="student_t", df=3, scale=0.5, random_state=7).generate()
print(f"Skewness: {asset.summary['skewness']:.4f}  Excess kurtosis: {asset.summary['excess_kurtosis']:.4f}\n")

# 4. Explicit transition matrix
print("=== Explicit transition matrix ===")
tm = [
    [0.97, 0.02, 0.01],
    [0.05, 0.90, 0.05],
    [0.01, 0.04, 0.95],
]
asset = ar.create(n=1000, k=3, transition_matrix=tm, random_state=1).generate()
print("Transition matrix:\n", asset.transition_matrix, "\n")

# 5. Per-regime statistics
print("=== Per-regime statistics ===")
asset = ar.create(n=2000, k=3, random_state=5).generate()
for i, stats in asset.summary["regime_statistics"].items():
    freq = asset.summary["regime_frequencies"][i]
    dur  = asset.summary["regime_durations"][i]
    print(
        f"Regime {i}: freq={freq:.2%}  mean={stats['mean']:.5f}  "
        f"std={stats['std']:.5f}  avg_run={dur['avg']:.1f} steps"
    )

# 6. Reset and re-run
print("\n=== Reset and re-run ===")
asset = ar.create(n=500, k=2, random_state=99)
asset.generate()
first_mean = asset.summary["mean"]
asset.reset().generate()
second_mean = asset.summary["mean"]
print(f"Run 1 mean: {first_mean:.6f}  Run 2 mean: {second_mean:.6f}")
print("(Different because reset() does not fix a new seed — same config, new draw)")

# 7. Export
# asset.to_csv("prices.csv")
# asset.to_parquet("prices.parquet")
print("\nDone.")
