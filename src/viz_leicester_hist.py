import pandas as pd
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "outputsx"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sims_df = pd.read_csv(os.path.join(OUTPUT_DIR, "poisson_simulations.csv"))

# Leicester distribution
leicester_points = sims_df["Leicester"]

plt.figure(figsize=(10,6))
plt.hist(leicester_points, bins=30, color="blue", alpha=0.7, edgecolor="black")
plt.axvline(leicester_points.mean(), color="red", linestyle="--", label=f"Mean: {leicester_points.mean():.1f}")
plt.title("Leicester City Simulated Points Distribution (2015/16)")
plt.xlabel("Points")
plt.ylabel("Frequency")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "leicester_points_distribution.png"))
plt.close()

print("✅ Saved Leicester points distribution → output/leicester_points_distribution.png")
