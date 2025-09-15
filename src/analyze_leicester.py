import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
DATA_DIR = "data"
OUTPUT_DIR = "outputs/figures"

# Make sure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load cleaned datasets
leicester = pd.read_csv(os.path.join(DATA_DIR, "leicester_xg_clean.csv"))
xpts = pd.read_csv(os.path.join(DATA_DIR, "epl_2015_16_xpts_clean.csv"))

# --- 1. Leicester: Actual Goals vs xG (cumulative) ---
leicester["Cum_Goals"] = leicester["LeicesterGoals"].cumsum()
leicester["Cum_xG"] = leicester["Leicester_xG"].cumsum()

plt.figure(figsize=(10,6))
plt.plot(leicester["Date"], leicester["Cum_Goals"], label="Actual Goals", linewidth=2)
plt.plot(leicester["Date"], leicester["Cum_xG"], label="Expected Goals (xG)", linewidth=2, linestyle="--")
plt.title("Leicester City 2015/16: Actual vs Expected Goals")
plt.xlabel("Date")
plt.ylabel("Cumulative Goals")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "leicester_goals_vs_xg.png"))
plt.close()

# --- Add Points column to Leicester data ---
def result_to_points(r):
    if r.lower() == "w":
        return 3
    elif r.lower() == "d":
        return 1
    else:
        return 0

leicester["Points"] = leicester["Result"].apply(result_to_points)

# --- Cumulative Points vs xPTS ---
leicester["Cum_Points"] = leicester["Points"].cumsum()
plt.figure(figsize=(10,6))
plt.plot(leicester["Date"], leicester["Cum_Points"], label="Actual Points", linewidth=2, color="blue")
plt.title("Leicester City 2015/16: Actual Points Progression")
plt.xlabel("Date")
plt.ylabel("Cumulative Points")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, "leicester_points.png"))
plt.close()



# --- 3. League-wide: Actual vs Expected Points (season totals) ---
plt.figure(figsize=(12,6))
plt.scatter(xpts["xPTS"], xpts["Wins"]*3 + xpts["Draws"], s=100, alpha=0.7)

for i, row in xpts.iterrows():
    plt.text(row["xPTS"]+0.2, row["Wins"]*3 + row["Draws"], row["Team"],
             fontsize=9, weight="bold" if row["Team"]=="Leicester" else "normal",
             color="red" if row["Team"]=="Leicester" else "black")

plt.xlabel("Expected Points (xPTS)")
plt.ylabel("Actual Points")
plt.title("Premier League 2015/16: Actual Points vs Expected Points")
plt.axline((0,0), slope=1, color="gray", linestyle="--")  # y=x reference
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(OUTPUT_DIR, "league_pts_vs_xpts.png"))
plt.close()

print("✅ Figures saved in", OUTPUT_DIR)


# --- 4. League-wide Over/Underperformance ---
xpts["PTS"] = xpts["Wins"]*3 + xpts["Draws"]
xpts["Overperf"] = xpts["PTS"] - xpts["xPTS"]
xpts_sorted = xpts.sort_values("Overperf", ascending=False)

plt.figure(figsize=(12,6))
plt.bar(xpts_sorted["Team"], xpts_sorted["Overperf"],
        color=(xpts_sorted["Team"]=="Leicester").map({True:"red", False:"gray"}))
plt.axhline(0, color="black", linewidth=1)
plt.title("Premier League 2015/16: Over/Underperformance (PTS - xPTS)")
plt.ylabel("Points Over Expected")
plt.xticks(rotation=60, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "league_overperformance.png"))
plt.close()